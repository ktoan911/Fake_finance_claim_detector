from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from database.opensearch import OpenSearchKB

from .config import LABEL_LIST, PROMPT_TEMPLATE
from .retrieval import QueryExpander, RetrievalResult, TemporalScorer

try:
    from .fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
except ImportError:
    ConfidenceAwareFusion = None  # type: ignore
    RetrievalFeatureEncoder = None  # type: ignore

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


def _parse_timestamp(value: Any) -> datetime:
    """Best-effort timestamp parser with UTC normalization."""
    now = datetime.now(timezone.utc)
    if value is None:
        return now

    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, (int, float)):
        ts = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        raw = str(value).strip()
        if not raw:
            return now
        raw = raw.replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(raw)
        except ValueError:
            try:
                ts = datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except (ValueError, OverflowError):
                return now

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _select_doc_text(source: Dict[str, Any]) -> str:
    """Pick evidence text from OpenSearch document source."""
    for key in ("text", "content", "description", "title"):
        value = source.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _truncate(text: str, max_chars: int) -> str:
    s = str(text)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokenize_for_overlap(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _is_verbatim_query(query: str) -> bool:
    normalized = _normalize_query_text(query)
    token_count = len(_tokenize_for_overlap(normalized))
    if not normalized:
        return False
    if "\n" in str(query):
        return True
    if len(normalized) >= 120:
        return True
    return token_count >= 20


def _token_overlap_ratio(query_text: str, doc_text: str) -> float:
    q_tokens = set(_tokenize_for_overlap(query_text))
    d_tokens = set(_tokenize_for_overlap(doc_text))
    if not q_tokens or not d_tokens:
        return 0.0
    return float(len(q_tokens & d_tokens) / len(q_tokens))


def _build_retrieval_features_train_compatible(
    retriever: Any, text: str, top_k: int, rrf_top_k: int = 100
) -> tuple[np.ndarray, List[str], List[RetrievalResult]]:
    """
    Same feature construction used in training:
    [score, rrf_score, recency_score, cyclicity_score] for top_k docs.
    """
    results = retriever.retrieve(text, top_k=top_k, rrf_top_k=rrf_top_k)
    features = []
    evidence_texts = []

    for r in results:
        features.append([r.score, r.rrf_score, r.recency_score, r.cyclicity_score])
        evidence_texts.append(r.text)

    if len(features) < top_k:
        features.extend([[0.0, 0.0, 0.0, 0.0]] * (top_k - len(features)))

    return np.array(features, dtype=np.float32), evidence_texts, results


class OpenSearchHybridRetriever:
    """
    Retrieval wrapper that keeps the train-time scoring pipeline, but uses
    OpenSearch for BM25 and vector retrieval.
    """

    def __init__(
        self,
        kb: OpenSearchKB,
        embedding_model: str,
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        use_query_expansion: bool = True,
        rrf_k: int = 60,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for OpenSearch vector retrieval."
            )

        self.kb = kb
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.temporal_scorer = TemporalScorer(
            alpha=alpha, lambda_decay=lambda_decay, gamma=gamma
        )
        self.query_expander = QueryExpander() if use_query_expansion else None
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = int(self.encoder.get_sentence_embedding_dimension())

        # Ensure dimension check in OpenSearch wrapper matches the active encoder.
        self.kb.embedding_dim = self.embedding_dim

    def _get_search_pool_size(self, rrf_top_k: int) -> int:
        """
        Match train-time "rank over all docs" behavior as closely as OpenSearch allows.
        """
        pool = max(rrf_top_k, 100)
        try:
            count = int(self.kb.client.count(index=self.kb.index).get("count", pool))
            if count > 0:
                # OpenSearch default result window is usually 10k.
                pool = min(count, 10000)
        except Exception as exc:
            pool = max(pool, 300)
            logger.warning(
                f"Could not fetch OpenSearch count, using pool={pool}: {exc}"
            )
        return pool

    def _doc_group_key(self, source: Dict[str, Any]) -> str:
        return str(source.get("type") or source.get("source") or "default")

    def _doc_timestamp(self, source: Dict[str, Any]) -> datetime:
        for key in ("timestamp", "published_at", "created_at", "fetched_at"):
            if key in source:
                return _parse_timestamp(source.get(key))
        return datetime.now(timezone.utc)

    def _encode_query(self, query: str) -> List[float]:
        vector = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return vector.astype(np.float32).tolist()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_temporal: bool = True,
        expand_query: bool = True,
        use_semantic: bool = True,
        rrf_top_k: int = 20,
    ) -> List[RetrievalResult]:
        debug = _env_flag("FUSION_INFERENCE_DEBUG", default=False) or _env_flag(
            "FUSION_INFERENCE_LOG_ALL", default=False
        )
        self.temporal_scorer.reference_date = datetime.now(timezone.utc)

        normalized_query = _normalize_query_text(query)
        if not normalized_query:
            return []

        verbatim_query = _is_verbatim_query(query)
        expanded_query = normalized_query
        if expand_query and self.query_expander is not None and not verbatim_query:
            expanded_query = self.query_expander.expand_query(normalized_query)

        effective_rrf_top_k = max(rrf_top_k, top_k * 10)
        search_pool_k = self._get_search_pool_size(rrf_top_k=effective_rrf_top_k)
        semantic_enabled = use_semantic and not verbatim_query
        if debug:
            logger.info(
                "[fusion_inference] retrieve"
                f" | query={query!r}"
                f" | normalized_query={normalized_query!r}"
                f" | expanded_query={expanded_query!r}"
                f" | top_k={top_k}"
                f" | rrf_top_k={effective_rrf_top_k}"
                f" | search_pool_k={search_pool_k}"
                f" | verbatim_query={verbatim_query}"
                f" | use_temporal={use_temporal}"
                f" | use_semantic={use_semantic}"
                f" | semantic_enabled={semantic_enabled}"
            )

        bm25_hits = self.kb.search_bm25(
            query=expanded_query,
            k=search_pool_k,
            fields=["title^3", "description^2", "content", "text"],
        )

        vector_hits = []
        if semantic_enabled:
            query_vec = self._encode_query(expanded_query)
            vector_hits = self.kb.search_vector(
                query_vector=query_vec,
                k=search_pool_k,
            )
        if debug:
            logger.info(
                f"[fusion_inference] retrieve_hits | bm25={len(bm25_hits)} | vector={len(vector_hits)}"
            )

        if not bm25_hits and not vector_hits:
            return []

        hit_by_id = {}
        for hit in bm25_hits:
            hit_by_id[hit.id] = hit
        for hit in vector_hits:
            if hit.id not in hit_by_id:
                hit_by_id[hit.id] = hit

        bm25_ranks = {hit.id: rank for rank, hit in enumerate(bm25_hits)}
        vector_ranks = {hit.id: rank for rank, hit in enumerate(vector_hits)}
        missing_rank = max(search_pool_k, len(hit_by_id))

        # Stage 2: Reciprocal Rank Fusion (same formula as training retriever).
        rrf_scores = {}
        bm25_weight = 1.35 if verbatim_query else 1.0
        vector_weight = 1.0
        for doc_id in hit_by_id:
            bm25_rank = bm25_ranks.get(doc_id, missing_rank)
            vector_rank = vector_ranks.get(doc_id, missing_rank)
            rrf_scores[doc_id] = (bm25_weight / (self.rrf_k + bm25_rank)) + (
                vector_weight / (self.rrf_k + vector_rank)
            )

        rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        rrf_candidates = rrf_ranked[:effective_rrf_top_k]
        if not rrf_candidates:
            return []

        max_rrf = max(score for _, score in rrf_candidates)
        max_rrf = max(max_rrf, 1e-8)
        rrf_scores_norm = {doc_id: score / max_rrf for doc_id, score in rrf_candidates}

        # Build group histories for temporal scoring.
        docs_by_group: Dict[str, List[datetime]] = {}
        for hit in hit_by_id.values():
            source = hit.source or {}
            group = self._doc_group_key(source)
            docs_by_group.setdefault(group, []).append(self._doc_timestamp(source))

        final_items = []
        for doc_id, _ in rrf_candidates:
            hit = hit_by_id[doc_id]
            source = hit.source or {}
            timestamp = self._doc_timestamp(source)
            group = self._doc_group_key(source)

            if use_temporal:
                temporal, recency, cyclicity = (
                    self.temporal_scorer.calculate_temporal_score(
                        timestamp,
                        docs_by_group.get(group, []),
                        use_adaptive_lambda=True,
                    )
                )
                final_score = (
                    self.alpha * rrf_scores_norm[doc_id] + (1.0 - self.alpha) * temporal
                )
            else:
                recency, cyclicity = 0.5, 0.5
                final_score = rrf_scores_norm[doc_id]

            final_items.append(
                RetrievalResult(
                    document_id=doc_id,
                    text=_select_doc_text(source),
                    score=float(final_score),
                    rrf_score=float(rrf_scores_norm[doc_id]),
                    recency_score=float(recency),
                    cyclicity_score=float(cyclicity),
                    timestamp=timestamp,
                    metadata=source,
                )
            )

        if verbatim_query:
            query_lower = normalized_query.lower()
            for item in final_items:
                doc_text = _normalize_query_text(item.text)
                doc_lower = doc_text.lower()
                if query_lower and len(query_lower) >= 40 and query_lower in doc_lower:
                    item.score += 0.35
                else:
                    item.score += 0.15 * _token_overlap_ratio(query_lower, doc_lower)

        final_items.sort(key=lambda x: x.score, reverse=True)
        return final_items[:top_k]


@dataclass
class ClaimPrediction:
    claim: str
    verdict: str  # "Đúng" | "Sai"
    label: str  # Model label token, e.g. "Đúng" | "Sai"
    label_id: int
    confidence: float
    evidence: List[str]


class FusionClaimVerifier:
    """
    Single-claim inference helper:
      claim -> retrieved evidence -> LLM logits -> fusion -> Đúng/Sai verdict.
    """

    def __init__(
        self,
        fusion_model_path: str,
        opensearch_index: Optional[str] = None,
        llm_model_path: Optional[str] = None,
        retriever_model_path: Optional[str] = None,
        device: Optional[str] = None,
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        rrf_k: int = 60,
        llm_evidence_top_k: Optional[int] = None,
        debug: Optional[bool] = None,
        log_evidence_chars: int = 240,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for fusion inference.")
        if ConfidenceAwareFusion is None or RetrievalFeatureEncoder is None:
            raise ImportError(
                "Fusion PyTorch modules are unavailable. Ensure torch is installed."
            )

        if device and str(device).lower() != "cpu":
            logger.warning(
                f"fusion_inference is running in CPU-only mode; got device='{device}', forcing 'cpu'."
            )
        self.device = "cpu"
        self.checkpoint = torch.load(
            fusion_model_path, map_location=torch.device("cpu")
        )
        self.saved_config = self.checkpoint.get("config", {})

        self.top_k = int(self.saved_config.get("top_k", 10))
        if llm_evidence_top_k is None:
            llm_evidence_top_k = int(os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "3"))
        self.llm_evidence_top_k = max(1, int(llm_evidence_top_k))
        self.label_list = list(self.saved_config.get("label_list", LABEL_LIST))
        log_all = _env_flag("FUSION_INFERENCE_LOG_ALL", default=False)
        self.debug = (
            _env_flag("FUSION_INFERENCE_DEBUG", default=False)
            if debug is None
            else bool(debug)
        )
        if log_all:
            self.debug = True
        self.log_evidence_chars = int(
            os.getenv("FUSION_INFERENCE_LOG_EVIDENCE_CHARS", str(log_evidence_chars))
        )
        self.log_full_evidence = _env_flag(
            "FUSION_INFERENCE_LOG_FULL_EVIDENCE", default=False
        )
        if log_all:
            self.log_full_evidence = True
            # 0 means "no truncate" for _truncate()
            self.log_evidence_chars = 0
        if not self.debug:
            logger.debug(
                "Fusion inference debug is OFF. Enable with env FUSION_INFERENCE_DEBUG=1 "
                "or pass debug=True to verify_claim_true_false()/FusionClaimVerifier."
            )

        self.retrieval_encoder = RetrievalFeatureEncoder(
            num_retrieved=self.top_k,
            score_features=4,
            hidden_dim=64,
            output_dim=64,
        ).to(self.device)

        self.fusion = ConfidenceAwareFusion(
            retrieval_input_dim=64,
            hidden_dim=128,
            num_classes=int(self.saved_config.get("num_classes", len(self.label_list))),
            initial_beta=float(self.saved_config.get("initial_beta", 0.8)),
            lambda_reg=float(self.saved_config.get("lambda_reg", 0.01)),
        ).to(self.device)

        self.retrieval_encoder.load_state_dict(self.checkpoint["retrieval_encoder"])
        self.fusion.load_state_dict(self.checkpoint["fusion"])
        self.retrieval_encoder.eval()
        self.fusion.eval()

        model_name = llm_model_path or self.saved_config.get("model_name")
        if not model_name:
            raise ValueError(
                "Missing LLM path. Provide llm_model_path or save model_name in fusion checkpoint."
            )

        from .llm_scorer import LLMScorer

        self.llm = LLMScorer(
            model_name=model_name,
            device=self.device,
            max_length=2048,
            labels=self.label_list,
            prompt_template=PROMPT_TEMPLATE,
        )

        retriever_model = retriever_model_path or self.saved_config.get(
            "retriever_model", "bge-vi-base"
        )

        index_name = (
            opensearch_index
            or os.getenv("OPENSEARCH_INDEX_NAME")
            or os.getenv("OP_KB_NAME")
        )
        if not index_name:
            raise ValueError(
                "Missing OpenSearch index name. Set OPENSEARCH_INDEX_NAME/OP_KB_NAME or pass opensearch_index."
            )

        kb = OpenSearchKB(index_name=index_name, embedding_dim=768)
        self.retriever = OpenSearchHybridRetriever(
            kb=kb,
            embedding_model=retriever_model,
            alpha=alpha,
            lambda_decay=lambda_decay,
            gamma=gamma,
            use_query_expansion=True,
            rrf_k=rrf_k,
        )

    def predict(self, claim: str) -> ClaimPrediction:
        t0 = perf_counter()
        text = str(claim).strip()
        if not text:
            raise ValueError("Claim is empty.")

        now_utc = datetime.now(timezone.utc)
        if self.debug:
            logger.info(
                f"[fusion_inference] start predict | now_utc={now_utc.isoformat()} | top_k={self.top_k} | llm_evidence_top_k={self.llm_evidence_top_k} | labels={self.label_list}"
            )
            logger.info(f"[fusion_inference] claim_input={text!r}")

        t_retrieval0 = perf_counter()
        retrieval_features_np, retrieved_evidence, retrieval_results = (
            _build_retrieval_features_train_compatible(self.retriever, text, self.top_k)
        )
        t_retrieval1 = perf_counter()

        llm_evidence = retrieved_evidence[: self.llm_evidence_top_k]

        if self.debug:
            logger.info(
                f"[fusion_inference] retrieval_done | n_results={len(retrieval_results)} | elapsed_ms={1000.0 * (t_retrieval1 - t_retrieval0):.2f}"
            )
            if retrieval_results:
                for idx, r in enumerate(retrieval_results, start=1):
                    ts = (
                        r.timestamp.astimezone(timezone.utc)
                        if isinstance(r.timestamp, datetime)
                        else _parse_timestamp(r.timestamp)
                    )
                    age_s = (now_utc - ts).total_seconds()
                    meta = r.metadata or {}
                    title = _truncate(str(meta.get("title") or ""), 120)
                    url = _truncate(
                        str(
                            meta.get("url")
                            or meta.get("link")
                            or meta.get("source_url")
                            or ""
                        ),
                        200,
                    )
                    source_name = str(meta.get("source") or meta.get("type") or "")
                    logger.info(
                        "[fusion_inference] retrieved"
                        f" | rank={idx}"
                        f" | doc_id={r.document_id}"
                        f" | ts_utc={ts.isoformat()}"
                        f" | age_hours={age_s / 3600.0:.2f}"
                        f" | score={r.score:.6f}"
                        f" | rrf={r.rrf_score:.6f}"
                        f" | recency={r.recency_score:.6f}"
                        f" | cyclicity={r.cyclicity_score:.6f}"
                        + (f" | source={source_name!r}" if source_name else "")
                        + (f" | title={title!r}" if title else "")
                        + (f" | url={url!r}" if url else "")
                    )

            logger.info(
                "[fusion_inference] retrieval_features_np shape="
                f"{retrieval_features_np.shape} | rows=[score, rrf, recency, cyclicity]"
            )
            logger.info(
                "[fusion_inference] retrieval_features_np="
                + np.array2string(
                    retrieval_features_np,
                    precision=6,
                    suppress_small=False,
                    separator=", ",
                )
            )

            if llm_evidence:
                logger.info(
                    f"[fusion_inference] llm_evidence_input | n_items={len(llm_evidence)} | selected_from={len(retrieved_evidence)} | log_full={self.log_full_evidence} | max_chars={self.log_evidence_chars}"
                )
                for idx, ev in enumerate(llm_evidence, start=1):
                    ev_text = str(ev)
                    if not self.log_full_evidence:
                        ev_text = _truncate(ev_text, self.log_evidence_chars)
                    logger.info(f"[fusion_inference] evidence[{idx}]={ev_text!r}")

        retrieval_features = torch.tensor(
            retrieval_features_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.inference_mode():
            t_llm0 = perf_counter()
            llm_logits = self.llm.score_logits([text], [llm_evidence]).to(self.device)
            t_llm1 = perf_counter()

            if self.debug:
                llm_probs = torch.softmax(llm_logits, dim=-1)
                logger.info(
                    f"[fusion_inference] llm_done | logits_shape={tuple(llm_logits.shape)} | elapsed_ms={1000.0 * (t_llm1 - t_llm0):.2f}"
                )
                logger.info(
                    "[fusion_inference] llm_logits="
                    + np.array2string(
                        llm_logits.detach().cpu().numpy(),
                        precision=6,
                        suppress_small=False,
                        separator=", ",
                    )
                )
                logger.info(
                    "[fusion_inference] llm_probs="
                    + np.array2string(
                        llm_probs.detach().cpu().numpy(),
                        precision=6,
                        suppress_small=False,
                        separator=", ",
                    )
                )

            retrieval_encoded = self.retrieval_encoder(retrieval_features)
            if self.debug:
                enc = retrieval_encoded.detach().cpu()
                logger.info(
                    f"[fusion_inference] retrieval_encoder_out | shape={tuple(enc.shape)} | mean={enc.mean().item():.6f} | std={enc.std(unbiased=False).item():.6f}"
                )

            fusion_output = self.fusion(llm_logits, retrieval_encoded)
            probs = fusion_output.final_probs[0]
            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())

            if self.debug:
                logger.info(
                    f"[fusion_inference] fusion_done | lm_weight={fusion_output.lm_weight:.6f} | retrieval_weight={fusion_output.retrieval_weight:.6f}"
                )
                logger.info(
                    "[fusion_inference] fused_logits="
                    + np.array2string(
                        fusion_output.fused_logits.detach().cpu().numpy(),
                        precision=6,
                        suppress_small=False,
                        separator=", ",
                    )
                )
                logger.info(
                    "[fusion_inference] final_probs="
                    + np.array2string(
                        fusion_output.final_probs.detach().cpu().numpy(),
                        precision=6,
                        suppress_small=False,
                        separator=", ",
                    )
                )

        pred_label = self.label_list[pred_id]
        # Keep verdict tied to stable label ID convention: 0=Đúng, 1=Sai.
        verdict = "Đúng" if pred_id == 0 else "Sai"

        if self.debug:
            logger.info(
                f"[fusion_inference] done predict | verdict={verdict!r} | label={pred_label!r} | confidence={confidence:.6f} | elapsed_ms={1000.0 * (perf_counter() - t0):.2f}"
            )

        return ClaimPrediction(
            claim=text,
            verdict=verdict,
            label=pred_label,
            label_id=pred_id,
            confidence=confidence,
            evidence=llm_evidence,
        )


_VERIFIER_CACHE: Dict[str, FusionClaimVerifier] = {}


def verify_claim_true_false(
    claim: str,
    fusion_model_path: str = "artifacts/fusion_model.pt",
    opensearch_index: Optional[str] = None,
    llm_model_path: Optional[str] = None,
    retriever_model_path: Optional[str] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    llm_evidence_top_k: Optional[int] = None,
    debug: Optional[bool] = None,
) -> str:
    """
    Convenience function requested:
      input: claim text
      output: "Đúng" hoặc "Sai"
    """
    effective_debug = (
        _env_flag("FUSION_INFERENCE_DEBUG", default=False)
        if debug is None
        else bool(debug)
    )
    cache_key = "|".join(
        [
            fusion_model_path,
            opensearch_index or "",
            llm_model_path or "",
            retriever_model_path or "",
            device or "",
            str(llm_evidence_top_k or ""),
            f"debug={int(effective_debug)}",
        ]
    )

    verifier = None
    if use_cache:
        verifier = _VERIFIER_CACHE.get(cache_key)

    if verifier is None:
        verifier = FusionClaimVerifier(
            fusion_model_path=fusion_model_path,
            opensearch_index=opensearch_index,
            llm_model_path=llm_model_path,
            retriever_model_path=retriever_model_path,
            device=device,
            llm_evidence_top_k=llm_evidence_top_k,
            debug=effective_debug,
        )
        if use_cache:
            _VERIFIER_CACHE[cache_key] = verifier

    prediction = verifier.predict(claim)
    return prediction.verdict
