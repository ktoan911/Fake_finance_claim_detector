import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from loguru import logger

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using numpy-based embeddings only.")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class TripletSample:
    """Container for triplet training sample"""

    anchor: str  # Query
    positive: str  # Matching scam pattern
    negatives: List[str]  # List of non-matching samples (hard + random)


class SimulatedEmbeddingModel:
    """
    Simulated embedding model for testing without heavy dependencies.
    Uses TF-IDF with dimension reduction.

    This is the main embedding model used when PyTorch/SentenceTransformers
    are not available.
    """

    def __init__(self, embedding_dim: int = 128, seed: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), stop_words="english"
        )
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=seed)
        self.is_fitted = False

    def fit(self, texts: List[str]) -> None:
        """Fit the embedding model"""
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self.is_fitted = True
        logger.info(f"SimulatedEmbeddingModel fitted on {len(texts)} texts")

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # TF-IDF + SVD
        tfidf = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embeddings = embeddings / norms

        return embeddings

    def similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """Compute cosine similarity between query and documents"""
        query_emb = self.encode([query])
        doc_embs = self.encode(documents)

        similarities = np.dot(doc_embs, query_emb.T).flatten()
        return similarities


# PyTorch-based classes (only available if torch is installed)
if TORCH_AVAILABLE:

    class CryptoEmbeddingDataset(TorchDataset):
        """
        Dataset for contrastive learning on crypto scam patterns.
        Creates triplets (q, d+, {d-}) for training.
        Supports hard negative mining and multiple negatives.
        """

        def __init__(
            self,
            scam_samples: List[Dict],
            legitimate_samples: List[Dict],
            num_triplets: int = 10000,
            num_negatives: int = 3,  # Number of negatives per anchor
            hard_negative_ratio: float = 0.5,
            embedding_model=None,  # Optional model for hard negative mining
        ):
            # --- Core state ---
            self.scam_samples = scam_samples
            self.num_negatives = num_negatives
            self.hard_negative_ratio = hard_negative_ratio
            self.embedding_model = embedding_model

            # Group scams by type for triplet construction
            self.scams_by_type: Dict[str, List[Dict]] = {}
            for sample in scam_samples:
                t = sample.get("scam_type", "unknown")
                self.scams_by_type.setdefault(t, []).append(sample)

            # Store only text (not full dicts) to minimize RAM for legitimate samples
            self.legit_texts: List[str] = [s["text"] for s in legitimate_samples]

            # Pre-compute embeddings for hard negative mining if model provided
            self.legit_embeddings = None
            if self.embedding_model and hasattr(self.embedding_model, "encode"):
                logger.info(
                    "Pre-computing legitimate embeddings for hard negative mining..."
                )
                L = len(self.legit_texts)

                # 1) Probe one mini-batch to determine D without storing outs list
                with torch.no_grad():
                    probe = self.embedding_model.encode(
                        self.legit_texts[: min(8, L)], normalize=True
                    ).cpu()
                D = int(probe.shape[1])
                del probe

                # 2) Pre-allocate final tensor once → no double-copy from torch.cat
                self.legit_embeddings = torch.empty(
                    (L, D), dtype=torch.float16, device="cpu"
                )

                bs = 256
                write_ptr = 0
                with torch.no_grad():
                    for i in range(0, L, bs):
                        emb = self.embedding_model.encode(
                            self.legit_texts[i : i + bs], normalize=True
                        ).cpu()  # float32 (from SentenceTransformer)
                        emb = emb.float()
                        emb = F.normalize(emb, p=2, dim=1)  # guarantee L2 norm
                        emb = emb.to(torch.float16)
                        bsz = emb.size(0)
                        self.legit_embeddings[write_ptr : write_ptr + bsz].copy_(emb)
                        write_ptr += bsz
                        del emb  # release immediately to avoid fragmentation

                self.legit_embeddings = self.legit_embeddings.contiguous()

                ram_mb = (
                    self.legit_embeddings.nelement()
                    * self.legit_embeddings.element_size()
                    / 1e6
                )
                logger.info(
                    f"legit_embeddings: shape={tuple(self.legit_embeddings.shape)}, "
                    f"dtype={self.legit_embeddings.dtype}, RAM≈{ram_mb:.1f} MB"
                )

            # LRU cache: hash64 key, float16 value, capped at 5k
            self._anchor_cache: OrderedDict = OrderedDict()
            self._anchor_cache_max = 5000

            # Generate triplets on the fly instead of storing strings in RAM
            self.num_triplets = num_triplets
            logger.info(
                f"Dataset ready to lazily generate {self.num_triplets} triplets via __getitem__"
            )

        def _topk_legit_for_scam(
            self, scam_emb: torch.Tensor, k: int = 20, block: int = 2048
        ):
            """
            Optimized chunked hard negative mining to prevent OOM
            scam_emb: [1, D] or [D] on CPU, normalized
            self.legit_embeddings: [L, D] on CPU, normalized
            Return: indices [k]
            """
            best_scores = torch.full((k,), -1e9, device="cpu")
            best_idx = torch.full((k,), -1, dtype=torch.long, device="cpu")

            L = self.legit_embeddings.size(0)
            # Cast to match float16 embeddings for dot product
            scam_emb_flat = scam_emb.view(-1).to(torch.float16)

            for start in range(0, L, block):
                end = min(start + block, L)
                chunk = self.legit_embeddings[start:end]  # [b, D]
                scores = torch.mv(chunk, scam_emb_flat)  # cosine = dot (đã normalize)

                # lấy topk trong block
                tk = min(k, scores.numel())
                s, idx = torch.topk(scores, k=tk)
                idx = idx + start

                # merge vào best hiện tại
                all_scores = torch.cat([best_scores, s])
                all_idx = torch.cat([best_idx, idx])
                best_scores, pos = torch.topk(all_scores, k=k)
                best_idx = all_idx[pos]

            return best_idx

        @staticmethod
        def _hash64(s: str) -> int:
            """Stable 64-bit hash for dict key to avoid storing full strings."""
            return int.from_bytes(
                hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little"
            )

        def _get_anchor_emb(self, anchor_text: str) -> torch.Tensor:
            """Encode anchor text with LRU caching (hash64 key, float16 value, max 5k entries)."""
            key = self._hash64(anchor_text)
            v = self._anchor_cache.get(key)
            if v is not None:
                self._anchor_cache.move_to_end(key)  # LRU update
                return v

            with torch.no_grad():
                emb = self.embedding_model.encode([anchor_text], normalize=True).cpu()[
                    0
                ]
            emb = emb.float()
            emb = F.normalize(
                emb, p=2, dim=0
            )  # guarantee L2 norm for dot-product cosine
            emb = emb.to(torch.float16)  # float16 to halve RAM

            self._anchor_cache[key] = emb
            if len(self._anchor_cache) > self._anchor_cache_max:
                self._anchor_cache.popitem(last=False)  # evict oldest
            return emb

        def __len__(self) -> int:
            return self.num_triplets

        def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str]]]:
            scam_types = list(self.scams_by_type.keys())

            # Bounded retry loop - prevent infinite loops if imbalanced dataset
            fallback = None
            for _ in range(50):
                anchor_type = np.random.choice(scam_types)
                anchor_samples = self.scams_by_type[anchor_type]

                if not anchor_samples:
                    continue

                # Pick anchor
                anchor_idx = np.random.choice(len(anchor_samples))
                anchor_sample = anchor_samples[anchor_idx]
                anchor_text = anchor_sample["text"]

                # Determine Positive
                # Priority: Use "evidence" field (Paper alignment: Query -> Evidence)
                if "evidence" in anchor_sample and anchor_sample["evidence"]:
                    positive_text = anchor_sample["evidence"]
                    using_evidence = True
                elif len(anchor_samples) >= 2:
                    # Fallback: Metric Learning (Cluster same scam types)
                    pos_idx = np.random.choice(
                        [i for i in range(len(anchor_samples)) if i != anchor_idx]
                    )
                    positive_text = anchor_samples[pos_idx]["text"]
                    using_evidence = False
                else:
                    continue

                negatives = []

                # Try to mine hard negatives on-the-fly
                if self.legit_embeddings is not None:
                    anchor_emb = self._get_anchor_emb(anchor_text)

                    # Call optimized chunked search instead of full matrix
                    top_k_indices = self._topk_legit_for_scam(
                        anchor_emb, k=min(20, len(self.legit_texts))
                    )

                    # Add hard negatives
                    num_hard = int(self.num_negatives * self.hard_negative_ratio)
                    for _ in range(num_hard):
                        neg_idx = top_k_indices[
                            np.random.randint(0, len(top_k_indices))
                        ].item()
                        negatives.append(self.legit_texts[neg_idx])

                # Fill remaining with random negatives
                while len(negatives) < self.num_negatives:
                    if np.random.random() < 0.5:
                        # Random negative from other scam types
                        other_types = [t for t in scam_types if t != anchor_type]
                        if other_types:
                            neg_type = np.random.choice(other_types)
                            neg_sample = np.random.choice(self.scams_by_type[neg_type])

                            if (
                                using_evidence
                                and "evidence" in neg_sample
                                and neg_sample["evidence"]
                            ):
                                negatives.append(neg_sample["evidence"])
                            else:
                                negatives.append(neg_sample["text"])
                        else:
                            negatives.append(np.random.choice(self.legit_texts))
                    else:
                        # Random legitimate sample
                        negatives.append(np.random.choice(self.legit_texts))

                # If we successfully made it here, cache as fallback too
                triplet = {
                    "anchor": anchor_text,
                    "positive": positive_text,
                    "negatives": negatives,
                }
                fallback = triplet
                return triplet

            # If 50 attempts all failed, return last known good or a random fallback
            if fallback is not None:
                return fallback

            # Final emergency fallback with random data to prevent DataLoader crash
            anchor_sample = np.random.choice(self.scam_samples)
            legit_text = np.random.choice(self.legit_texts)
            return {
                "anchor": anchor_sample["text"],
                "positive": anchor_sample.get("evidence", anchor_sample["text"]),
                "negatives": [legit_text] * self.num_negatives,
            }

    class ContrastiveEmbeddingModel(nn.Module):
        """
        Embedding model with contrastive fine-tuning capability.
        Implements Equation (7):
        L = -log(e^s+ / (e^s+ + Σe^s-)) + λ||θ||²
        """

        def __init__(
            self,
            base_model_name: str = "BAAI/bge-small-en-v1.5",
            embedding_dim: int = 384,
            lambda_reg: float = 0.001,
            freeze_base: bool = True,
            max_length: int = 256,
            encoder_device: str = "cpu",
            encode_batch_size: int = 16,
        ):
            super().__init__()

            self.lambda_reg = lambda_reg
            self.embedding_dim = embedding_dim
            # Strict memory cap on attention matrices
            self.max_length = max_length
            self.encoder_device = encoder_device
            self.encode_batch_size = encode_batch_size

            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.encoder = SentenceTransformer(
                    base_model_name, device=encoder_device
                )

                if hasattr(self.encoder, "max_seq_length"):
                    self.encoder.max_seq_length = max_length

                # CRITICAL: auto-cap max_length at the model's hard limit.
                # BERT-based models (bge-small etc.) have exactly 512 position
                # embeddings — passing anything larger causes a guaranteed crash.
                try:
                    model_max_pos = self.encoder[
                        0
                    ].auto_model.config.max_position_embeddings
                    if max_length > model_max_pos:
                        logger.warning(
                            f"--max_length {max_length} exceeds model's "
                            f"max_position_embeddings ({model_max_pos}). "
                            f"Auto-capping to {model_max_pos}."
                        )
                        max_length = model_max_pos
                        self.max_length = max_length
                except Exception:
                    pass  # Best-effort; proceed with user-supplied value

                # Propagate capped value to all tokenizer layers
                try:
                    if hasattr(self.encoder[0], "max_seq_length"):
                        self.encoder[0].max_seq_length = max_length
                    if hasattr(self.encoder[0], "tokenizer"):
                        self.encoder[0].tokenizer.model_max_length = max_length
                except Exception:
                    pass

                # Explicitly enforce device placement before anything else
                self.encoder.to(encoder_device)

                # If doing long sequences, enable gradient checkpointing to save VRAM
                if max_length > 512 and not freeze_base:
                    if hasattr(
                        self.encoder[0].auto_model, "gradient_checkpointing_enable"
                    ):
                        logger.warning(
                            f"Max length {max_length} is large! Enabling Gradient Checkpointing to save VRAM."
                        )
                        self.encoder[0].auto_model.gradient_checkpointing_enable()

                if freeze_base:
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
            else:
                self.encoder = None

            self.projection = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )

            logger.info(
                f"ContrastiveEmbeddingModel initialized: dim={self.embedding_dim}, λ={lambda_reg}"
            )

        def to(self, *args, **kwargs):
            """Override to() to prevent encoder from moving to GPU when freeze_base=True"""
            # First, move everything to the requested device normally
            super().to(*args, **kwargs)

            # Then gracefully pin the encoder back to the chosen encoder_device
            # if we are doing cpu-microbatching.
            if getattr(self, "encoder", None) is not None:
                self.encoder.to(self.encoder_device)

            return self

        def _encode_base_microbatch(self, texts: List[str]) -> torch.Tensor:
            """
            Encode base embeddings in micro-batches to control peak memory.
            Runs on self.encoder_device (CPU recommended).
            """
            if self.encoder is None:
                return torch.randn(len(texts), self.embedding_dim)

            # If we are training and base is not frozen, we need gradients
            if self.training and any(
                p.requires_grad for p in self.encoder.parameters()
            ):
                device = next(self.projection.parameters()).device
                # Tokenize with strict mapping to prevent OOM
                features = self.encoder.tokenize(texts)

                if hasattr(self.encoder.tokenizer, "pad_token_id"):
                    features = self.encoder.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )

                features = {k: v.to(device) for k, v in features.items()}
                # Forward pass through base model to get gradients
                out_features = self.encoder(features)
                return out_features["sentence_embedding"]

            # Ordinary frozen execution (micro-batched on CPU for safety)
            outs = []
            # Get the actual HuggingFace tokenizer for guaranteed truncation.
            # ST's encoder.tokenize() sometimes ignores max_seq_length, so we
            # call the underlying tokenizer directly with explicit truncation args.
            try:
                hf_tokenizer = self.encoder[0].tokenizer
            except (IndexError, AttributeError):
                hf_tokenizer = getattr(self.encoder, "tokenizer", None)

            with torch.no_grad():
                for i in range(0, len(texts), self.encode_batch_size):
                    chunk = texts[i : i + self.encode_batch_size]

                    if hf_tokenizer is not None:
                        # Use HF tokenizer directly with strict truncation — guaranteed safe
                        features = hf_tokenizer(
                            chunk,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                    else:
                        # Last resort: rely on ST tokenize() with max_seq_length set above
                        features = self.encoder.tokenize(chunk)

                    features = {
                        k: v.to(self.encoder_device) for k, v in features.items()
                    }
                    out_features = self.encoder.forward(features)

                    if "sentence_embedding" in out_features:
                        emb = out_features["sentence_embedding"]
                    else:
                        # Mean pooling fallback for models without pooling layer
                        token_embs = out_features["token_embeddings"]
                        mask = out_features["attention_mask"].unsqueeze(-1).float()
                        emb = (token_embs * mask).sum(1) / torch.clamp(
                            mask.sum(1), min=1e-9
                        )

                    outs.append(emb)
            return torch.cat(outs, dim=0)

        def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
            """Encode texts to embeddings."""
            proj_device = next(self.projection.parameters()).device

            base_embeddings = self._encode_base_microbatch(texts)
            base_embeddings = base_embeddings.to(proj_device, non_blocking=True)

            projected = self.projection(base_embeddings)

            if normalize:
                projected = F.normalize(projected, p=2, dim=-1)

            return projected

        def compute_similarity(
            self, anchor_emb: torch.Tensor, target_emb: torch.Tensor
        ) -> torch.Tensor:
            """Compute cosine similarity between embeddings"""
            return F.cosine_similarity(anchor_emb, target_emb)

        def compute_contrastive_loss(
            self,
            anchor_emb: torch.Tensor,
            positive_emb: torch.Tensor,
            negative_embs: torch.Tensor,  # Can be [B, N_neg, Dim] or [B, Dim]
            temperature: float = 0.07,
        ) -> torch.Tensor:
            """
            Compute contrastive loss (Equation 7) with multiple negatives support:
            L = -log(e^s+ / (e^s+ + Σe^s-)) + λ||θ||²
            """
            # Positive similarity: [B]
            pos_sim = self.compute_similarity(anchor_emb, positive_emb) / temperature

            # Negative similarity
            # If negative_embs is [B, Dim], treat as 1 negative per sample
            if negative_embs.dim() == 2:
                neg_sim = (
                    self.compute_similarity(anchor_emb, negative_embs) / temperature
                )
                neg_sim = neg_sim.unsqueeze(1)  # [B, 1]
            # If negative_embs is [B, N_neg, Dim], compute sim for each
            elif negative_embs.dim() == 3:
                # anchor: [B, 1, Dim]
                anchor_expanded = anchor_emb.unsqueeze(1)
                # neg_sim: [B, N_neg]
                neg_sim = (
                    F.cosine_similarity(anchor_expanded, negative_embs, dim=2)
                    / temperature
                )

            # InfoNCE Loss
            # numerator: e^pos
            numerator = torch.exp(pos_sim)

            # denominator: e^pos + Σe^neg
            denominator = numerator + torch.sum(torch.exp(neg_sim), dim=-1)

            contrastive_loss = -torch.log(numerator / (denominator + 1e-8))
            contrastive_loss = contrastive_loss.mean()

            # Regularization (Squared L2 norm)
            reg_loss = 0.0
            for param in self.projection.parameters():
                reg_loss += torch.sum(param**2)
            reg_loss *= self.lambda_reg

            return contrastive_loss + reg_loss

    class RetrievalDataset(TorchDataset):
        """
        Dataset for training a retrieval model on claim-evidence pairs.

        Format of ``records``:
            [
              {"claim": "...", "evidences": ["sentence 1", "sentence 2", ...]},
              ...
            ]

        Each triplet:
          anchor   = claim text
          positive = one randomly sampled evidence sentence from that claim
          negatives = evidence sentences sampled from *other* claims

        This is a drop-in replacement for CryptoEmbeddingDataset when the
        training data comes from finfact_raw_truefalse.csv (claim / evidence
        columns) rather than the scam/legit split used by CryptoEmbeddingDataset.
        """

        def __init__(
            self,
            records: List[Dict],
            num_triplets: int = 8000,
            num_negatives: int = 3,
        ):
            """
            Args:
                records: list of dicts with keys "claim" and "evidences"
                num_triplets: total samples __len__ reports (lazy generation)
                num_negatives: how many negative evidence sentences per anchor
            """
            self.records = records
            self.num_triplets = num_triplets
            self.num_negatives = num_negatives

            # Pre-build a flat list of (record_idx, sentence) for fast negative sampling
            self._all_sentences: List[tuple] = []  # (record_idx, sentence)
            for i, rec in enumerate(records):
                for ev in rec["evidences"]:
                    self._all_sentences.append((i, ev))

            logger.info(
                f"RetrievalDataset | records={len(records)} | "
                f"total_sentences={len(self._all_sentences)} | "
                f"num_triplets={num_triplets}"
            )

        def __len__(self) -> int:
            return self.num_triplets

        def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str]]]:
            # Pick a random record as anchor
            anchor_idx = np.random.randint(0, len(self.records))
            anchor_rec = self.records[anchor_idx]

            # Anchor = claim text
            anchor_text = anchor_rec["claim"]

            # Positive = one of its evidence sentences
            positive_text = anchor_rec["evidences"][
                np.random.randint(0, len(anchor_rec["evidences"]))
            ]

            # Negatives = evidence sentences from *other* records
            negatives: List[str] = []
            attempts = 0
            while (
                len(negatives) < self.num_negatives
                and attempts < self.num_negatives * 20
            ):
                attempts += 1
                neg_global_idx = np.random.randint(0, len(self._all_sentences))
                neg_rec_idx, neg_sentence = self._all_sentences[neg_global_idx]
                if neg_rec_idx != anchor_idx:
                    negatives.append(neg_sentence)

            # Safety: if we couldn't find enough, reuse what we have
            while len(negatives) < self.num_negatives:
                negatives.append(negatives[0] if negatives else positive_text)

            return {
                "anchor": anchor_text,
                "positive": positive_text,
                "negatives": negatives,
            }

    class CryptoEmbeddingTrainer:
        """Trainer for contrastive embedding fine-tuning."""

        def __init__(
            self,
            model: ContrastiveEmbeddingModel,
            learning_rate: float = 1e-4,
            batch_size: int = 32,
            device: str = "cpu",
        ):
            self.model = model.to(device)
            self.device = device
            self.batch_size = batch_size

            self.optimizer = torch.optim.Adam(
                model.projection.parameters(), lr=learning_rate
            )

            logger.info(
                f"Trainer initialized: lr={learning_rate}, batch_size={batch_size}"
            )

        def train_epoch(self, dataloader: DataLoader) -> float:
            """Train for one epoch"""
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            # Gradient Accumulation Steps
            accum_steps = 4

            # Automatic Mixed Precision to save GPU memory
            scaler = (
                torch.amp.GradScaler(device=self.device)
                if self.device == "cuda"
                else None
            )

            for step, batch in enumerate(dataloader):
                # 1. Prepare flat text list to encode everything in ONE pass
                anchors: List[str] = batch["anchor"]
                positives: List[str] = batch["positive"]
                negatives: List[List[str]] = batch["negatives"]  # [B][N_neg]

                B = len(anchors)
                N_neg = len(negatives[0]) if B > 0 else 0

                # Flatten negatives: [B * N_neg]
                neg_flat = [negatives[i][j] for i in range(B) for j in range(N_neg)]

                # Combine all texts
                all_texts = anchors + positives + neg_flat

                with (
                    torch.amp.autocast(device_type=self.device)
                    if scaler
                    else torch.autocast(device_type=self.device, enabled=False)
                ):
                    # One big encode: [2B + B*N_neg, Dim]
                    all_embs = self.model.encode(all_texts)

                    # Extract slices
                    anchor_emb = all_embs[0:B]
                    positive_emb = all_embs[B : 2 * B]
                    neg_emb_flat = all_embs[2 * B :]
                    negative_embs = neg_emb_flat.view(B, N_neg, -1)  # [B, N_neg, Dim]

                    # Compute loss
                    loss = self.model.compute_contrastive_loss(
                        anchor_emb, positive_emb, negative_embs
                    )

                    # Scale down loss for gradient accumulation
                    loss = loss / accum_steps

                if scaler:
                    scaler.scale(loss).backward()

                    if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.item()) * accum_steps
                num_batches += 1

            return total_loss / max(num_batches, 1)

        def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
            """Evaluate embedding quality"""
            self.model.eval()

            pos_similarities = []
            neg_similarities = []

            with torch.no_grad():
                for batch in dataloader:
                    anchor_emb = self.model.encode(batch["anchor"])
                    positive_emb = self.model.encode(batch["positive"])

                    # Handle multiple negatives: Transpose to get [N_neg][B]
                    negatives = batch["negatives"]
                    negatives_by_k = list(zip(*negatives))

                    # Just take the first negative batch for simple evaluation metric
                    first_neg_batch = negatives_by_k[0]
                    negative_emb = self.model.encode(list(first_neg_batch))

                    pos_sim = self.model.compute_similarity(anchor_emb, positive_emb)
                    neg_sim = self.model.compute_similarity(anchor_emb, negative_emb)

                    pos_similarities.extend(pos_sim.cpu().numpy().tolist())
                    neg_similarities.extend(neg_sim.cpu().numpy().tolist())

            pos_similarities = np.array(pos_similarities)
            neg_similarities = np.array(neg_similarities)

            metrics = {
                "mean_pos_similarity": np.mean(pos_similarities),
                "mean_neg_similarity": np.mean(neg_similarities),
                "pos_neg_gap": np.mean(pos_similarities) - np.mean(neg_similarities),
                "separation_rate": np.mean(pos_similarities > neg_similarities),
                "target_pos_distance": 1 - np.mean(pos_similarities),
                "target_neg_distance": 1 - np.mean(neg_similarities),
            }

            return metrics

else:
    # Dummy classes when torch is not available
    class CryptoEmbeddingDataset:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("CryptoEmbeddingDataset requires PyTorch")

    class RetrievalDataset:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("RetrievalDataset requires PyTorch")

    class ContrastiveEmbeddingModel:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("ContrastiveEmbeddingModel requires PyTorch")

    class CryptoEmbeddingTrainer:
        """Placeholder when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            logger.warning("CryptoEmbeddingTrainer requires PyTorch")
