import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using numpy-based similarity")

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available, using TF-IDF")
    CrossEncoder = None

import re

from scipy.fft import fft
from scipy.signal import find_peaks

try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer

    NLTK_AVAILABLE = True
except ImportError as e:
    NLTK_AVAILABLE = False
    logger.warning(f"NLTK not available, advanced preprocessing disabled. Error: {e}")


class TextPreprocessor:
    """
    Handles text preprocessing for retrieval:
    - URL removal
    - Stopword removal
    - Lemmatization
    """

    def __init__(self):
        if not NLTK_AVAILABLE:
            return

        # Download necessary NLTK data
        try:
            nltk.data.find("corpora/stopwords")
            nltk.data.find("corpora/wordnet")
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def _get_wordnet_pos(self, treebank_tag):
        """Map NLTK POS tag to WordNet POS tag"""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text and return list of tokens.
        1. Remove URLs
        2. Lowercase
        3. Tokenize
        4. Remove stopwords & Lemmatize (with POS tags)
        """
        if not NLTK_AVAILABLE:
            # Fallback to simple tokenization
            return re.findall(r"\b\w+\b", text.lower())

        # 1. Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # 2. Lowercase
        text = text.lower()

        # 3. Tokenize
        try:
            tokens = nltk.word_tokenize(text)
        except LookupError:
            # Fallback if punkt fails
            tokens = re.findall(r"\b\w+\b", text)

        # 4. Remove stopwords and Lemmatize
        clean_tokens = []

        # Get POS tags for better lemmatization
        try:
            pos_tags = nltk.pos_tag(tokens)
        except LookupError:
            # Fallback if tagger fails
            pos_tags = [(t, "N") for t in tokens]

        for token, tag in pos_tags:
            # Simple check for alphanumeric to avoid punctuation
            if token.isalnum() and token not in self.stop_words:
                wn_tag = self._get_wordnet_pos(tag)
                clean_tokens.append(self.lemmatizer.lemmatize(token, wn_tag))

        return clean_tokens


@dataclass
class RetrievalResult:
    document_id: str
    text: str
    score: float
    rrf_score: float  # Changed from bm25_score to rrf_score
    recency_score: float
    cyclicity_score: float
    timestamp: datetime
    metadata: Dict


class TemporalScorer:
    """
    Implements temporal scoring from Equation (1):
    Score(q, di) = α · BM25(q, di) + (1 − α) · Recency(di)

    With extensions from Section 3.5:
    - Cycle-Aware Scoring (Eq. 8)
    - Parameter Adaptation (Eq. 9)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        reference_date: datetime = None,  # Will be set dynamically per query if None
    ):
        """
        Initialize temporal scorer with paper parameters.

        Args:
            alpha: Weight for BM25 vs temporal (paper: 0.7)
            lambda_decay: Exponential decay factor (paper: 0.1)
            gamma: Recency vs cyclicity mix (paper: 0.5)
            reference_date: Reference date for recency calculation
        """
        self.alpha = alpha
        self.lambda_decay = lambda_decay
        self.lambda_base = lambda_decay
        self.gamma = gamma
        self._reference_date = reference_date  # Private, can be overridden

        logger.info(
            f"TemporalScorer initialized: α={alpha}, λ={lambda_decay}, γ={gamma}"
        )

    @property
    def reference_date(self) -> datetime:
        """Get reference date, defaulting to current time if not set."""
        if self._reference_date is None:
            return datetime.now(timezone.utc)
        return self._reference_date

    @reference_date.setter
    def reference_date(self, value: datetime):
        """Set reference date for recency calculations."""
        self._reference_date = value

    def calculate_recency(self, timestamp: datetime) -> float:
        """
        Calculate recency score using exponential decay: e^(-λt)

        Args:
            timestamp: Document timestamp

        Returns:
            Recency score between 0 and 1
        """
        if timestamp is None:
            return 0.0

        # Normalize timezone awareness
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ref = self.reference_date
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)

        # Paper Eq.1: e^(-λt) where t is time difference
        # Use total_seconds for precise resolution (not quantized by .days)
        time_diff_seconds = (ref - ts).total_seconds()
        time_diff_days = max(0, time_diff_seconds / 86400.0)  # Convert to days
        return float(np.exp(-self.lambda_decay * time_diff_days))

    def calculate_cyclicity(
        self, timestamps: List[datetime], pattern_type: str = None
    ) -> float:
        """
        Calculate cyclicity score using FFT to detect repeating patterns.
        Implements Section 4.1's cycle-aware scoring (Eq.8).

        Args:
            timestamps: List of document timestamps for pattern analysis
            pattern_type: Optional pattern type hint

        Returns:
            Cyclicity score between 0 and 1
        """
        valid_timestamps = [t for t in timestamps if t is not None]
        if len(valid_timestamps) < 10:
            return 0.5  # Default for insufficient data

        # Convert to daily occurrence counts
        min_date = min(valid_timestamps)
        max_date = max(valid_timestamps)
        date_range = (max_date - min_date).days + 1

        if date_range < 7:
            return 0.5

        # Create time series
        daily_counts = np.zeros(date_range)
        for ts in valid_timestamps:
            day_idx = (ts - min_date).days
            if 0 <= day_idx < date_range:
                daily_counts[day_idx] += 1

        # Apply FFT
        fft_result = fft(daily_counts)
        power_spectrum = np.abs(fft_result[: len(fft_result) // 2]) ** 2

        # Find dominant frequencies
        peaks, properties = find_peaks(power_spectrum, height=np.mean(power_spectrum))

        if len(peaks) > 0:
            # Calculate cyclicity based on peak strength
            max_peak_power = max(properties["peak_heights"])
            total_power = np.sum(power_spectrum)
            cyclicity = min(1.0, max_peak_power / (total_power + 1e-8) * 5)
        else:
            cyclicity = 0.3

        return cyclicity

    def adapt_lambda(self, trend_indicator: float) -> None:
        """
        Dynamically adjust λ based on trend.
        Implements Equation (9): λt = Sigmoid(Trend(di)) · λbase

        Args:
            trend_indicator: Trend value between -1 and 1
        """
        sigmoid = 1 / (1 + np.exp(-trend_indicator))
        self.lambda_decay = sigmoid * self.lambda_base

    def calculate_trend(self, timestamps: List[datetime]) -> float:
        """
        Calculate trend indicator for adaptive λ.
        Simple approach: slope of recent occurrences.
        Returns value suitable for sigmoid (-3 to 3 range).
        """
        valid_timestamps = [t for t in timestamps if t is not None]
        if len(valid_timestamps) < 5:
            return 0.0

        # Sort and get recent window
        sorted_ts = sorted(valid_timestamps, reverse=True)[:30]
        if len(sorted_ts) < 5:
            return 0.0

        # Count occurrences in recent weeks
        now = self.reference_date
        week1 = sum(1 for t in sorted_ts if (now - t).days <= 7)
        week2 = sum(1 for t in sorted_ts if 7 < (now - t).days <= 14)
        week3 = sum(1 for t in sorted_ts if 14 < (now - t).days <= 21)

        # Trend: positive if increasing, negative if decreasing
        if week3 > 0:
            trend = (week1 - week3) / (week3 + 1)  # Normalized change
        else:
            trend = week1 / 3.0

        return np.clip(trend, -3, 3)  # For sigmoid input

    def calculate_temporal_score(
        self,
        timestamp: datetime,
        historical_timestamps: List[datetime] = None,
        use_adaptive_lambda: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Calculate combined temporal score.
        Implements Equation (8): temporal = γ·Recency + (1-γ)·Cyclicity
        with Equation (9): λt = Sigmoid(Trend) · λbase for adaptive decay

        Returns:
            Tuple of (temporal_score, recency_score, cyclicity_score)
        """
        # Apply adaptive lambda if historical data available
        if use_adaptive_lambda and historical_timestamps:
            trend = self.calculate_trend(historical_timestamps)
            self.adapt_lambda(trend)

        recency = self.calculate_recency(timestamp)

        if historical_timestamps:
            cyclicity = self.calculate_cyclicity(historical_timestamps)
        else:
            cyclicity = 0.5

        temporal_score = self.gamma * recency + (1 - self.gamma) * cyclicity

        # Reset lambda to base for next calculation
        if use_adaptive_lambda:
            self.lambda_decay = self.lambda_base

        return temporal_score, recency, cyclicity


class QueryExpander:
    """
    Implements query expansion for fact-checking.
    Uses synonyms for better evidence retrieval.
    """

    def __init__(self, glossary: Dict[str, List[str]] = None):
        """
        Initialize query expander.

        Args:
            glossary: Dictionary mapping terms to synonyms
        """
        self.glossary = glossary or self._default_glossary()
        logger.info(f"QueryExpander initialized with {len(self.glossary)} terms")

    def _default_glossary(self) -> Dict[str, List[str]]:
        """Default fact-checking glossary"""
        return {
            # Claim verification terms
            "claim": ["statement", "assertion", "allegation"],
            "evidence": ["proof", "support", "documentation"],
            "false": ["incorrect", "inaccurate", "misleading", "wrong"],
            "true": ["correct", "accurate", "verified", "confirmed"],
            "refuted": ["debunked", "disproven", "contradicted"],
            "supported": ["confirmed", "validated", "corroborated"],
            # Financial terms
            "stock": ["shares", "equity", "securities"],
            "investment": ["capital", "funding", "portfolio"],
            "profit": ["gain", "return", "earnings"],
            "loss": ["decline", "drop", "deficit"],
            "market": ["exchange", "trading", "financial"],
            # Verification terms
            "verify": ["confirm", "validate", "authenticate", "check"],
            "source": ["reference", "citation", "origin"],
            "fact": ["truth", "reality", "data"],
        }

    def expand_query(self, query: str) -> str:
        """
        Expand query with crypto-specific synonyms.

        Args:
            query: Original query string

        Returns:
            Expanded query with synonyms
        """
        query_lower = query.lower()
        expanded_terms = [query]

        for term, synonyms in self.glossary.items():
            if term in query_lower:
                # Add first two synonyms
                expanded_terms.extend(synonyms[:2])

        return " ".join(expanded_terms)


class KnowledgeAugmentedRetriever:
    """
    Main retrieval system implementing hybrid RRF-based retrieval.

    Combines:
    - BGE embeddings for semantic search (via FAISS) - Stage 1a
    - BM25 for lexical matching - Stage 1b
    - Reciprocal Rank Fusion (RRF) to combine rankings - Stage 2
    - Temporal scoring for recency awareness - Stage 3
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        use_query_expansion: bool = True,
        rrf_k: int = 60,
        index_path: str = None,
    ):
        """
        Initialize the retrieval system.

        Args:
            embedding_model: Name of sentence transformer model
            alpha: RRF vs temporal weight (default: 0.7)
            lambda_decay: Recency decay factor (default: 0.1)
            gamma: Recency vs cyclicity mix (default: 0.5)
            use_query_expansion: Enable query expansion
            rrf_k: RRF constant k (default: 60)
            index_path: Path to save/load FAISS index
        """
        self.alpha = alpha
        self.index_path = index_path
        self.rrf_k = rrf_k

        # Initialize components
        self.temporal_scorer = TemporalScorer(
            alpha=alpha, lambda_decay=lambda_decay, gamma=gamma
        )

        self.query_expander = QueryExpander() if use_query_expansion else None
        self.preprocessor = TextPreprocessor()

        # Initialize embedding model (Bi-Encoder for FAISS)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            self.encoder = None
            self.embedding_dim = 384  # Default

        # Cross-Encoder removed - using RRF-based hybrid retrieval instead

        # Storage
        self.documents = []
        self.document_embeddings = None
        self.bm25 = None
        self.faiss_index = None

        logger.info("KnowledgeAugmentedRetriever initialized")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenization using preprocessor"""
        return self.preprocessor.preprocess(text)

    def index_documents(
        self,
        documents: List[Dict],
        text_field: str = "text",
        id_field: str = "id",
        timestamp_field: str = "timestamp",
    ) -> None:
        """
        Index documents for retrieval.

        Args:
            documents: List of document dictionaries
            text_field: Key for document text
            id_field: Key for document ID
            timestamp_field: Key for timestamp
        """
        logger.info(f"Indexing {len(documents)} documents...")

        self.documents = []
        texts = []

        for doc in documents:
            doc_entry = {
                "id": doc.get(id_field, str(len(self.documents))),
                "text": doc.get(text_field, ""),
                "timestamp": doc.get(timestamp_field, datetime.now()),
                "metadata": {
                    k: v
                    for k, v in doc.items()
                    if k not in [text_field, id_field, timestamp_field]
                },
            }
            self.documents.append(doc_entry)
            texts.append(doc_entry["text"])

        # Build BM25 index
        # Note: Using BM25Okapi (standard BM25). Paper mentions BM25+ in Eq.8.
        # BM25+ adds delta term to avoid negative scores, but difference is minor.
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)

        # Build FAISS index
        if self.encoder is not None:
            logger.info("Creating embeddings...")
            self.document_embeddings = self.encoder.encode(
                texts, show_progress_bar=True, convert_to_numpy=True
            )

            if FAISS_AVAILABLE:
                # Create FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                # Normalize for cosine similarity
                faiss.normalize_L2(self.document_embeddings)
                self.faiss_index.add(self.document_embeddings)
                logger.info(
                    f"FAISS index created with {self.faiss_index.ntotal} vectors"
                )

        logger.info(f"Indexing complete: {len(self.documents)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_temporal: bool = True,
        expand_query: bool = True,
        use_semantic: bool = True,
        rrf_top_k: int = 20,
    ) -> List[RetrievalResult]:

        if not self.documents:
            logger.warning("No documents indexed")
            return []

        # Update reference date to current query time
        self.temporal_scorer.reference_date = datetime.now(timezone.utc)

        # Query expansion
        if expand_query and self.query_expander:
            expanded_query = self.query_expander.expand_query(query)
        else:
            expanded_query = query

        # Stage 1a: BM25 scoring for all documents
        tokenized_query = self._tokenize(expanded_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Create BM25 rankings (higher score = lower rank number)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_ranks = {idx: rank for rank, (idx, score) in enumerate(bm25_ranked)}

        # Stage 1b: FAISS semantic search (if available)
        faiss_ranks = {}
        if use_semantic and self.encoder is not None and self.faiss_index is not None:
            query_embedding = self.encoder.encode(
                [expanded_query], convert_to_numpy=True
            )
            faiss.normalize_L2(query_embedding)

            # Search all documents to get complete ranking
            n_docs = len(self.documents)
            distances, indices = self.faiss_index.search(query_embedding, n_docs)
            faiss_ranks = {idx: rank for rank, idx in enumerate(indices[0].tolist())}
        else:
            # No FAISS available: use uniform ranks
            faiss_ranks = {i: i for i in range(len(self.documents))}

        # Stage 2: Reciprocal Rank Fusion (RRF)
        # RRF(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))
        rrf_scores = {}
        for i in range(len(self.documents)):
            bm25_rank = bm25_ranks.get(i, len(self.documents))
            faiss_rank = faiss_ranks.get(i, len(self.documents))
            rrf_scores[i] = (1.0 / (self.rrf_k + bm25_rank)) + (
                1.0 / (self.rrf_k + faiss_rank)
            )

        # Sort by RRF score and take top rrf_top_k (default 20)
        rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        rrf_candidates = rrf_ranked[:rrf_top_k]

        # Normalize RRF scores to [0,1] for final score calculation
        max_rrf = max(score for idx, score in rrf_candidates) if rrf_candidates else 1.0
        rrf_scores_norm = {idx: score / max_rrf for idx, score in rrf_candidates}

        # Build group histories from FULL corpus for cyclicity calculation
        docs_by_group = {}
        for i, doc in enumerate(self.documents):
            group_key = (
                doc["metadata"].get("type")
                or doc["metadata"].get("source")
                or "default"
            )
            if group_key not in docs_by_group:
                docs_by_group[group_key] = []
            docs_by_group[group_key].append((i, doc["timestamp"]))

        # Stage 3: Temporal scoring on RRF candidates
        final_scores = []

        for idx, rrf_score in rrf_candidates:
            doc = self.documents[idx]

            if use_temporal:
                # Get group-specific timestamps for cyclicity
                group_key = (
                    doc["metadata"].get("type")
                    or doc["metadata"].get("source")
                    or "default"
                )
                group_timestamps = [
                    ts for idx_g, ts in docs_by_group.get(group_key, [])
                ]

                # Calculate temporal score with group-based cyclicity + adaptive λ
                temporal, recency, cyclicity = (
                    self.temporal_scorer.calculate_temporal_score(
                        doc["timestamp"], group_timestamps, use_adaptive_lambda=True
                    )
                )
                # Final Score = α × RRF + (1-α) × Temporal
                score = self.alpha * rrf_scores_norm[idx] + (1 - self.alpha) * temporal
            else:
                recency, cyclicity = 0.5, 0.5
                score = rrf_scores_norm[idx]

            final_scores.append(
                {
                    "index": idx,
                    "score": score,
                    "rrf_score": rrf_scores_norm[idx],
                    "recency_score": recency,
                    "cyclicity_score": cyclicity,
                }
            )

        # Sort by final score and take top_k (default 10)
        final_scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = final_scores[:top_k]

        # Build final results
        results = []
        for item in top_results:
            doc = self.documents[item["index"]]
            results.append(
                RetrievalResult(
                    document_id=doc["id"],
                    text=doc["text"],
                    score=item["score"],
                    rrf_score=item["rrf_score"],  # Changed from bm25_score
                    recency_score=item["recency_score"],
                    cyclicity_score=item["cyclicity_score"],
                    timestamp=doc["timestamp"],
                    metadata=doc["metadata"],
                )
            )

        return results

    def save_index(self, path: str = None) -> None:
        """Save the index to disk"""
        path = path or self.index_path
        if path is None:
            logger.warning("No path specified for saving index")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "documents": self.documents,
            "document_embeddings": self.document_embeddings,
            "alpha": self.alpha,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Index saved to {path}")

    def load_index(self, path: str = None) -> None:
        """Load the index from disk"""
        path = path or self.index_path
        if path is None or not os.path.exists(path):
            logger.warning(f"Index file not found: {path}")
            return

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.documents = state["documents"]
        self.document_embeddings = state.get("document_embeddings")
        self.alpha = state.get("alpha", self.alpha)

        # Rebuild indices
        texts = [doc["text"] for doc in self.documents]
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)

        if self.document_embeddings is not None and FAISS_AVAILABLE:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            # Normalize for cosine similarity (must match indexing behavior)
            faiss.normalize_L2(self.document_embeddings)
            self.faiss_index.add(self.document_embeddings)

        logger.info(f"Index loaded from {path}: {len(self.documents)} documents")


if __name__ == "__main__":
    print("KnowledgeAugmentedRetriever module. Use via pipeline.")
