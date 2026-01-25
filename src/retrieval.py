import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import pickle
import os

from rank_bm25 import BM25Okapi
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using numpy-based similarity")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available, using TF-IDF")

from scipy.fft import fft
from scipy.signal import find_peaks


@dataclass
class RetrievalResult:
    document_id: str
    text: str
    score: float
    bm25_score: float
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
        reference_date: datetime = None  # Will be set dynamically per query if None
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
        
        logger.info(f"TemporalScorer initialized: α={alpha}, λ={lambda_decay}, γ={gamma}")
    
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
        self,
        timestamps: List[datetime],
        pattern_type: str = None
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
        power_spectrum = np.abs(fft_result[:len(fft_result)//2]) ** 2
        
        # Find dominant frequencies
        peaks, properties = find_peaks(power_spectrum, height=np.mean(power_spectrum))
        
        if len(peaks) > 0:
            # Calculate cyclicity based on peak strength
            max_peak_power = max(properties['peak_heights'])
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
        use_adaptive_lambda: bool = True
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
    Main retrieval system implementing Section 3.1.
    
    Combines:
    - BM25 for lexical matching
    - BGE embeddings for semantic search (via FAISS)
    - Temporal scoring for recency awareness
    - Cycle-aware scoring for pattern detection
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        alpha: float = 0.7,
        lambda_decay: float = 0.1,
        gamma: float = 0.5,
        use_query_expansion: bool = True,
        index_path: str = None
    ):
        """
        Initialize the retrieval system.
        
        Args:
            embedding_model: Name of sentence transformer model
            alpha: BM25 vs temporal weight (paper: 0.7)
            lambda_decay: Recency decay factor (paper: 0.1)
            gamma: Recency vs cyclicity mix (paper: 0.5)
            use_query_expansion: Enable crypto query expansion
            index_path: Path to save/load FAISS index
        """
        self.alpha = alpha
        self.index_path = index_path
        
        # Initialize components
        self.temporal_scorer = TemporalScorer(
            alpha=alpha,
            lambda_decay=lambda_decay,
            gamma=gamma
        )
        
        self.query_expander = QueryExpander() if use_query_expansion else None
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            self.encoder = None
            self.embedding_dim = 384  # Default
        
        # Storage
        self.documents = []
        self.document_embeddings = None
        self.bm25 = None
        self.faiss_index = None
        
        logger.info("KnowledgeAugmentedRetriever initialized")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def index_documents(
        self,
        documents: List[Dict],
        text_field: str = "text",
        id_field: str = "id",
        timestamp_field: str = "timestamp"
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
                "metadata": {k: v for k, v in doc.items() 
                           if k not in [text_field, id_field, timestamp_field]}
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
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            if FAISS_AVAILABLE:
                # Create FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                # Normalize for cosine similarity
                faiss.normalize_L2(self.document_embeddings)
                self.faiss_index.add(self.document_embeddings)
                logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")
        
        logger.info(f"Indexing complete: {len(self.documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_temporal: bool = True,
        expand_query: bool = True,
        use_semantic: bool = True,
        candidate_pool_size: int = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        PAPER-FAITHFUL IMPLEMENTATION:
        
        Stage 1 (Optional): FAISS dense retrieval for candidates
        - Uses BGE embeddings indexed via FAISS
        - Returns top-N candidates by semantic similarity
        
        Stage 2: Rerank with Equation (1) and (8):
        - Eq.1: Score(q,di) = α·BM25(q,di) + (1-α)·Temporal(di)
        - Eq.8: Temporal = γ·Recency + (1-γ)·Cyclicity
        - Eq.9: λt = Sigmoid(Trend)·λbase (adaptive decay)
        
        NOTE: Semantic similarity is used ONLY for candidate retrieval,
        NOT mixed into final scoring (per paper Eq.1 which only mentions BM25).
        
        KNOWN DEVIATIONS FROM PAPER:
        - Using BM25Okapi instead of BM25+ (minor difference)
        - Normalize BM25 to [0,1] for score stability
        - Query expansion adapted for fact-checking (not crypto-specific)
        
        Args:
            query: Search query
            top_k: Number of final results
            use_temporal: Apply temporal scoring (Eq.8/9)
            expand_query: Expand query with synonyms
            use_semantic: Use FAISS for candidate retrieval
            candidate_pool_size: FAISS pool size. None = score all docs
            
        Returns:
            List of RetrievalResult sorted by Eq.1 score
        """
        if not self.documents:
            logger.warning("No documents indexed")
            return []
        
        # Update reference date to current query time (fix G)
        self.temporal_scorer.reference_date = datetime.now(timezone.utc)
        
        # Query expansion
        if expand_query and self.query_expander:
            expanded_query = self.query_expander.expand_query(query)
        else:
            expanded_query = query
        
        # Stage 1: Get candidate pool via FAISS (dense retrieval)
        # Paper: "dynamic databases indexed via FAISS"
        if use_semantic and self.encoder is not None and self.faiss_index is not None:
            if candidate_pool_size is None:
                # Paper-faithful: score all documents
                candidate_indices = set(range(len(self.documents)))
            else:
                # Practical: FAISS semantic search for top-N candidates
                query_embedding = self.encoder.encode([expanded_query], convert_to_numpy=True)
                faiss.normalize_L2(query_embedding)
                
                pool_size = min(candidate_pool_size, len(self.documents))
                distances, indices = self.faiss_index.search(query_embedding, pool_size)
                candidate_indices = set(indices[0].tolist())
        else:
            # No FAISS: score all documents
            candidate_indices = set(range(len(self.documents)))
        
        # Stage 2: BM25 scoring
        # Note: Using BM25Okapi (standard). Paper Eq.8 mentions BM25+.
        tokenized_query = self._tokenize(expanded_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 to [0,1] for score stability
        # (Paper doesn't specify normalization, but needed for α weighting)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = bm25_scores / max_bm25
        
        # Build group histories from FULL corpus (fix E)
        # Paper: cyclicity should reflect full pattern history, not just candidates
        docs_by_group = {}
        for i, doc in enumerate(self.documents):  # Full corpus
            group_key = doc["metadata"].get("type") or doc["metadata"].get("source") or "default"
            if group_key not in docs_by_group:
                docs_by_group[group_key] = []
            docs_by_group[group_key].append((i, doc["timestamp"]))
        
        #  Stage 3: Rerank candidates with Paper Eq.1 + Eq.8
        final_scores = []
        
        for i in candidate_indices:
            doc = self.documents[i]
            
            if use_temporal:
                # Get group-specific timestamps for cyclicity (Eq.8)
                group_key = doc["metadata"].get("type") or doc["metadata"].get("source") or "default"
                group_timestamps = [ts for idx, ts in docs_by_group.get(group_key, [])]
                
                # Calculate temporal score with group-based cyclicity + adaptive λ (Eq.9)
                temporal, recency, cyclicity = self.temporal_scorer.calculate_temporal_score(
                    doc["timestamp"],
                    group_timestamps,
                    use_adaptive_lambda=True
                )
                # Paper Equation (1): Score = α·BM25 + (1-α)·Temporal
                # where Temporal = γ·Recency + (1-γ)·Cyclicity from Eq.8
                score = self.alpha * bm25_scores_norm[i] + (1 - self.alpha) * temporal
            else:
                recency, cyclicity = 0.5, 0.5
                score = bm25_scores_norm[i]
            
            final_scores.append({
                "index": i,
                "score": score,
                "bm25_score": bm25_scores_norm[i],
                "recency_score": recency,
                "cyclicity_score": cyclicity
            })
        
        # Step 4: Sort and get top-k
        final_scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = final_scores[:top_k]
        
        # Build results
        results = []
        for item in top_results:
            doc = self.documents[item["index"]]
            results.append(RetrievalResult(
                document_id=doc["id"],
                text=doc["text"],
                score=item["score"],
                bm25_score=item["bm25_score"],
                recency_score=item["recency_score"],
                cyclicity_score=item["cyclicity_score"],
                timestamp=doc["timestamp"],
                metadata=doc["metadata"]
            ))
        
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
            "alpha": self.alpha
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str = None) -> None:
        """Load the index from disk"""
        path = path or self.index_path
        if path is None or not os.path.exists(path):
            logger.warning(f"Index file not found: {path}")
            return
        
        with open(path, 'rb') as f:
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
