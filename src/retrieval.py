"""
Knowledge-Augmented Retrieval System

Implements Section 3.1 of the paper:
- Temporal relevance scoring for scam templates
- Score(q, di) = α · BM25(q, di) + (1 − α) · Recency(di)
- FAISS indexing with BGE embeddings

Also implements Section 3.5:
- Semantic-Aware Retrieval with crypto-specific query expansion
- Cycle-Aware Scoring with FFT for detecting repeating patterns
"""

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
    """Container for retrieval results"""
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
        alpha: float = 0.7,  # Trade-off between semantic and temporal
        lambda_decay: float = 0.1,  # Decay factor for recency
        gamma: float = 0.5,  # Mix between recency and cyclicity
        reference_date: datetime = None
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
        # Use timezone-aware UTC to avoid naive/aware datetime errors
        self.reference_date = reference_date or datetime.now(timezone.utc)
        
        logger.info(f"TemporalScorer initialized: α={alpha}, λ={lambda_decay}, γ={gamma}")
    
    def calculate_recency(self, timestamp: datetime) -> float:
        """
        Calculate recency score using exponential decay: e^(-λt)
        
        Args:
            timestamp: Document timestamp
            
        Returns:
            Recency score between 0 and 1
        """
        # Normalize timezone awareness
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ref = self.reference_date
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)

        days_diff = (ref - ts).days
        days_diff = max(0, days_diff)  # Handle future dates
        return np.exp(-self.lambda_decay * days_diff / 30)  # Normalize by month
    
    def calculate_cyclicity(
        self,
        timestamps: List[datetime],
        pattern_type: str = None
    ) -> float:
        """
        Calculate cyclicity score using FFT to detect repeating patterns.
        Implements Section 3.5's cycle-aware scoring.
        
        Args:
            timestamps: List of document timestamps for pattern analysis
            pattern_type: Optional pattern type hint
            
        Returns:
            Cyclicity score between 0 and 1
        """
        if len(timestamps) < 10:
            return 0.5  # Default for insufficient data
        
        # Convert to daily occurrence counts
        min_date = min(timestamps)
        max_date = max(timestamps)
        date_range = (max_date - min_date).days + 1
        
        if date_range < 7:
            return 0.5
        
        # Create time series
        daily_counts = np.zeros(date_range)
        for ts in timestamps:
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
    
    def calculate_temporal_score(
        self,
        timestamp: datetime,
        historical_timestamps: List[datetime] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate combined temporal score.
        Implements Equation (8): Score = α·BM25 + (1-α)·[γ·Recency + (1-γ)·Cyclicity]
        
        Returns:
            Tuple of (temporal_score, recency_score, cyclicity_score)
        """
        recency = self.calculate_recency(timestamp)
        
        if historical_timestamps and len(historical_timestamps) > 10:
            cyclicity = self.calculate_cyclicity(historical_timestamps)
        else:
            cyclicity = 0.5
        
        temporal_score = self.gamma * recency + (1 - self.gamma) * cyclicity
        
        return temporal_score, recency, cyclicity


class CryptoQueryExpander:
    """
    Implements crypto-specific query expansion from Section 3.5.
    Uses CryptoGlossary for synonym expansion.
    """
    
    def __init__(self, glossary: Dict[str, List[str]] = None):
        """
        Initialize query expander with crypto glossary.
        
        Args:
            glossary: Dictionary mapping terms to synonyms
        """
        self.glossary = glossary or self._default_glossary()
        logger.info(f"QueryExpander initialized with {len(self.glossary)} terms")
    
    def _default_glossary(self) -> Dict[str, List[str]]:
        """Default crypto-specific glossary"""
        return {
            "rug pull": ["exit scam", "liquidity drain", "developer abandonment"],
            "pump and dump": ["market manipulation", "price inflation", "coordinated selling"],
            "dusting attack": ["wallet spam", "tracking attack", "dust transaction"],
            "phishing": ["credential theft", "fake website", "social engineering"],
            "ponzi": ["pyramid scheme", "investment fraud", "mlm scam"],
            "giveaway scam": ["double your crypto", "celebrity impersonation", "send to receive"],
            "honeypot": ["sell restriction", "locked token", "contract trap"],
            "scam": ["fraud", "theft", "deception", "fake"],
            "guaranteed": ["certain", "assured", "promised", "100%"],
            "verify": ["confirm", "validate", "authenticate", "check"],
            "urgent": ["immediate", "time-sensitive", "act now", "limited time"],
            "airdrop": ["free tokens", "token distribution", "giveaway"],
            "seed phrase": ["recovery phrase", "mnemonic", "backup words", "private key"],
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
        
        self.query_expander = CryptoQueryExpander() if use_query_expansion else None
        
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
        expand_query: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        Implements Equation (1): Score(q, di) = α·BM25 + (1-α)·Recency
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_temporal: Whether to apply temporal scoring
            expand_query: Whether to expand query with synonyms
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.documents:
            logger.warning("No documents indexed")
            return []
        
        # Query expansion
        if expand_query and self.query_expander:
            expanded_query = self.query_expander.expand_query(query)
        else:
            expanded_query = query
        
        # BM25 scores
        tokenized_query = self._tokenize(expanded_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = bm25_scores / max_bm25
        
        # Semantic scores (if available)
        if self.encoder is not None and self.faiss_index is not None:
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Get all scores for combining
            distances, indices = self.faiss_index.search(
                query_embedding,
                min(len(self.documents), len(self.documents))
            )
            
            semantic_scores = np.zeros(len(self.documents))
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    semantic_scores[idx] = (dist + 1) / 2  # Convert to 0-1
        else:
            semantic_scores = bm25_scores_norm
        
        # Combine BM25 and semantic
        combined_lexical_semantic = 0.5 * bm25_scores_norm + 0.5 * semantic_scores
        
        # Calculate final scores with temporal component
        final_scores = []
        all_timestamps = [doc["timestamp"] for doc in self.documents]
        
        for i, doc in enumerate(self.documents):
            if use_temporal:
                temporal, recency, cyclicity = self.temporal_scorer.calculate_temporal_score(
                    doc["timestamp"],
                    all_timestamps
                )
                # Equation (1): Score = α·BM25 + (1-α)·Temporal
                score = self.alpha * combined_lexical_semantic[i] + (1 - self.alpha) * temporal
            else:
                recency, cyclicity = 0.5, 0.5
                score = combined_lexical_semantic[i]
            
            final_scores.append({
                "index": i,
                "score": score,
                "bm25_score": bm25_scores_norm[i],
                "recency_score": recency,
                "cyclicity_score": cyclicity
            })
        
        # Sort and get top-k
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
            self.faiss_index.add(self.document_embeddings)
        
        logger.info(f"Index loaded from {path}: {len(self.documents)} documents")


if __name__ == "__main__":
    print("KnowledgeAugmentedRetriever module. Use via pipeline.")
