"""
Crypto Claim Verification Pipeline

Goal:
Input: one post/comment/message (Reddit/Telegram)
Output:
  - label ∈ {SUPPORTED, REFUTED, NEI}
  - evidence: top-k retrieved evidence (links/snippets)
  - confidence: reliability score

The retrieval score uses the paper's formula:
Score(q, d_i) = α · BM25(q, d_i) + (1 − α) · Recency(d_i)
Recency(d_i) = e^(−λt), with α=0.7, λ=0.1
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from .retrieval import KnowledgeAugmentedRetriever
from .llm_scorer import LLMScorer

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


@dataclass
class PipelineConfig:
    """Configuration for the claim verification pipeline"""
    # Retrieval parameters (Section 3.1)
    alpha: float = 0.7  # BM25 vs temporal weight
    lambda_decay: float = 0.1  # Recency decay factor
    gamma: float = 0.5  # Recency vs cyclicity mix
    top_k_retrieval: int = 5  # Number of documents to retrieve
    
    # General
    seed: int = 42
    verbose: bool = True

    # LLM usage for claim verification
    llm_model_name: str = "meta-llama/Llama-3.1-8B"
    device: str = "cpu"
    use_llm: bool = False
    support_threshold: float = 0.7


@dataclass 
class PredictionResult:
    """Container for single prediction"""
    text: str
    predicted_label: str
    confidence: float
    evidence: List[Dict]
    processing_time_ms: float


class CryptoClaimVerificationPipeline:
    """
    Pipeline for crypto-focused claim verification.
    Output labels: SUPPORTED / REFUTED / NEI.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize the pipeline with given configuration."""
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.retriever = None
        self.llm_scorer = None
        
        self.is_fitted = False
        
        logger.info("CryptoClaimVerificationPipeline initialized")
    
    def build(self) -> None:
        """Build all pipeline components."""
        logger.info("Building pipeline components...")
        
        # Retrieval system (Section 3.1)
        self.retriever = KnowledgeAugmentedRetriever(
            alpha=self.config.alpha,
            lambda_decay=self.config.lambda_decay,
            gamma=self.config.gamma,
            use_query_expansion=True
        )

        logger.info("Pipeline components built")
    
    def fit(
        self,
        knowledge_base: List[Dict],
        training_data: Optional[object] = None
    ) -> None:
        """
        Fit the pipeline on knowledge base and optional training data.
        
        Args:
            knowledge_base: List of scam templates/patterns for retrieval
            training_data: Optional labeled training data for threshold tuning
        """
        if self.retriever is None:
            self.build()
        
        logger.info(f"Fitting pipeline on {len(knowledge_base)} knowledge items...")
        
        # Index knowledge base for retrieval
        self.retriever.index_documents(
            knowledge_base,
            text_field="example_text" if "example_text" in knowledge_base[0] else "text",
            timestamp_field="timestamp"
        )
        
        # Fit embedding model
        texts = [doc.get("example_text", doc.get("text", "")) for doc in knowledge_base]
        self.embedding_model.fit(texts)
        
        self.is_fitted = True
        logger.info("Pipeline fitted successfully")

    def _predict_single(
        self,
        text: str
    ) -> Tuple[str, float, List[Dict]]:
        """Verify single claim."""
        
        retrieval_results = self.retriever.retrieve(text, top_k=self.config.top_k_retrieval)
        evidence = []
        for r in retrieval_results:
            evidence.append({
                "text": r.text[:300],
                "score": r.score,
                "timestamp": r.timestamp,
                "source": r.metadata.get("source", r.metadata.get("subreddit", "unknown")),
                "link": r.metadata.get("link"),
            })

        if self.config.use_llm:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch + transformers required for LLM verification.")

            if self.llm_scorer is None:
                self.llm_scorer = LLMScorer(
                    model_name=self.config.llm_model_name,
                    device=self.config.device,
                    labels=["SUPPORTED", "REFUTED", "NEI"],
                    prompt_template=(
                        "You are verifying a crypto/finance claim.\n"
                        "Claim: {text}\n"
                        "Answer with one label: SUPPORTED, REFUTED, or NEI."
                    ),
                )
            logits = self.llm_scorer.score_texts([text])
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            labels = ["SUPPORTED", "REFUTED", "NEI"]
            idx = int(np.argmax(probs))
            label = labels[idx]
            confidence = float(probs[idx])
        else:
            top_score = retrieval_results[0].score if retrieval_results else 0.0
            label = "SUPPORTED" if top_score >= self.config.support_threshold else "NEI"
            confidence = float(top_score)

        return label, confidence, evidence
    
    def predict(
        self,
        texts: List[str]
    ) -> List[PredictionResult]:
        """
        Make predictions for multiple texts.
        
        Args:
            texts: List of text strings to classify
            use_optimal_threshold: Whether to use optimized threshold
            
        Returns:
            List of PredictionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        results = []
        for text in texts:
            start_time = time.time()

            label, confidence, evidence = self._predict_single(text)
            processing_time = (time.time() - start_time) * 1000

            results.append(PredictionResult(
                text=text,
                predicted_label=label,
                confidence=confidence,
                evidence=evidence,
                processing_time_ms=processing_time
            ))
        
        return results

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("Claim verification evaluation is not implemented yet.")
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        use_optimal_threshold: bool = True
    ) -> EvaluationResult:
        """
        Evaluate pipeline on test data.
        Produces metrics similar to Tables 1-7 in the paper.
        
        Args:
            test_data: DataFrame with 'text' and 'label' columns
            use_optimal_threshold: Whether to use optimized threshold
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating on {len(test_data)} samples...")
        
        y_true = test_data["label"].values
        texts = test_data["text"].tolist()
        
        start_time = time.time()
        predictions = self.predict(texts, use_optimal_threshold)
        total_time = time.time() - start_time
        
        y_pred = [p.predicted_label for p in predictions]
        y_pred_proba = [p.probability for p in predictions]
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Compute Fβ
        beta_sq = self.config.threshold_beta ** 2
        if precision + recall > 0:
            f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        else:
            f_beta = 0.0
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        avg_latency = total_time / len(texts) * 1000
        
        # Estimate hallucination rate (based on confidence)
        confidences = [p.confidence for p in predictions]
        # Low confidence predictions on wrong samples = potential hallucinations
        wrong_preds = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
        if wrong_preds:
            halluc_confidences = [confidences[i] for i in wrong_preds]
            hallucination_rate = np.mean([c for c in halluc_confidences if c > 0.5]) * 100
        else:
            hallucination_rate = 0.0
        
        factual_consistency = 100 - hallucination_rate
        
        result = EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            f_beta=f_beta,
            confusion_matrix=conf_matrix,
            latency_ms=avg_latency,
            hallucination_rate=hallucination_rate,
            factual_consistency=factual_consistency,
            metadata={
                "num_samples": len(test_data),
                "threshold": self.optimal_threshold if use_optimal_threshold else 0.5,
                "avg_confidence": np.mean(confidences),
                "predictions": predictions
            }
        )
        
        return result
    
    def run_ablation_study(
        self,
        test_data: pd.DataFrame
    ) -> Dict[str, EvaluationResult]:
        """
        Run ablation study similar to Table 7 in the paper.
        Tests contribution of each component.
        
        Returns:
            Dictionary mapping variant names to results
        """
        logger.info("Running ablation study...")
        results = {}
        
        # Full system
        results["Full System"] = self.evaluate(test_data)
        
        # Without temporal scoring (α=1.0)
        original_alpha = self.config.alpha
        self.retriever.alpha = 1.0
        results["w/o Temporal Scoring"] = self.evaluate(test_data)
        self.retriever.alpha = original_alpha
        
        # Without confidence fusion (β=1.0 - only LM)
        original_beta = self.fusion_classifier.beta
        self.fusion_classifier.beta = 1.0
        results["w/o Confidence Fusion"] = self.evaluate(test_data)
        self.fusion_classifier.beta = original_beta
        
        # Without threshold adaptation (fixed τ=0.5)
        results["w/o Threshold Adapt"] = self.evaluate(test_data, use_optimal_threshold=False)
        
        return results
    
    def print_ablation_table(self, ablation_results: Dict[str, EvaluationResult]) -> None:
        """Print ablation study in paper format (Table 7)."""
        print("\n" + "=" * 60)
        print("Ablation Study Results (Table 7 format)")
        print("=" * 60)
        print(f"{'Variant':<25} {'F1 Score':>15} {'Change':>15}")
        print("-" * 60)
        
        baseline_f1 = ablation_results["Full System"].f1
        
        for variant, result in ablation_results.items():
            change = (result.f1 - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
            change_str = f"{change:+.1f}%" if variant != "Full System" else "-"
            print(f"{variant:<25} {result.f1:>15.4f} {change_str:>15}")
        
        print("=" * 60)


