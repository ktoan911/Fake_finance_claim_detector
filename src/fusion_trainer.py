
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .retrieval import KnowledgeAugmentedRetriever
from .fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
from .llm_scorer import LLMScorer
from .config import PROMPT_TEMPLATE, LABEL_LIST, LABEL_TO_ID


@dataclass
class FusionTrainingConfig:
    model_name: str = "artifacts/lora_llm"
    device: str = "auto"
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 1e-4
    top_k: int = 5
    alpha: float = 0.7
    lambda_decay: float = 0.1
    gamma: float = 0.5
    initial_beta: float = 0.5
    lambda_reg: float = 0.01  # Only used in fusion layer, not doubled
    max_length: int = 1024
    evidence_mode: str = "retrieved"  # "gold" or "retrieved" - per paper, should be "retrieved"
    label_list: List[str] = field(default_factory=lambda: LABEL_LIST)


def _normalize_label(label_value) -> int:
    """Convert label to integer ID."""
    if isinstance(label_value, (int, float)):
        return int(label_value)
    
    label_upper = str(label_value).upper().strip()
    
    if label_upper in ["SUPPORTED", "TRUE", "LEGIT", "0"]:
        return 0
    if label_upper in ["REFUTED", "FALSE", "SCAM", "1"]:
        return 1
    if label_upper in ["NEI", "UNSURE", "NEUTRAL", "UNKNOWN", "2"]:
        return 2
    
    return 2


def _build_retrieval_features(
    retriever: KnowledgeAugmentedRetriever,
    text: str,
    top_k: int,
) -> tuple:
    """Returns (features, retrieved_evidence_text)."""
    results = retriever.retrieve(text, top_k=top_k)
    features = []
    evidence_texts = []
    
    for r in results:
        features.append([r.score, r.bm25_score, r.recency_score, r.cyclicity_score])
        evidence_texts.append(r.text)
    
    if len(features) < top_k:
        features.extend([[0.0, 0.0, 0.0, 0.0]] * (top_k - len(features)))
    
    # Combine retrieved evidence into single string
    combined_evidence = " ".join(evidence_texts) if evidence_texts else ""
    
    return np.array(features, dtype=np.float32), combined_evidence


def train_fusion_from_dataframe(
    knowledge_base: List[Dict],
    labeled_df,
    config: Optional[FusionTrainingConfig] = None,
    save_path: str = "artifacts/fusion_model.pt",
) -> str:
    """
    Train fusion MLP + beta from a labeled pandas DataFrame.
    
    Per paper Eq.2: pfinal = β·pLM + (1-β)·MLP(pret)
    Uses LOGITS, not probabilities, for proper fusion.
    
    Args:
        knowledge_base: List of dicts with text/timestamp for retrieval
        labeled_df: DataFrame with text, evidence, label columns
        config: Training configuration
        save_path: Where to save trained model
        
    evidence_mode options:
        - "retrieved": Use evidence from retriever (paper-accurate)
        - "gold": Use evidence column from dataframe (for debugging)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for fusion training.")

    config = config or FusionTrainingConfig()
    
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Training fusion on device: {config.device}")
    logger.info(f"Evidence mode: {config.evidence_mode}")

    if labeled_df is None or labeled_df.empty:
        raise ValueError("Labeled DataFrame is empty.")

    # Initialize retriever
    retriever = KnowledgeAugmentedRetriever(
        alpha=config.alpha,
        lambda_decay=config.lambda_decay,
        gamma=config.gamma,
        use_query_expansion=True
    )
    retriever.index_documents(knowledge_base, text_field="text", timestamp_field="timestamp")
    logger.info(f"Indexed {len(knowledge_base)} documents in retriever")

    # Initialize LLM scorer (returns LOGITS per paper Eq.2)
    llm = LLMScorer(
        model_name=config.model_name,
        device=config.device,
        max_length=config.max_length,
        labels=config.label_list,
        prompt_template=PROMPT_TEMPLATE,
    )

    # Initialize retrieval encoder
    retrieval_encoder = RetrievalFeatureEncoder(
        num_retrieved=config.top_k,
        score_features=4,
        hidden_dim=64,
        output_dim=64
    ).to(config.device)

    # Initialize fusion layer
    num_classes = len(config.label_list)
    fusion = ConfidenceAwareFusion(
        retrieval_input_dim=64,
        hidden_dim=128,
        num_classes=num_classes,
        initial_beta=config.initial_beta,
        lambda_reg=config.lambda_reg,  # Regularization only here, not doubled
    ).to(config.device)

    # Optimizer for fusion components only (LLM is frozen)
    params = list(retrieval_encoder.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)

    # Prepare data
    texts = labeled_df["text"].tolist()
    gold_evidences = labeled_df["evidence"].tolist()
    labels = [_normalize_label(l) for l in labeled_df["label"].tolist()]
    
    logger.info(f"Training samples: {len(texts)}")
    logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    fusion.train()
    retrieval_encoder.train()

    for epoch in range(config.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for i in range(0, len(texts), config.batch_size):
            batch_texts = texts[i:i + config.batch_size]
            batch_gold_evidences = gold_evidences[i:i + config.batch_size]
            batch_labels = labels[i:i + config.batch_size]

            # Build retrieval features and get retrieved evidence
            batch_features = []
            batch_retrieved_evidences = []
            for t in batch_texts:
                feats, retrieved_evidence = _build_retrieval_features(retriever, t, config.top_k)
                batch_features.append(feats)
                batch_retrieved_evidences.append(retrieved_evidence)
            
            # Choose evidence source based on config
            if config.evidence_mode == "retrieved":
                batch_evidences = batch_retrieved_evidences  # Paper-accurate
            else:
                batch_evidences = batch_gold_evidences  # Gold labels for debugging

            # Get LLM LOGITS (not probabilities!) per paper Eq.2
            lm_logits = llm.score_logits(batch_texts, batch_evidences)  # [batch, num_classes]

            # Encode retrieval features
            retrieval_scores = torch.tensor(batch_features, dtype=torch.float32, device=config.device)
            retrieval_features = retrieval_encoder(retrieval_scores)
            
            # Fusion: β·pLM + (1-β)·MLP(pret) per Eq.2
            output = fusion(lm_logits, retrieval_features)
            fused_logits = output.fused_logits

            # Cross-entropy loss
            targets = torch.tensor(batch_labels, dtype=torch.long, device=config.device)
            ce_loss = F.cross_entropy(fused_logits, targets)
            
            # Add beta regularization (L = CE + λ||β||²)
            beta_reg = fusion.lambda_reg * (fusion.beta ** 2)
            loss = ce_loss + beta_reg

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(fused_logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += len(batch_labels)
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = correct / total if total > 0 else 0
        beta_val = fusion.beta.item()
        logger.info(f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f} - acc: {accuracy:.4f} - β: {beta_val:.4f}")

    # Save model
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "retrieval_encoder": retrieval_encoder.state_dict(),
        "fusion": fusion.state_dict(),
        "beta": fusion.beta.item(),
        "config": {
            "model_name": config.model_name,
            "top_k": config.top_k,
            "num_classes": num_classes,
            "initial_beta": config.initial_beta,
            "lambda_reg": config.lambda_reg,
            "evidence_mode": config.evidence_mode,
            "label_list": config.label_list,
        },
    }, save_path)

    logger.info(f"Fusion model saved to {save_path}")
    return save_path
