"""
Fusion Layer Trainer

Trains the retrieval MLP and learns beta (gating parameter) as in paper Section 3.2.
Uses labeled CSV data for supervision and evidence-based KB for retrieval features (d_i).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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


@dataclass
class FusionTrainingConfig:
    model_name: str
    device: str = "cpu"
    batch_size: int = 8
    epochs: int = 2
    learning_rate: float = 1e-4
    top_k: int = 5
    alpha: float = 0.7
    lambda_decay: float = 0.1
    gamma: float = 0.5
    initial_beta: float = 0.5
    lambda_reg: float = 0.01
    label_list: List[str] = None


def _build_retrieval_features(
    retriever: KnowledgeAugmentedRetriever,
    text: str,
    top_k: int,
) -> np.ndarray:
    results = retriever.retrieve(text, top_k=top_k)
    features = []
    for r in results:
        features.append([r.score, r.bm25_score, r.recency_score, r.cyclicity_score])
    if len(features) < top_k:
        features.extend([[0.0, 0.0, 0.0, 0.0]] * (top_k - len(features)))
    return np.array(features, dtype=np.float32)


def train_fusion_from_dataframe(
    knowledge_base: List[Dict],
    labeled_df,
    config: Optional[FusionTrainingConfig] = None,
    save_path: str = "artifacts/fusion_model.pt",
) -> str:
    """
    Train fusion MLP + beta from a labeled pandas DataFrame.
    Required columns: text, label. Optional: timestamp.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for fusion training.")

    config = config or FusionTrainingConfig(model_name="meta-llama/Llama-3.1-8B")
    if config.label_list is None:
        config.label_list = ["SUPPORTED", "REFUTED", "NEI"]

    if labeled_df is None or labeled_df.empty:
        raise ValueError("Labeled DataFrame is empty.")

    retriever = KnowledgeAugmentedRetriever(
        alpha=config.alpha,
        lambda_decay=config.lambda_decay,
        gamma=config.gamma,
        use_query_expansion=True
    )
    retriever.index_documents(knowledge_base, text_field="text", timestamp_field="timestamp")

    llm = LLMScorer(
        model_name=config.model_name,
        device=config.device,
        labels=config.label_list,
        prompt_template=(
            "You are verifying a crypto/finance claim.\n"
            "Claim: {text}\n"
            "Answer with one label: SUPPORTED, REFUTED, or NEI."
        ),
    )

    retrieval_encoder = RetrievalFeatureEncoder(
        num_retrieved=config.top_k,
        score_features=4,
        hidden_dim=64,
        output_dim=64
    ).to(config.device)

    num_classes = len(config.label_list)
    fusion = ConfidenceAwareFusion(
        retrieval_input_dim=64,
        hidden_dim=128,
        num_classes=num_classes,
        initial_beta=config.initial_beta,
        lambda_reg=config.lambda_reg
    ).to(config.device)

    params = list(retrieval_encoder.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    texts = labeled_df["text"].tolist()
    labels = labeled_df["label"].astype(int).tolist()

    fusion.train()
    retrieval_encoder.train()

    for epoch in range(config.epochs):
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(texts), config.batch_size):
            batch_texts = texts[i:i + config.batch_size]
            batch_labels = labels[i:i + config.batch_size]

            lm_logits = llm.score_texts(batch_texts)

            batch_features = []
            for t in batch_texts:
                feats = _build_retrieval_features(retriever, t, config.top_k)
                batch_features.append(feats)
            retrieval_scores = torch.tensor(batch_features, dtype=torch.float32, device=config.device)

            retrieval_features = retrieval_encoder(retrieval_scores)
            output = fusion(lm_logits, retrieval_features)
            fused_logits = output.fused_logits

            targets = torch.tensor(batch_labels, dtype=torch.long, device=config.device)
            loss = F.cross_entropy(fused_logits, targets) + config.lambda_reg * (fusion.beta ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f} - beta: {fusion.beta.item():.4f}")

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "retrieval_encoder": retrieval_encoder.state_dict(),
        "fusion": fusion.state_dict(),
        "beta": fusion.beta.item(),
        "config": config.__dict__,
    }, save_path)

    logger.info(f"Fusion model saved to {save_path}")
    return save_path

