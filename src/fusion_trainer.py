from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import LABEL_LIST, PROMPT_TEMPLATE
from .fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
from .llm_scorer import LLMScorer
from .retrieval import KnowledgeAugmentedRetriever


@dataclass
class FusionTrainingConfig:
    model_name: str = "artifacts/lora_llm"
    device: str = "auto"
    batch_size: int = 4
    llm_batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 1e-4
    top_k: int = 10
    alpha: float = 0.7
    lambda_decay: float = 0.1
    gamma: float = 0.5
    initial_beta: float = 0.5
    lambda_reg: float = 0.01  # Only used in fusion layer, not doubled
    max_length: int = 2048
    evidence_mode: str = (
        "retrieved"  # "gold" or "retrieved" - per paper, should be "retrieved"
    )
    label_list: List[str] = field(default_factory=lambda: LABEL_LIST)
    retriever_model: str = "BAAI/bge-small-en-v1.5"


def _build_retrieval_features(
    retriever: KnowledgeAugmentedRetriever, text: str, top_k: int, rrf_top_k: int = 20
) -> tuple:
    """Returns (features, retrieved_evidence_text)."""
    # RRF hybrid: top 20 candidates, then Temporal scoring to get top_k
    results = retriever.retrieve(text, top_k=top_k, rrf_top_k=rrf_top_k)
    features = []
    evidence_texts = []

    for r in results:
        features.append([r.score, r.rrf_score, r.recency_score, r.cyclicity_score])
        evidence_texts.append(r.text)

    if len(features) < top_k:
        features.extend([[0.0, 0.0, 0.0, 0.0]] * (top_k - len(features)))

    # Return list of evidence texts (for smart truncation in LLMScorer)
    return np.array(features, dtype=np.float32), evidence_texts


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

    # Initialize retriever with RRF hybrid
    retriever = KnowledgeAugmentedRetriever(
        embedding_model=config.retriever_model,
        alpha=config.alpha,
        lambda_decay=config.lambda_decay,
        gamma=config.gamma,
        use_query_expansion=True,
        rrf_k=60,  # RRF constant
    )
    retriever.index_documents(
        knowledge_base, text_field="text", timestamp_field="timestamp"
    )
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
        num_retrieved=config.top_k, score_features=4, hidden_dim=64, output_dim=64
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
    # CSVLabeledLoader already normalized labels (true→0, false→1)
    # Do NOT call _normalize_label again or labels will be flipped!
    labels = labeled_df["label"].tolist()

    logger.info(f"Training samples: {len(texts)}")
    logger.info(
        f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}"
    )

    # --- PRE-COMPUTATION PHASE ---
    logger.info("Starting pre-computation of retrieval features and LLM logits...")
    all_retrieval_features = []
    all_llm_logits = []

    # Ensure LLM is in eval mode and cache is disabled
    llm.model.eval()
    llm.model.config.use_cache = False
    for p in llm.model.parameters():
        p.requires_grad_(False)

    # Use micro-batch size for LLM to save memory (fusion batch size can be larger)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Process in batches for pre-computation
    for i in range(0, len(texts), config.batch_size):
        batch_texts = texts[i : i + config.batch_size]
        batch_gold_evidences = gold_evidences[i : i + config.batch_size]

        # 1. Retrieval
        batch_feats = []
        batch_retrieved_evidences = []
        for t in batch_texts:
            feats, retrieved_evidence = _build_retrieval_features(
                retriever, t, config.top_k
            )
            batch_feats.append(feats)
            batch_retrieved_evidences.append(retrieved_evidence)

        all_retrieval_features.extend(batch_feats)

        # 2. LLM Scoring
        if config.evidence_mode == "retrieved":
            batch_evidences = batch_retrieved_evidences
        else:
            batch_evidences = batch_gold_evidences

        # Micro-batching for LLM inference
        batch_logits_list = []
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                for j in range(0, len(batch_texts), config.llm_batch_size):
                    sub_texts = batch_texts[j : j + config.llm_batch_size]
                    sub_evidences = batch_evidences[j : j + config.llm_batch_size]

                    logits = llm.score_logits(sub_texts, sub_evidences)
                    batch_logits_list.append(logits)

        # Concatenate micro-batches
        lm_logits = torch.cat(batch_logits_list, dim=0)
        all_llm_logits.append(lm_logits.cpu())  # Store on CPU to save GPU memory

        if (i // config.batch_size) % 10 == 0:
            logger.info(f"Pre-computed {i + len(batch_texts)}/{len(texts)} samples")

    # Convert to tensors
    tensor_retrieval = torch.tensor(
        np.array(all_retrieval_features), dtype=torch.float32
    )
    tensor_llm_logits = torch.cat(all_llm_logits, dim=0)
    tensor_labels = torch.tensor(labels, dtype=torch.long)

    logger.info("Pre-computation complete. Unloading LLM...")

    # Unload LLM to free memory
    del llm
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    fusion.train()
    retrieval_encoder.train()

    dataset_size = len(texts)
    indices = torch.randperm(dataset_size)

    for epoch in range(config.epochs):
        # Shuffle indices at the start of each epoch
        indices = torch.randperm(dataset_size)

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for i in range(0, dataset_size, config.batch_size):
            batch_indices = indices[i : i + config.batch_size]

            # Move batch to device
            b_retrieval = tensor_retrieval[batch_indices].to(config.device)
            b_llm_logits = tensor_llm_logits[batch_indices].to(config.device)
            b_labels = tensor_labels[batch_indices].to(config.device)

            # Encode retrieval features
            retrieval_features = retrieval_encoder(b_retrieval)

            # Fusion: β·pLM + (1-β)·MLP(pret) per Eq.2
            output = fusion(b_llm_logits, retrieval_features)
            fused_logits = output.fused_logits

            # Loss computation
            if num_classes == 2:
                # Binary classification: use BCEWithLogitsLoss
                # fused_logits: [B] (single logit per sample)
                # Convert label IDs to BCE targets:
                # - Label ID 0 (True) → BCE target 1.0 (positive class)
                # - Label ID 1 (False) → BCE target 0.0 (negative class)
                target_binary = (
                    b_labels == 0
                ).float()  # Convert: 0→1.0 (True), 1→0.0 (False)
                ce_loss = F.binary_cross_entropy_with_logits(
                    fused_logits, target_binary
                )
            else:
                # Multi-class: use CrossEntropyLoss
                ce_loss = F.cross_entropy(fused_logits, b_labels)

            # Add beta regularization (L = CE + λ||β||²)
            beta_reg = fusion.lambda_reg * (fusion.beta**2)
            loss = ce_loss + beta_reg

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()

            # Get predictions using argmax (works for both binary and multi-class)
            preds = torch.argmax(output.final_probs, dim=-1)

            correct += (preds == b_labels).sum().item()
            total += len(b_labels)
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = correct / total if total > 0 else 0
        beta_val = fusion.beta.item()
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} - loss: {avg_loss:.4f} - acc: {accuracy:.4f} - β: {beta_val:.4f}"
        )

    # Save model
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "retrieval_encoder": retrieval_encoder.state_dict(),
            "fusion": fusion.state_dict(),
            "beta": fusion.beta.item(),
            "config": {
                "model_name": config.model_name,
                "retriever_model": config.retriever_model,
                "top_k": config.top_k,
                "num_classes": num_classes,
                "initial_beta": config.initial_beta,
                "lambda_reg": config.lambda_reg,
                "evidence_mode": config.evidence_mode,
                "label_list": config.label_list,
            },
        },
        save_path,
    )

    logger.info(f"Fusion model saved to {save_path}")
    return save_path
