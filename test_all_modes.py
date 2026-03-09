#!/usr/bin/env python3
"""
Comprehensive Test Script: LoRA vs Fusion (Gold vs Retrieval)
Evaluates 4 modes:
1. LoRA + Retrieval Evidence
2. LoRA + Gold Evidence
3. Fusion + Retrieval Evidence
4. Fusion + Gold Evidence
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.config import LABEL_LIST, PROMPT_TEMPLATE
from src.csv_loader import CSVLabeledLoader
from src.fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
from src.fusion_trainer import _build_retrieval_features  # Re-use helper
from src.llm_scorer import LLMScorer
from src.retrieval import KnowledgeAugmentedRetriever
from src.utils import normalize_text

# Label mapping for metrics
LABEL_MAP = {idx: label for idx, label in enumerate(LABEL_LIST)}


def calculate_metrics(y_true, y_pred, mode_name):
    """Calculate and print metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_binary = f1_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )  # negative/refuted class as positive

    logger.info(f"--- Results for {mode_name} ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Macro:  {f1_macro:.4f}")
    logger.info(f"F1 Binary (negative class): {f1_binary:.4f}")

    return {
        "Mode": mode_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Macro": f1_macro,
        "F1_Binary": f1_binary,
    }


def load_fusion_model(model_path, device, num_classes=2):
    """Load trained fusion model components."""
    checkpoint = torch.load(model_path, map_location=device)

    # Load config from checkpoint if available, else standard
    saved_config = checkpoint.get("config", {})

    retrieval_encoder = RetrievalFeatureEncoder(
        num_retrieved=saved_config.get("top_k", 10),
        score_features=4,
        hidden_dim=64,
        output_dim=64,
    ).to(device)

    fusion = ConfidenceAwareFusion(
        retrieval_input_dim=64,
        hidden_dim=128,
        num_classes=num_classes,
        initial_beta=saved_config.get("initial_beta", 0.5),
    ).to(device)

    retrieval_encoder.load_state_dict(checkpoint["retrieval_encoder"])
    fusion.load_state_dict(checkpoint["fusion"])
    fusion.beta.data = torch.tensor(checkpoint["beta"]).to(device)

    retrieval_encoder.eval()
    fusion.eval()

    return retrieval_encoder, fusion, saved_config


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA and Fusion models in all modes"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument(
        "--lora_model", type=str, required=True, help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--fusion_model",
        type=str,
        required=True,
        help="Path to Fusion checkpoint (.pt)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples for testing"
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=None,
        help="Override trained dense retrieval model path. If not provided, will use the one used during fusion training.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--llm_batch_size", type=int, default=1, help="LLM inference batch size"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    logger.info(f"Testing on device: {args.device}")

    # 1. Load Data
    logger.info(f"Loading data from {args.csv}...")
    df = CSVLabeledLoader(args.csv).load()
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to {args.limit} samples")

    texts = df["text"].tolist()
    gold_evidences = df["evidence"].tolist()
    labels = df["label"].tolist()  # 0=positive label, 1=negative label

    # Build Knowledge Base for Retrieval
    # Deduplicate documents from evidence column using normalization (match train_fusion.py)

    unique_docs = {}
    for evidence in gold_evidences:
        if pd.isna(evidence):
            continue
        parts = str(evidence).split("|||")
        for part in parts:
            part = part.strip()
            if len(part) > 10:
                # Normalize for deduplication key, but store original text
                norm_key = normalize_text(part)

                if norm_key not in unique_docs:
                    unique_docs[norm_key] = {
                        "text": part,
                        "timestamp": None,
                    }

    kb_docs = list(unique_docs.values())
    logger.info(
        f"Built temporary KB with {len(kb_docs)} unique documents from test set evidence (deduplicated)"
    )

    # Fusion Model
    retrieval_encoder, fusion_layer, fusion_config = load_fusion_model(
        args.fusion_model, args.device
    )
    top_k = fusion_config.get("top_k", 10)
    retriever_model = args.retriever_model or fusion_config.get(
        "retriever_model", "BAAI/bge-small-en-v1.5"
    )

    # 2. Initialize Components
    # Retriever
    logger.info(f"Using retriever model: {retriever_model}")
    retriever = KnowledgeAugmentedRetriever(embedding_model=retriever_model, rrf_k=60)
    retriever.index_documents(kb_docs, text_field="text", timestamp_field="timestamp")

    # LLM Scorer
    llm = LLMScorer(
        model_name=args.lora_model,
        device=args.device,
        max_length=2048,
        labels=LABEL_LIST,
        prompt_template=PROMPT_TEMPLATE,
    )

    # 3. Running Inference
    results_summary = []

    # Store predictions
    preds_lora_retrieval = []
    preds_lora_gold = []
    preds_fusion_retrieval = []
    preds_fusion_gold = []

    # We need to process in batches
    num_samples = len(texts)

    # Pre-compute retrieval features for all samples (needed for Fusion)
    # Note: For Fusion + Gold, we still use retrieval features from retriever
    # but feed Gold Evidence to LLM.

    logger.info("Step 1/3: Running Retrieval...")
    all_retrieval_features = []
    all_retrieved_evidences = []

    for text in tqdm(texts, desc="Retrieving"):
        feats, retrieved_evidence_list = _build_retrieval_features(
            retriever, text, top_k
        )
        all_retrieval_features.append(feats)
        all_retrieved_evidences.append(retrieved_evidence_list)

    tensor_retrieval_features = torch.tensor(
        np.array(all_retrieval_features), dtype=torch.float32
    ).to(args.device)

    logger.info("Step 2/3: Running LLM Inference (Retrieval & Gold)...")

    # Only need logits for Fusion, but we can get probs/preds from logits too
    logits_retrieval = []
    logits_gold = []

    # Batch processing for LLM
    for i in tqdm(range(0, num_samples, args.batch_size), desc="LLM Scoring"):
        batch_texts = texts[i : i + args.batch_size]
        batch_gold_evidences = gold_evidences[i : i + args.batch_size]
        batch_retrieved_evidences = all_retrieved_evidences[i : i + args.batch_size]

        # A. Mode: Retrieval Evidence (Micro-batching)
        sub_logits_ret = []
        for j in range(0, len(batch_texts), args.llm_batch_size):
            sub_texts = batch_texts[j : j + args.llm_batch_size]
            sub_evs = batch_retrieved_evidences[j : j + args.llm_batch_size]
            sub_logits_ret.append(llm.score_logits(sub_texts, sub_evs))

        if sub_logits_ret:
            logits_retrieval.append(torch.cat(sub_logits_ret, dim=0))

        # B. Mode: Gold Evidence (Micro-batching)
        sub_logits_gold = []
        for j in range(0, len(batch_texts), args.llm_batch_size):
            sub_texts = batch_texts[j : j + args.llm_batch_size]
            sub_evs = batch_gold_evidences[j : j + args.llm_batch_size]
            sub_logits_gold.append(llm.score_logits(sub_texts, sub_evs))

        if sub_logits_gold:
            logits_gold.append(torch.cat(sub_logits_gold, dim=0))

    tensor_logits_retrieval = torch.cat(logits_retrieval, dim=0).to(args.device)
    tensor_logits_gold = torch.cat(logits_gold, dim=0).to(args.device)

    logger.info("Step 3/3: Computing Metrics...")

    # --- Mode 1: LoRA + Retrieval ---
    # Preds = argmax(logits)
    preds_lora_retrieval = torch.argmax(tensor_logits_retrieval, dim=1).cpu().numpy()
    results_summary.append(
        calculate_metrics(labels, preds_lora_retrieval, "LoRA + Retrieval")
    )

    # --- Mode 2: LoRA + Gold ---
    preds_lora_gold = torch.argmax(tensor_logits_gold, dim=1).cpu().numpy()
    results_summary.append(calculate_metrics(labels, preds_lora_gold, "LoRA + Gold"))

    # --- Mode 3: Fusion + Retrieval ---
    # Fusion(logits_retrieval, features_retrieval)
    with torch.no_grad():
        encoded_feats = retrieval_encoder(tensor_retrieval_features)
        fusion_out_ret = fusion_layer(tensor_logits_retrieval, encoded_feats)
        # Final probs or logits? Fusion returns FusionOutput with final_probs
        # Use simple argmax on final_probs
        preds_fusion_retrieval = (
            torch.argmax(fusion_out_ret.final_probs, dim=1).cpu().numpy()
        )

    results_summary.append(
        calculate_metrics(labels, preds_fusion_retrieval, "Fusion + Retrieval")
    )

    # --- Mode 4: Fusion + Gold ---
    # Fusion(logits_gold, features_retrieval)
    # Rationale: LLM sees Gold evidence (perfect context), but Fusion layer still sees
    # "how hard was it to retrieve info" features. This tests if Fusion improves
    # even with perfect LLM context (unlikely, but requested).
    with torch.no_grad():
        # Reuse encoded_feats from retrieval
        fusion_out_gold = fusion_layer(tensor_logits_gold, encoded_feats)
        preds_fusion_gold = (
            torch.argmax(fusion_out_gold.final_probs, dim=1).cpu().numpy()
        )

    results_summary.append(
        calculate_metrics(labels, preds_fusion_gold, "Fusion + Gold")
    )

    # Summary Table
    print("\n" + "=" * 60)
    print(f"{'Mode':<20} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 60)
    for res in results_summary:
        print(
            f"{res['Mode']:<20} | {res['Accuracy']:.4f}   | {res['Precision']:.4f}   | {res['Recall']:.4f}   | {res['F1_Macro']:.4f}"
        )
    print("=" * 60 + "\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()
