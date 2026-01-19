#!/usr/bin/env python3
"""
Train Fusion MLP + beta only using CSV labeled data.
Required CSV columns: text/claim, evidence, label
"""

import os
from loguru import logger

from src.csv_loader import CSVLabeledLoader
from src.fusion_trainer import train_fusion_from_dataframe, FusionTrainingConfig


def main():
    labeled_csv = os.getenv("LABELED_CSV_PATH")
    if not labeled_csv:
        raise ValueError("Provide LABELED_CSV_PATH for training. Format: text,evidence,label")

    logger.info(f"Loading labeled data from {labeled_csv}...")
    labeled_df = CSVLabeledLoader(labeled_csv).load()
    logger.info(f"Labeled data: {len(labeled_df)} samples")

    evidences = labeled_df["evidence"].tolist()
    timestamps = labeled_df["timestamp"].tolist() if "timestamp" in labeled_df.columns else [None] * len(evidences)

    kb_docs = []
    for evidence, ts in zip(evidences, timestamps):
        kb_docs.append({
            "text": str(evidence),
            "timestamp": ts,
            "source": "csv"
        })
    logger.info(f"Knowledge base built from CSV evidence: {len(kb_docs)} documents")

    model_path = os.getenv("LORA_MODEL_PATH") or os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B")
    fusion_config = FusionTrainingConfig(
        model_name=model_path,
        device=os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
    )

    save_path = os.getenv("FUSION_OUTPUT_PATH", "artifacts/fusion_model.pt")
    train_fusion_from_dataframe(
        knowledge_base=kb_docs,
        labeled_df=labeled_df,
        config=fusion_config,
        save_path=save_path
    )

    logger.info(f"Fusion training complete. Model saved to: {save_path}")


if __name__ == "__main__":
    main()
