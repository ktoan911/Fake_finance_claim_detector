#!/usr/bin/env python3
"""
Train Fusion MLP + beta only using CSV labeled data.
Required CSV columns: text/claim, evidence, label
"""

import argparse
import os
from loguru import logger

from src.csv_loader import CSVLabeledLoader
from src.fusion_trainer import train_fusion_from_dataframe, FusionTrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Fusion MLP + beta only using CSV labeled data.")
    parser.add_argument("--labeled_csv", type=str, required=True, help="Path to the labeled CSV file (text,evidence,label)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--model_path", type=str, default=os.getenv("LORA_MODEL_PATH") or os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B"), help="Path to the model (LoRA or base model)")
    parser.add_argument("--device", type=str, default="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") or os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu", help="Device to use (cuda/cpu)")
    parser.add_argument("--save_path", type=str, default=os.getenv("FUSION_OUTPUT_PATH", "artifacts/fusion_model.pt"), help="Path to save the fusion model")
    
    args = parser.parse_args()

    logger.info(f"Loading labeled data from {args.labeled_csv}...")
    labeled_df = CSVLabeledLoader(args.labeled_csv).load()
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

    fusion_config = FusionTrainingConfig(
        model_name=args.model_path,
        device=args.device
    )

    train_fusion_from_dataframe(
        knowledge_base=kb_docs,
        labeled_df=labeled_df,
        config=fusion_config,
        save_path=args.save_path
    )

    logger.info(f"Fusion training complete. Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
