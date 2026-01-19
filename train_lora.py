#!/usr/bin/env python3
"""
Train LoRA only (supervised classification) using CSV labeled data.
Required CSV columns: text/claim, evidence, label
"""

import os
import argparse
from loguru import logger

from src.csv_loader import CSVLabeledLoader
from src.lora_trainer import train_lora_classification, LoRATrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for crypto claim classification")
    parser.add_argument("--csv", type=str, default=None, help="Path to labeled CSV file")
    parser.add_argument("--model", type=str, default=None, help="Model name from HuggingFace")
    parser.add_argument("--output", type=str, default=None, help="Output directory for LoRA model")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Priority: args > env > default
    labeled_csv = args.csv or os.getenv("LABELED_CSV_PATH")
    if not labeled_csv:
        raise ValueError("Provide --csv or LABELED_CSV_PATH for training. Format: text,evidence,label")

    logger.info(f"Loading labeled data from {labeled_csv}...")
    labeled_df = CSVLabeledLoader(labeled_csv).load()
    logger.info(f"Labeled data: {len(labeled_df)} samples")

    claims = labeled_df["text"].tolist()
    labels = labeled_df["label"].tolist()
    evidences = labeled_df["evidence"].tolist()

    output_dir = args.output or os.getenv("LORA_OUTPUT_DIR", "artifacts/lora_llm")
    lora_config = LoRATrainingConfig(
        model_name=args.model or os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B"),
        output_dir=output_dir,
        epochs=args.epochs or int(os.getenv("LORA_EPOCHS", "3")),
        batch_size=args.batch_size or int(os.getenv("LORA_BATCH_SIZE", "1")),
        learning_rate=args.lr or float(os.getenv("LORA_LR", "2e-4")),
        max_length=args.max_length or int(os.getenv("LORA_MAX_LENGTH", "256")),
    )

    lora_path = train_lora_classification(
        claims=claims,
        evidences=evidences,
        labels=labels,
        config=lora_config,
        gradient_accumulation_steps=args.grad_accum or int(os.getenv("LORA_GRAD_ACCUM", "4"))
    )

    logger.info(f"LoRA training complete. Model saved to: {lora_path}")


if __name__ == "__main__":
    main()
