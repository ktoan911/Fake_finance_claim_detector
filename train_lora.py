#!/usr/bin/env python3
"""
Train LoRA only (supervised classification) using CSV labeled data.
Required CSV columns: text/claim, evidence, label
"""

import os
import argparse
from loguru import logger
from dotenv import load_dotenv

from src.csv_loader import CSVLabeledLoader
from src.lora_trainer import train_lora_classification, LoRATrainingConfig

# Load environment variables from .env file
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for crypto claim classification")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to TRAIN CSV file")
    parser.add_argument("--dev-csv", type=str, default=None, help="Path to DEV/EVAL CSV file")
    parser.add_argument("--csv", type=str, default=None, help="(Deprecated) Alias for --train-csv")
    parser.add_argument("--model", type=str, default=None, help="Model name from HuggingFace")
    parser.add_argument("--output", type=str, default=None, help="Output directory for LoRA model")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "bf16", "fp16", "fp32"],
        default=None,
        help="Training precision (default: auto)",
    )
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--early-stopping", type=int, default=None, help="Early stopping patience (default: 3)")
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to LoRA checkpoint to resume training (e.g., artifacts/lora_llm/checkpoint-190)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Priority: args > env > legacy env fallback
    train_csv = (
        args.train_csv
        or args.csv
        or os.getenv("TRAIN_CSV_PATH")
        or os.getenv("LABELED_CSV_PATH")
    )
    dev_csv = args.dev_csv or os.getenv("DEV_CSV_PATH")
    if not train_csv or not dev_csv:
        raise ValueError(
            "Provide --train-csv and --dev-csv (or TRAIN_CSV_PATH and DEV_CSV_PATH). "
            "Format: text,evidence,label"
        )

    logger.info(f"Loading TRAIN data from {train_csv}...")
    train_df = CSVLabeledLoader(train_csv).load()
    logger.info(f"TRAIN samples: {len(train_df)}")

    logger.info(f"Loading DEV data from {dev_csv}...")
    dev_df = CSVLabeledLoader(dev_csv).load()
    logger.info(f"DEV samples: {len(dev_df)}")

    claims = train_df["text"].tolist()
    labels = train_df["label"].tolist()
    evidences = train_df["evidence"].tolist()
    eval_claims = dev_df["text"].tolist()
    eval_labels = dev_df["label"].tolist()
    eval_evidences = dev_df["evidence"].tolist()

    output_dir = args.output or os.getenv("LORA_OUTPUT_DIR", "artifacts/lora_llm")
    lora_config = LoRATrainingConfig(
        model_name=args.model or os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B"),
        output_dir=output_dir,
        epochs=args.epochs or int(os.getenv("LORA_EPOCHS", "3")),
        batch_size=args.batch_size or int(os.getenv("LORA_BATCH_SIZE", "1")),
        learning_rate=args.lr or float(os.getenv("LORA_LR", "2e-4")),
        precision=args.precision or os.getenv("LORA_PRECISION", "auto"),
        max_length=args.max_length or int(os.getenv("LORA_MAX_LENGTH", "256")),
        early_stopping_patience=args.early_stopping or int(os.getenv("LORA_EARLY_STOPPING", "3")),
    )

    lora_path = train_lora_classification(
        claims=claims, 
        evidences=evidences,
        labels=labels,
        eval_claims=eval_claims,
        eval_evidences=eval_evidences,
        eval_labels=eval_labels,
        config=lora_config,
        gradient_accumulation_steps=args.grad_accum or int(os.getenv("LORA_GRAD_ACCUM", "4")),
        checkpoint_path=args.load_model  # Load checkpoint to resume training
    )

    logger.info(f"LoRA training complete. Model saved to: {lora_path}")


if __name__ == "__main__":
    main()
