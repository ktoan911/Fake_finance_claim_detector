#!/usr/bin/env python3
"""
Train Fusion MLP + beta only using CSV labeled data.
Required CSV columns: text/claim, evidence, label
"""

import argparse
import os

from loguru import logger

from src.csv_loader import CSVLabeledLoader
from src.fusion_trainer import FusionTrainingConfig, train_fusion_from_dataframe
from src.utils import normalize_text


def main():
    parser = argparse.ArgumentParser(
        description="Train Fusion MLP + beta only using CSV labeled data."
    )
    parser.add_argument(
        "--labeled_csv",
        type=str,
        required=True,
        help="Path to the labeled CSV file (text,evidence,label)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--llm_batch_size", type=int, default=8, help="Batch size for LLM"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("LORA_MODEL_PATH", "artifacts/lora_llm"),
        help="Path to the LoRA-trained model (default: artifacts/lora_llm)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
        if os.getenv("CUDA_VISIBLE_DEVICES")
        or os.system("nvidia-smi > /dev/null 2>&1") == 0
        else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.getenv("FUSION_OUTPUT_PATH", "artifacts/fusion_model.pt"),
        help="Path to save the fusion model",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=os.getenv("RETRIEVER_MODEL_PATH", "AITeamVN/Vietnamese_Embedding"),
        help="Path to trained dense retrieval model (default: artifacts/retriever_model)",
    )

    args = parser.parse_args()

    logger.info(f"Loading labeled data from {args.labeled_csv}...")
    labeled_df = CSVLabeledLoader(args.labeled_csv).load()
    logger.info(f"Labeled data: {len(labeled_df)} samples")

    # Extract evidence and timestamps from dataframe
    evidences = labeled_df["evidence"].tolist()
    timestamps = (
        labeled_df["timestamp"].tolist()
        if "timestamp" in labeled_df.columns
        else [None] * len(evidences)
    )

    # Use dict to deduplicate by normalized text, keeping original text
    unique_docs = {}

    for evidence, ts in zip(evidences, timestamps):
        # Split evidence into individual articles
        # Evidence articles are separated by |||
        evidence_str = str(evidence)
        articles = evidence_str.split("|||")

        for article in articles:
            article = article.strip()
            if len(article) > 10:  # Filter out empty or very short strings
                # Normalize for deduplication key, but store original text
                norm_key = normalize_text(article)

                if norm_key not in unique_docs:
                    unique_docs[norm_key] = {
                        "text": article,  # Keep original text
                        "timestamp": ts,
                        "source": "csv",
                    }
                else:
                    # If duplicate, keep the document with non-None timestamp
                    if ts is not None and unique_docs[norm_key]["timestamp"] is None:
                        unique_docs[norm_key]["timestamp"] = ts

    kb_docs = list(unique_docs.values())
    logger.info(
        f"Knowledge base built: {len(kb_docs)} unique documents (deduplicated from {len(evidences)} evidence entries)"
    )

    fusion_config = FusionTrainingConfig(
        model_name=args.model_path,
        retriever_model=args.retriever_model,
        device=args.device,
        batch_size=args.batch_size,
        llm_batch_size=args.llm_batch_size,
        epochs=args.epochs,
    )

    train_fusion_from_dataframe(
        knowledge_base=kb_docs,
        labeled_df=labeled_df,
        config=fusion_config,
        save_path=args.save_path,
    )

    logger.info(f"Fusion training complete. Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
