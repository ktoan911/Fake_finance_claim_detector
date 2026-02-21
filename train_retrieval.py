#!/usr/bin/env python3
"""
Train the dense retrieval model using the custom Contrastive Learning pipeline
defined in src.embeddings. Extends the base SentenceTransformer (bge-small)
with a custom projection layer and Multiple Negatives + Hard Negatives Mining.
"""

import argparse
import os

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.csv_loader import CSVLabeledLoader

try:
    from src.embeddings import (
        SENTENCE_TRANSFORMERS_AVAILABLE,
        TORCH_AVAILABLE,
        ContrastiveEmbeddingModel,
        CryptoEmbeddingDataset,
        CryptoEmbeddingTrainer,
    )
except ImportError as e:
    raise ImportError(f"Could not import embedding modules: {e}")


def collate_triplets(batch):
    """
    Custom collate function for CryptoEmbeddingDataset.
    Expected batch is a list of dicts:
    {"anchor": str, "positive": str, "negatives": List[str]}
    """
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    # List of lists [B, num_negatives]
    negatives = [item["negatives"] for item in batch]

    return {"anchor": anchors, "positive": positives, "negatives": negatives}


def main():
    if not TORCH_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "PyTorch and SentenceTransformers are required for training."
        )

    parser = argparse.ArgumentParser(
        description="Fine-tune Retrieval Model using Custom Pipeline"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument(
        "--model_name", type=str, default="BAAI/bge-small-en-v1.5", help="Base model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_triplets", type=int, default=5000, help="Number of triplets to generate"
    )
    parser.add_argument(
        "--num_negatives", type=int, default=3, help="Negatives per anchor"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/retriever_model",
        help="Directory to save the trained model",
    )

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device", type=str, default=device, help="Device to train on (cuda/cpu)"
    )

    args = parser.parse_args()

    logger.info(f"Loading data from {args.csv}")
    df = CSVLabeledLoader(args.csv).load()

    scam_samples = []
    legit_samples = []

    # Split dataset based on label
    # In Fake Crypto Claim Detector: False (scam/refuted) = 1, True (legitimate/supported) = 0
    for _, row in df.iterrows():
        claim = str(row["text"]).strip()
        evidence_full = str(row["evidence"]).strip()
        label = row["label"]

        if not claim or not evidence_full or pd.isna(row["evidence"]):
            continue

        # Support multiple evidence segments split by "|||"
        # Create separate sample entries for each evidence snippet
        evidences = [
            e.strip() for e in evidence_full.split("|||") if len(e.strip()) > 10
        ]

        for ev in evidences:
            sample = {
                "text": claim,
                "evidence": ev,
                "scam_type": "default",  # Can be expanded if CSV contains more metadata
            }
            if label == 1:
                scam_samples.append(sample)
            elif label == 0:
                legit_samples.append(sample)

    logger.info(
        f"Loaded {len(scam_samples)} scam entries and {len(legit_samples)} legit entries."
    )

    if len(scam_samples) == 0 or len(legit_samples) == 0:
        logger.error("Need both scam and legitimate samples to train contrastively.")
        return

    # 1. Initialize custom Model
    logger.info(
        f"Initializing ContrastiveEmbeddingModel tightly wrapping {args.model_name}..."
    )
    # NOTE: freeze_base=False allows the base transformer to be fully finetuned during the contrastive learning process
    # instead of just training the linear projection head, leading to better representations.
    model = ContrastiveEmbeddingModel(
        base_model_name=args.model_name, lambda_reg=0.001, freeze_base=False
    )
    model = model.to(args.device)

    # 2. Initialize Dataset
    logger.info(
        f"Building Dataset with {args.num_triplets} triplets and Hard Negative Mining..."
    )
    dataset = CryptoEmbeddingDataset(
        scam_samples=scam_samples,
        legitimate_samples=legit_samples,
        num_triplets=args.num_triplets,
        num_negatives=args.num_negatives,
        hard_negative_ratio=0.5,
        embedding_model=model,  # Use Model for hard negative mining
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_triplets
    )

    # 3. Initialize Trainer
    trainer = CryptoEmbeddingTrainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 4. Train Loop
    logger.info(f"Starting training for {args.epochs} epochs on {args.device}...")
    for epoch in range(args.epochs):
        avg_loss = trainer.train_epoch(dataloader)

        # Only evaluate every few epochs or on the last one to save time
        if epoch == args.epochs - 1:
            metrics = trainer.evaluate(dataloader)
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                + f"Pos-Neg Gap: {metrics['pos_neg_gap']:.4f} | Separation: {metrics['separation_rate']:.4f}"
            )
        else:
            logger.info(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f}")

    # 5. Save the Fine-Tuned Base Model (so sentence-transformers can natively load it later without projection artifacts!)
    os.makedirs(args.output_dir, exist_ok=True)

    if hasattr(model, "encoder") and model.encoder is not None:
        logger.info(f"Saving fine-tuned base SentenceTransformer to {args.output_dir}")
        model.encoder.save(args.output_dir)

        # Also save the custom projection state_dict just in case we need it for advanced inference later
        torch.save(
            model.projection.state_dict(),
            os.path.join(args.output_dir, "custom_projection.pt"),
        )
        logger.info("Training completed successfully. Model saved.")
    else:
        logger.error("Model encoder not found. Cannot save.")


if __name__ == "__main__":
    main()
