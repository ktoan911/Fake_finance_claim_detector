#!/usr/bin/env python3
# train_retrieval.py — with detailed logging
"""
Train the dense retrieval model using the custom Contrastive Learning pipeline
defined in src.embeddings.

Dataset: data/finfact_raw_truefalse.csv
  - claim   : the claim text (anchor)
  - evidence : Python-list string of evidence sentences (positives)

Goal: Given a claim, the retriever should rank its supporting evidence
      sentences higher than evidence from other claims.
"""

import argparse
import ast
import os
import re
import time

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

try:
    from src.embeddings import (
        SENTENCE_TRANSFORMERS_AVAILABLE,
        TORCH_AVAILABLE,
        ContrastiveEmbeddingModel,
        CryptoEmbeddingTrainer,
        RetrievalDataset,
    )
except ImportError as e:
    raise ImportError(f"Could not import embedding modules: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_evidence_list(raw: str) -> list[str]:
    """
    Parse the evidence column which may be stored as:
      1. A Python list literal: ['sentence 1', 'sentence 2', ...]
      2. A plain multi-sentence string separated by period / newline
    Returns a de-duplicated list of non-empty strings (min 10 chars).
    """
    raw = str(raw).strip()
    sentences = []

    # Try ast.literal_eval first (handles proper Python list literals)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            sentences = [str(s).strip() for s in parsed]
        else:
            sentences = [str(parsed).strip()]
    except (ValueError, SyntaxError):
        # Fallback: split on newlines or double-spaces
        parts = re.split(r"\n{1,}|\.\s{2,}", raw)
        sentences = [p.strip() for p in parts]

    # Filter short / empty strings
    sentences = [s for s in sentences if len(s) >= 10]
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for s in sentences:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def load_finfact_dataset(csv_path: str):
    """
    Load finfact_raw_truefalse.csv and return list of
    {'claim': str, 'evidences': List[str]} dicts.
    Only rows with at least one valid evidence sentence are kept.
    """
    df = pd.read_csv(csv_path)

    required = {"claim", "evidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. Found: {df.columns.tolist()}"
        )

    records = []
    skipped = 0
    total_rows = len(df)
    logger.info(f"Parsing {total_rows} rows from CSV...")

    for idx, (_, row) in enumerate(df.iterrows()):
        claim = str(row["claim"]).strip()
        if not claim or claim.lower() == "nan":
            skipped += 1
            continue

        if pd.isna(row["evidence"]):
            logger.debug(f"  Row {idx}: skipped (NaN evidence)")
            skipped += 1
            continue

        evs = parse_evidence_list(row["evidence"])
        if not evs:
            logger.debug(f"  Row {idx}: skipped (no valid evidence sentences)")
            skipped += 1
            continue

        records.append({"claim": claim, "evidences": evs})

        if (idx + 1) % 500 == 0:
            logger.info(
                f"  Parsed {idx + 1}/{total_rows} rows | kept={len(records)} skipped={skipped}"
            )

    all_evs = [ev for r in records for ev in r["evidences"]]
    ev_counts = [len(r["evidences"]) for r in records]
    logger.info(
        f"\n{'=' * 50}\n"
        f"  CSV loaded: {csv_path}\n"
        f"  Total rows      : {total_rows}\n"
        f"  Kept records    : {len(records)}\n"
        f"  Skipped         : {skipped}\n"
        f"  Total sentences : {len(all_evs)}\n"
        f"  Avg ev/claim    : {len(all_evs) / max(1, len(records)):.1f}\n"
        f"  Min ev/claim    : {min(ev_counts) if ev_counts else 0}\n"
        f"  Max ev/claim    : {max(ev_counts) if ev_counts else 0}\n"
        f"{'=' * 50}"
    )
    return records


def collate_triplets(batch):
    """
    Custom collate for RetrievalDataset.
    Each item: {'anchor': str, 'positive': str, 'negatives': List[str]}
    """
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negatives"] for item in batch]  # List[List[str]]
    return {"anchor": anchors, "positive": positives, "negatives": negatives}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not TORCH_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "PyTorch and SentenceTransformers are required for training."
        )

    parser = argparse.ArgumentParser(
        description="Fine-tune Retrieval Model on finfact_raw_truefalse.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/finfact_raw_truefalse.csv",
        help="Path to training CSV (claim/evidence columns)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Base SentenceTransformer model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenizer",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_triplets",
        type=int,
        default=8000,
        help="Number of (anchor, positive, negatives) triplets to sample per epoch",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="Device (cuda/cpu)")

    args = parser.parse_args()

    # 1. Load data
    t0 = time.time()
    logger.info(f"Loading dataset from {args.csv}")
    records = load_finfact_dataset(args.csv)
    logger.info(f"Data loading took {time.time() - t0:.1f}s")

    if len(records) < 2:
        logger.error("Need at least 2 records to build negatives. Aborting.")
        return

    logger.info(
        f"Device: {args.device} | epochs={args.epochs} | batch={args.batch_size} | triplets/epoch={args.num_triplets}"
    )

    # 2. Initialize model
    logger.info(f"Initializing ContrastiveEmbeddingModel ({args.model_name})...")
    model = ContrastiveEmbeddingModel(
        base_model_name=args.model_name,
        lambda_reg=0.001,
        freeze_base=True,
        max_length=args.max_length,
        encoder_device="cpu",
        encode_batch_size=args.batch_size,
    )
    model = model.to(args.device)

    # 3. Build dataset
    logger.info(
        f"Building RetrievalDataset with {args.num_triplets} triplets "
        f"and {args.num_negatives} negatives each..."
    )
    dataset = RetrievalDataset(
        records=records,
        num_triplets=args.num_triplets,
        num_negatives=args.num_negatives,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_triplets,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    # 4. Initialize trainer
    trainer = CryptoEmbeddingTrainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 5. Training loop
    logger.info(f"\n{'=' * 50}")
    logger.info(
        f"Starting training: {args.epochs} epochs | {args.num_triplets} triplets/epoch | device={args.device}"
    )
    logger.info(f"{'=' * 50}")
    train_start = time.time()
    history = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        logger.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        avg_loss = trainer.train_epoch(dataloader)
        epoch_time = time.time() - epoch_start

        if epoch == args.epochs - 1:
            logger.info("Running final evaluation...")
            metrics = trainer.evaluate(dataloader)
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Pos-Neg Gap: {metrics['pos_neg_gap']:.4f} | "
                f"Separation: {metrics['separation_rate']:.4f} | "
                f"Pos sim: {metrics['mean_pos_similarity']:.4f} | "
                f"Neg sim: {metrics['mean_neg_similarity']:.4f}"
            )
            history.append(
                {
                    **{"epoch": epoch + 1, "loss": avg_loss, "time": epoch_time},
                    **metrics,
                }
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s"
            )
            history.append({"epoch": epoch + 1, "loss": avg_loss, "time": epoch_time})

    total_time = time.time() - train_start
    logger.info(f"\n{'=' * 50}")
    logger.info(
        f"Training complete | Total time: {total_time:.1f}s ({total_time / 60:.1f} min)"
    )
    logger.info("Loss history: " + " -> ".join(f"{h['loss']:.4f}" for h in history))
    logger.info(f"{'=' * 50}")

    # 6. Save model
    os.makedirs(args.output_dir, exist_ok=True)

    if hasattr(model, "encoder") and model.encoder is not None:
        logger.info(f"Saving fine-tuned SentenceTransformer to {args.output_dir}")
        model.encoder.save(args.output_dir)

        torch.save(
            model.projection.state_dict(),
            os.path.join(args.output_dir, "custom_projection.pt"),
        )
        logger.info("Training completed successfully. Model saved.")
    else:
        logger.error("Model encoder not found. Cannot save.")


if __name__ == "__main__":
    main()
