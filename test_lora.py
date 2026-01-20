#!/usr/bin/env python3
"""
Test LoRA model (F1, Precision, Recall, Accuracy) on a labeled CSV dataset.
"""

import argparse
import os
import torch
import gc
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
from peft import PeftModel, PeftConfig
from src.csv_loader import CSVLabeledLoader
from src.lora_trainer import _prepare_classification_dataset, compute_metrics
from src.config import PROMPT_TEMPLATE

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model F1 score")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved LoRA adapter (e.g., artifacts/lora_llm)")
    parser.add_argument("--csv", type=str, required=True, help="Path to labeled CSV file")
    parser.add_argument("--base-model", type=str, default="instruction-pretrain/finance-Llama3-8B", help="Base model name")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    # Load Data
    logger.info(f"Loading data from {args.csv}...")
    loader = CSVLabeledLoader(args.csv)
    df = loader.load()
    claims = df["text"].tolist()
    evidences = df["evidence"].tolist()
    labels = df["label"].tolist()
    logger.info(f"Loaded {len(claims)} samples.")

    # Load Tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from checkpoint ({e}), loading from base model...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    logger.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    logger.info(f"Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    # Prepare Dataset
    logger.info("Preparing dataset...")
    dataset = _prepare_classification_dataset(
        claims, evidences, labels, tokenizer, args.max_length, PROMPT_TEMPLATE
    )

    # Trainer for Evaluation
    training_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=args.batch_size,
        fp16=torch.cuda.is_available(),
        report_to="none",
        eval_accumulation_steps=16, # Same optimization as training to prevent OOM
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # Evaluate
    logger.info("Running evaluation...")
    metrics = trainer.evaluate()

    # Print Results
    logger.info("Evaluation Results:")
    print("\n" + "=" * 30)
    print(f"Model:         {args.model_path}")
    print(f"Data:          {args.csv}")
    print("-" * 30)
    print(f"F1 Macro:      {metrics.get('eval_f1_macro', 0):.4f}")
    print(f"Precision:     {metrics.get('eval_precision_macro', 0):.4f}")
    print(f"Recall:        {metrics.get('eval_recall_macro', 0):.4f}")
    print(f"Accuracy:      {metrics.get('eval_accuracy', 0):.4f}")
    print(f"Loss:          {metrics.get('eval_loss', 0):.4f}")
    print("=" * 30 + "\n")

if __name__ == "__main__":
    main()
