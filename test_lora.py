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
    DataCollatorForSeq2Seq,
)
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from src.csv_loader import CSVLabeledLoader
from src.lora_trainer import _prepare_classification_dataset
from src.config import PROMPT_TEMPLATE

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model F1 score")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved LoRA adapter (e.g., artifacts/lora_llm)")
    parser.add_argument("--csv", type=str, required=True, help="Path to labeled CSV file")
    parser.add_argument("--base-model", type=str, default=None, help="Override base model path (optional, auto-detected from adapter)")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length (reduced for lower RAM usage)")
    parser.add_argument("--eval-accumulation", type=int, default=4, help="Number of batches to accumulate before metric computation")
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

    # Auto-detect base model from adapter_config.json
    logger.info(f"Reading adapter config from {args.model_path}...")
    adapter_config = PeftConfig.from_pretrained(args.model_path)
    base_model_path = args.base_model or adapter_config.base_model_name_or_path
    logger.info(f"Base model: {base_model_path}")

    # Load Tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from checkpoint ({e}), loading from base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    logger.info(f"Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Prepare Dataset
    logger.info("Preparing dataset...")
    dataset = _prepare_classification_dataset(
        claims, evidences, labels, tokenizer, args.max_length, PROMPT_TEMPLATE
    )

    # Prepare DataLoader for manual evaluation
    logger.info("Preparing dataloader...")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=False  # Disable pin_memory to save RAM
    )
    
    # Custom evaluation loop - Memory efficient
    logger.info("Running evaluation...")
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels_batch = batch["labels"].to(model.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )
            
            # Get loss
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
                num_batches += 1
            
            # Get predictions - extract the generated label token
            logits = outputs.logits
            
            # Find the position where we predict the label (after the prompt)
            # We look at the last non-padding position
            for i in range(len(labels_batch)):
                label_positions = (labels_batch[i] != -100).nonzero(as_tuple=True)[0]
                if len(label_positions) > 0:
                    # Get the first label position (where we predict "true" or "false")
                    label_pos = label_positions[0].item()
                    pred_logits = logits[i, label_pos - 1, :]  # Get logits before the label
                    
                    # Get the tokens for "true" and "false"
                    true_token_id = tokenizer.encode("true", add_special_tokens=False)[0]
                    false_token_id = tokenizer.encode("false", add_special_tokens=False)[0]
                    
                    # Compare logits for true vs false
                    true_score = pred_logits[true_token_id].item()
                    false_score = pred_logits[false_token_id].item()
                    
                    # Predict based on higher score
                    pred_label = 1 if true_score > false_score else 0
                    
                    # Get ground truth label
                    label_token_id = labels_batch[i, label_pos].item()
                    gt_label = 1 if label_token_id == true_token_id else 0
                    
                    # Store predictions and labels (move to CPU immediately)
                    all_predictions.append(pred_label)
                    all_labels.append(gt_label)
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available() and num_batches % 10 == 0:
                torch.cuda.empty_cache()
    
    # Compute metrics using sklearn
    logger.info("Computing metrics...")
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = {
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'accuracy': accuracy,
        'loss': avg_loss
    }
    
    # Clean up memory after evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print Results
    logger.info("Evaluation Results:")
    print("\n" + "=" * 30)
    print(f"Model:         {args.model_path}")
    print(f"Data:          {args.csv}")
    print("-" * 30)
    print(f"F1 Macro:      {metrics['f1_macro']:.4f}")
    print(f"Precision:     {metrics['precision_macro']:.4f}")
    print(f"Recall:        {metrics['recall_macro']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Loss:          {metrics['loss']:.4f}")
    print(f"Samples:       {len(all_labels)}")
    print("=" * 30 + "\n")

if __name__ == "__main__":
    main()
