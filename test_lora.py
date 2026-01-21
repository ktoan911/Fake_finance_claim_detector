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
    
    logger.info("Running evaluation...")
    all_predictions = []
    all_labels = []
    skipped_samples = 0
    
    # Label mapping to match training (SUPPORTED=0, REFUTED=1, NEI=2)
    # But for binary classification in test set: SUPPORTED/LEGIT=0, REFUTED/SCAM=1
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels_batch = batch["labels"].to(model.device)
            
            # Forward pass to get logits
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits
            
            # Process each sample in the batch
            for i in range(len(labels_batch)):
                # Find where the label tokens are (non -100 positions)
                label_positions = (labels_batch[i] != -100).nonzero(as_tuple=True)[0]
                
                if len(label_positions) == 0:
                    skipped_samples += 1
                    logger.warning(f"Batch {batch_idx}, sample {i}: No label position found, skipping")
                    continue
                
                # Get the first label position (where model should predict the label)
                label_start_pos = label_positions[0].item()
                
                # Get predicted tokens by taking argmax over vocabulary
                # We need to look at a few tokens to capture full label (e.g., "SUPPORTED", "REFUTED")
                num_label_tokens = min(5, len(label_positions))  # Labels are typically 1-3 tokens
                pred_token_ids = []
                
                for offset in range(num_label_tokens):
                    pos = label_start_pos + offset - 1  # -1 because we predict next token
                    if pos >= 0 and pos < logits.shape[1]:
                        pred_token_id = torch.argmax(logits[i, pos, :]).item()
                        pred_token_ids.append(pred_token_id)
                
                # Decode predicted tokens to text
                pred_text = tokenizer.decode(pred_token_ids, skip_special_tokens=True).strip().upper()
                
                # Get ground truth tokens from label_positions
                gt_token_ids = labels_batch[i, label_positions[:num_label_tokens]].tolist()
                gt_text = tokenizer.decode(gt_token_ids, skip_special_tokens=True).strip().upper()
                
                # Debug: Log first 10 samples to understand the format
                if len(all_labels) + skipped_samples < 10:
                    logger.info(f"Sample {len(all_labels) + skipped_samples}: GT='{gt_text}' | PRED='{pred_text}'")
                
                # Map to binary classification (0=SUPPORTED/LEGIT, 1=REFUTED/SCAM)
                # Prediction
                if "SUPPORTED" in pred_text or "LEGIT" in pred_text:
                    pred_label = 0
                elif "REFUTED" in pred_text or "SCAM" in pred_text:
                    pred_label = 1
                else:
                    # NEI or unclear - default to 0
                    pred_label = 0
                
                # Ground truth
                if "SUPPORTED" in gt_text or "LEGIT" in gt_text:
                    gt_label = 0
                elif "REFUTED" in gt_text or "SCAM" in gt_text:
                    gt_label = 1
                else:
                    # NEI or unclear - skip this sample
                    skipped_samples += 1
                    logger.warning(f"Batch {batch_idx}, sample {i}: Unclear GT label '{gt_text}', skipping")
                    continue
                
                # Store predictions and labels
                all_predictions.append(pred_label)
                all_labels.append(gt_label)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_idx % 20 == 0:
                torch.cuda.empty_cache()
    
    # Compute metrics using sklearn
    logger.info("Computing metrics...")
    logger.info(f"Processed {len(all_labels)} samples, skipped {skipped_samples} samples")
    
    if len(all_labels) == 0:
        logger.error("No samples were successfully processed!")
        return
    
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    metrics = {
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'accuracy': accuracy,
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
    print("-" * 30)
    print(f"Processed:     {len(all_labels)}")
    print(f"Skipped:       {skipped_samples}")
    print(f"Total:         {len(all_labels) + skipped_samples}")
    print("=" * 30 + "\n")

if __name__ == "__main__":
    main()
