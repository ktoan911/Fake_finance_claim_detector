#!/usr/bin/env python3
"""
Test LoRA model (F1, Precision, Recall, Accuracy) on a labeled CSV dataset.
Updated to match paper-accurate logits-based implementation with existing vocab tokens.
"""

import argparse
import os
import torch
import gc
import numpy as np
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
)
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from src.csv_loader import CSVLabeledLoader
from src.lora_trainer import _prepare_classification_dataset, _get_label_token_ids
from src.config import PROMPT_TEMPLATE, LABEL_TO_ID, LABEL_LIST

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model with logits-based classification")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved LoRA adapter")
    parser.add_argument("--csv", type=str, required=True, help="Path to labeled CSV file")
    parser.add_argument("--base-model", type=str, default=None, help="Override base model path")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
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

    # Auto-detect base model
    logger.info(f"Reading adapter config from {args.model_path}...")
    adapter_config = PeftConfig.from_pretrained(args.model_path)
    base_model_path = args.base_model or adapter_config.base_model_name_or_path
    logger.info(f"Base model: {base_model_path}")

    # Load Tokenizer
    logger.info(f"Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get label token IDs (True/False/Unsure)
    logger.info("Extracting label token IDs...")
    label_token_ids = _get_label_token_ids(tokenizer)
    logger.info(f"Label token IDs: {label_token_ids}")

    # Load Model
    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    logger.info(f"Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Prepare Dataset
    logger.info("Preparing dataset...")
    dataset = _prepare_classification_dataset(
        claims, evidences, labels, tokenizer, args.max_length, PROMPT_TEMPLATE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=0,
        pin_memory=False
    )
    
    logger.info("Running logits-based evaluation...")
    all_predictions = []
    all_labels = []
    skipped_samples = 0
    
    # Get token IDs in LABEL_LIST order
    label_token_id_list = [label_token_ids[label] for label in LABEL_LIST]
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels_batch = batch["labels"].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            for i in range(len(labels_batch)):
                # Find label position
                label_positions = (labels_batch[i] != -100).nonzero(as_tuple=True)[0]
                
                if len(label_positions) == 0:
                    skipped_samples += 1
                    continue
                
                # CausalLM shift: logits[t] predicts token at t+1
                label_pos = label_positions[0].item()
                pred_pos = label_pos - 1
                
                if pred_pos < 0:
                    skipped_samples += 1
                    continue
                
                # Extract logits for label tokens
                position_logits = logits[i, pred_pos, :]
                label_logits = torch.tensor([
                    position_logits[token_id].item() for token_id in label_token_id_list
                ])
                
                # Softmax → probabilities → prediction
                probs = torch.softmax(label_logits, dim=0).cpu().numpy()
                pred_label_idx = int(np.argmax(probs))
                pred_label = LABEL_LIST[pred_label_idx]
                
                # Extract ground truth
                true_label_token = labels_batch[i, label_pos].item()
                true_label = "NEI"
                for label_name, token_id in label_token_ids.items():
                    if token_id == true_label_token:
                        true_label = label_name
                        break
                
                # Debug first 10
                if len(all_labels) < 10:
                    logger.info(f"Sample {len(all_labels)}: GT={true_label} | PRED={pred_label} | Probs={probs}")
                
                all_predictions.append(LABEL_TO_ID[pred_label])
                all_labels.append(LABEL_TO_ID[true_label])
            
            if torch.cuda.is_available() and batch_idx % 20 == 0:
                torch.cuda.empty_cache()
    
    # Compute metrics
    logger.info("Computing metrics...")
    logger.info(f"Processed {len(all_labels)} samples, skipped {skipped_samples} samples")
    
    if len(all_labels) == 0:
        logger.error("No samples processed!")
        return
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = {
        'f1_macro': f1_score(all_labels, all_predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'precision_macro': precision_score(all_labels, all_predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_predictions, average='macro', zero_division=0),
        'accuracy': accuracy_score(all_labels, all_predictions),
    }
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print Results
    print("\n" + "=" * 50)
    print(f"Model:         {args.model_path}")
    print(f"Data:          {args.csv}")
    print("-" * 50)
    print(f"F1 Macro:      {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted:   {metrics['f1_weighted']:.4f}")
    print(f"Precision:     {metrics['precision_macro']:.4f}")
    print(f"Recall:        {metrics['recall_macro']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print("-" * 50)
    print(f"Processed:     {len(all_labels)}")
    print(f"Skipped:       {skipped_samples}")
    print("=" * 50)
    
    # Per-class breakdown
    print("\nPer-class distribution:")
    for i, label_name in enumerate(LABEL_LIST):
        count = int(np.sum(all_labels == i))
        pred_count = int(np.sum(all_predictions == i))
        print(f"  {label_name}: GT={count}, Pred={pred_count}")

if __name__ == "__main__":
    main()
