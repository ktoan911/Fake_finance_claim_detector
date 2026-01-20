"""
LoRA Fine-tuning for LLM (Supervised Classification)

Implements paper's LoRA fine-tuning step.
LLM learns to predict: pLM(y | query, context)

Input: Claim + Retrieved evidence
Output: Label (SUPPORTED / REFUTED / NEI)

This is SUPERVISED fine-tuning and requires labeled data.
"""

from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
import gc

try:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        TrainerCallback
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA supervised fine-tuning."""
    model_name: str = "meta-llama/Llama-3.1-8B"
    output_dir: str = "artifacts/lora_llm"
    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 2e-4
    max_length: int = 256
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    eval_ratio: float = 0.1  # 10% data for evaluation
    early_stopping_patience: int = 3  # Stop if F1 doesn't improve for 3 evals
    
    # Prompt template for classification
    prompt_template: str = """You are a crypto claim verification assistant.

Given the following claim and evidence, classify the claim as:
- SUPPORTED: The evidence supports the claim
- REFUTED: The evidence contradicts the claim
- NEI: Not Enough Information to verify

Claim: {claim}

Evidence: {evidence}

Classification:"""


def _build_prompt(claim: str, evidence: str, template: str) -> str:
    """Build prompt from claim and evidence."""
    return template.format(claim=claim, evidence=evidence)


# Label mapping for metrics computation
LABEL_TO_ID = {"SUPPORTED": 0, "REFUTED": 1, "NEI": 2}
ID_TO_LABEL = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}


def compute_metrics(eval_pred, tokenizer):
    """
    Compute F1, Precision, Recall, Accuracy for evaluation.
    Optimizes for macro F1 score.
    Memory-optimized version with explicit cleanup.
    """
    predictions, labels = eval_pred
    
    # For causal LM, we need to decode and extract the predicted label
    pred_labels = []
    true_labels = []
    
    for pred_ids, label_ids in zip(predictions, labels):
        # Find the predicted token (last non-padding token)
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        
        # Extract predicted label from generated text
        pred_label = "NEI"  # default
        for label in ["SUPPORTED", "REFUTED", "NEI"]:
            if label in pred_text.upper():
                pred_label = label
                break
        pred_labels.append(LABEL_TO_ID[pred_label])
        
        # Extract true label from label_ids (non -100 tokens)
        valid_label_ids = [l for l in label_ids if l != -100]
        if valid_label_ids:
            true_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
            true_label = "NEI"
            for label in ["SUPPORTED", "REFUTED", "NEI"]:
                if label in true_text.upper():
                    true_label = label
                    break
            true_labels.append(LABEL_TO_ID[true_label])
        else:
            true_labels.append(LABEL_TO_ID["NEI"])
    
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    
    # Compute metrics
    metrics = {
        "f1_macro": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "precision_macro": precision_score(true_labels, pred_labels, average="macro", zero_division=0),
        "recall_macro": recall_score(true_labels, pred_labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }
    
    # Clean up memory
    del pred_labels, true_labels, predictions, labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics


def _prepare_classification_dataset(
    claims: List[str],
    evidences: List[str],
    labels: List[str],
    tokenizer,
    max_length: int,
    prompt_template: str
):
    """
    Prepare dataset for supervised classification fine-tuning.
    Format: prompt + label (causal LM style)
    """
    prompts = []
    targets = []
    
    def normalize_label(label_value) -> str:
        if isinstance(label_value, (int, float)):
            idx = int(label_value)
            if idx == 0:
                return "SUPPORTED"
            if idx == 1:
                return "REFUTED"
            return "NEI"
        
        label_upper = str(label_value).upper().strip()
        if label_upper in ["SUPPORTED", "REFUTED", "NEI"]:
            return label_upper
        if label_upper in ["SCAM", "1"]:
            return "REFUTED"
        if label_upper in ["LEGIT", "LEGITIMATE", "0"]:
            return "SUPPORTED"
        return "NEI"
    
    for claim, evidence, label in zip(claims, evidences, labels):
        prompt = _build_prompt(claim, evidence, prompt_template)
        target = normalize_label(label)
        prompts.append(prompt)
        targets.append(target)
    
    # Tokenize
    def tokenize_function(examples):
        # Tokenize prompts and targets separately to get accurate lengths
        prompts_only = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        
        # Combine prompt + target for causal LM training
        full_texts = [p + " " + t for p, t in zip(examples["prompt"], examples["target"])]
        
        model_inputs = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
        )
        
        # Create labels by masking prompt tokens
        labels = []
        for i, (input_ids, prompt_ids) in enumerate(zip(model_inputs["input_ids"], prompts_only["input_ids"])):
            # Find the actual prompt length in the full sequence
            prompt_len = len(prompt_ids)
            
            # Create label sequence: mask prompt, keep target
            label = [-100] * prompt_len + input_ids[prompt_len:]
            
            # Ensure label has same length as input_ids
            if len(label) > len(input_ids):
                label = label[:len(input_ids)]
            elif len(label) < len(input_ids):
                # Pad with -100 if needed
                label = label + [-100] * (len(input_ids) - len(label))
            
            # Also mask padding tokens
            attention_mask = model_inputs["attention_mask"][i]
            label = [l if attention_mask[j] == 1 else -100 for j, l in enumerate(label)]
            
            labels.append(label)
        
        model_inputs["labels"] = labels
        
        return model_inputs
    
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "target": targets
    })
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "target"]
    )
    
    return tokenized


class MemoryCleanupCallback(TrainerCallback):
    """
    Callback to aggressively clean up memory after evaluation to prevent OOM.
    """
    def on_evaluate(self, args, state, control, **kwargs):
        """Clean up memory after evaluation."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Periodically clean up memory during training."""
        if state.global_step % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def train_lora_classification(
    claims: List[str],
    evidences: List[str],
    labels: List[str],
    config: Optional[LoRATrainingConfig] = None,
    gradient_accumulation_steps: int = 4,
) -> str:
    """
    Train LLM with LoRA for classification task.
    
    Args:
        claims: List of claims to verify
        evidences: List of retrieved evidence for each claim
        labels: List of ground truth labels (SUPPORTED/REFUTED/NEI)
        config: Training configuration
        gradient_accumulation_steps: Number of gradient accumulation steps
    
    Returns:
        Path to saved LoRA model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch/transformers/peft/datasets are required for LoRA training.")
    
    config = config or LoRATrainingConfig()
    
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    
    # Configure LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Attention layers
    )
    
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    
    # Prepare dataset
    logger.info(f"Preparing dataset with {len(claims)} samples...")
    full_dataset = _prepare_classification_dataset(
        claims, evidences, labels,
        tokenizer, config.max_length, config.prompt_template
    )
    
    # Split into train and eval datasets for F1 optimization
    split_dataset = full_dataset.train_test_split(
        test_size=config.eval_ratio,
        seed=42,
        shuffle=True
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Training arguments - optimized for F1 score with memory management
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_steps=200,  # Reduced save frequency to save memory
        save_total_limit=2,  # Keep only 2 checkpoints to save disk space
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=200,  # Evaluate less frequently to reduce memory pressure
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_strategy="steps",
        eval_accumulation_steps=8,  # Accumulate more to reduce memory footprint
        # Gradient clipping to prevent NaN gradients
        max_grad_norm=1.0,
        # Memory optimization settings
        dataloader_num_workers=0,  # Avoid multiprocessing overhead
        ddp_find_unused_parameters=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Create compute_metrics function with tokenizer closure
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
            MemoryCleanupCallback(),  # Add memory cleanup
        ],
    )
    
    logger.info("Starting LoRA fine-tuning with F1 optimization...")
    
    # Clear cache before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.train()
    
    # Clear cache after training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    logger.info(f"Final metrics: F1={final_metrics.get('eval_f1_macro', 0):.4f}, "
                f"Precision={final_metrics.get('eval_precision_macro', 0):.4f}, "
                f"Recall={final_metrics.get('eval_recall_macro', 0):.4f}, "
                f"Accuracy={final_metrics.get('eval_accuracy', 0):.4f}")
    
    # Save best model
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info(f"LoRA model saved to {config.output_dir}")
    return config.output_dir