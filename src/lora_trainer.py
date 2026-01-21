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

from .config import PROMPT_TEMPLATE, LABEL_TO_ID, ID_TO_LABEL, LABEL_LIST


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA supervised fine-tuning."""
    model_name: str = "meta-llama/Llama-3.1-8B"
    output_dir: str = "artifacts/lora_llm"
    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 2e-4
    max_length: int = 256
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    eval_ratio: float = 0.1  # 10% data for evaluation
    early_stopping_patience: int = 3  # Stop if F1 doesn't improve for 3 evals
    
    # Prompt template for classification
    prompt_template: str = PROMPT_TEMPLATE


def _build_prompt(claim: str, evidence: str, template: str) -> str:
    """Build prompt from claim and evidence."""
    return template.format(claim=claim, evidence=evidence)


def _get_label_token_ids(tokenizer, labels: list = None):
    """
    Get token ID for each label using special tokens.
    Used for logits-based classification (pLM from paper).
    
    Special tokens ensure each label is exactly 1 token, making pLM extraction reliable.
    """
    if labels is None:
        labels = LABEL_LIST
    
    label_token_ids = {}
    for label in labels:
        # Use special token format: <LABEL>
        special_token = f"<{label}>"
        tokens = tokenizer(special_token, add_special_tokens=False)["input_ids"]
        if len(tokens) != 1:
            raise ValueError(f"Special token '{special_token}' should be exactly 1 token, got {len(tokens)}")
        label_token_ids[label] = tokens[0]
    
    return label_token_ids


def compute_metrics(eval_pred, tokenizer, label_token_ids):
    """
    Compute F1, Precision, Recall, Accuracy for evaluation.
    
    PAPER-ACCURATE: Extracts pLM(y|q) from logits, not from text generation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
            - predictions: preprocessed logits with shape [batch, 3] (label logits only)
            - labels: label IDs with shape [batch, seq_len], -100 for masked positions
        tokenizer: Tokenizer to decode labels
        label_token_ids: Dict mapping label names to their token IDs
    
    Returns:
        Dict of metrics (F1, precision, recall, accuracy)
    """
    logits, labels = eval_pred
    
    # logits shape: [batch, 3] - already preprocessed to label logits only!
    # labels shape: [batch, seq_len]
    
    pred_labels = []
    true_labels = []
    
    for batch_idx in range(len(logits)):
        # Logits are already extracted for label tokens: [3] = [SUPPORTED, REFUTED, NEI]
        label_logits_array = logits[batch_idx]  # Shape: [3]
        
        # Apply softmax to get probabilities (pLM)
        exp_logits = np.exp(label_logits_array - np.max(label_logits_array))  # numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Choose label with highest probability
        pred_label_idx = np.argmax(probs)
        pred_label = LABEL_LIST[pred_label_idx]
        pred_labels.append(LABEL_TO_ID[pred_label])
        
        # Extract true label from labels
        batch_labels = labels[batch_idx]
        label_positions = np.where(batch_labels != -100)[0]
        
        if len(label_positions) == 0:
            logger.warning(f"Sample {batch_idx}: No valid label position found, using NEI as default")
            true_labels.append(LABEL_TO_ID["NEI"])
            continue
        valid_label_ids = batch_labels[label_positions]
        true_label_token = valid_label_ids[0]  # First token of label
        
        # Map back to label name
        true_label = "NEI"  # default
        for label_name, token_id in label_token_ids.items():
            if token_id == true_label_token:
                true_label = label_name
                break
        
        true_labels.append(LABEL_TO_ID[true_label])
    
    if len(pred_labels) == 0:
        logger.error("No valid predictions found!")
        return {
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "accuracy": 0.0,
        }
    
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
    del pred_labels, true_labels, logits, labels
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
        # CRITICAL FIX: Add TRUE/FALSE/NEUTRAL mappings (from finfact dataset)
        if label_upper in ["FALSE", "SCAM", "1"]:
            return "REFUTED"
        if label_upper in ["TRUE", "LEGIT", "LEGITIMATE", "0"]:
            return "SUPPORTED"
        if label_upper in ["NEUTRAL"]:
            return "NEI"
        return "NEI"
    
    for claim, evidence, label in zip(claims, evidences, labels):
        prompt = _build_prompt(claim, evidence, prompt_template)
        target = normalize_label(label)
        prompts.append(prompt)
        targets.append(target)
    
    # Tokenize
    def tokenize_function(examples):
        """
        FIXED: Build input_ids by direct concatenation to avoid alignment bugs.
        This ensures labels mask exactly the prompt tokens.
        """
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for prompt, target in zip(examples["prompt"], examples["target"]):
            # Tokenize prompt with special tokens (BOS)
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,  # We'll handle truncation manually
            )["input_ids"]
            
            # Tokenize target using special token format (NO space prefix)
            # CRITICAL: Must match _get_label_token_ids() format exactly
            # "<LABEL>" not " <LABEL>" to ensure single token
            target_text = f"<{target}>"
            target_ids = tokenizer(
                target_text,
                add_special_tokens=False,
            )["input_ids"]
            
            # Add EOS token at the end
            if tokenizer.eos_token_id is not None:
                target_ids = target_ids + [tokenizer.eos_token_id]
            
            # Concatenate: [BOS, prompt_tokens, target_tokens, EOS]
            full_input_ids = prompt_ids + target_ids
            
            # Truncate if too long (truncate prompt, keep target intact)
            if len(full_input_ids) > max_length:
                # Calculate how much to keep from prompt
                target_len = len(target_ids)
                max_prompt_len = max_length - target_len
                if max_prompt_len < 1:
                    # Edge case: target itself is too long, keep at least BOS
                    max_prompt_len = 1
                    target_ids = target_ids[:max_length - 1]
                
                prompt_ids = prompt_ids[:max_prompt_len]
                full_input_ids = prompt_ids + target_ids
            
            # Create labels: mask prompt (all -100), keep ONLY label token, mask EOS
            # This is more paper-accurate: only train on the label token itself
            prompt_len = len(prompt_ids)
            
            # Count label tokens (excluding EOS)
            label_token_count = len(target_ids) - (1 if tokenizer.eos_token_id is not None else 0)
            
            # Labels: [-100 for prompt] + [label_tokens] + [-100 for EOS]
            labels = [-100] * prompt_len + target_ids[:label_token_count] + [-100] * (len(target_ids) - label_token_count)
            
            # Pad to max_length
            padding_length = max_length - len(full_input_ids)
            if padding_length > 0:
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                full_input_ids = full_input_ids + [pad_token_id] * padding_length
                labels = labels + [-100] * padding_length
                attention_mask = [1] * (max_length - padding_length) + [0] * padding_length
            else:
                attention_mask = [1] * max_length
            
            model_inputs["input_ids"].append(full_input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
        
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
    
    def on_prediction_step(self, args, state, control, **kwargs):
        """Clean up after each prediction step during evaluation."""
        # Clean every 10 predictions to prevent accumulation
        if state.global_step % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def train_lora_classification(
    claims: List[str],
    evidences: List[str],
    labels: List[str],
    config: Optional[LoRATrainingConfig] = None,
    gradient_accumulation_steps: int = 4,
    skip_final_eval: bool = True,  # Skip final eval by default to prevent OOM
    checkpoint_path: Optional[str] = None,  # Path to existing checkpoint to resume training
) -> str:
    """
    Train LLM with LoRA for classification task.
    
    Args:
        claims: List of claims to verify
        evidences: List of retrieved evidence for each claim
        labels: List of ground truth labels (SUPPORTED/REFUTED/NEI)
        config: Training configuration
        gradient_accumulation_steps: Number of gradient accumulation steps
        skip_final_eval: Skip final evaluation to save memory
        checkpoint_path: Path to existing LoRA checkpoint to resume training (optional)
    
    Returns:
        Path to saved LoRA model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch/transformers/peft/datasets are required for LoRA training.")
    
    config = config or LoRATrainingConfig()
    
    # Check if resuming from checkpoint
    import os
    from peft import PeftModel
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        
        # Check if checkpoint has adapter files
        adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.exists(adapter_config):
            raise ValueError(f"Invalid checkpoint: missing adapter_config.json in {checkpoint_path}")
        
        # Load tokenizer from checkpoint (if available) or base model
        if os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")):
            logger.info(f"Loading tokenizer from checkpoint...")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False, trust_remote_code=True)
        else:
            logger.info(f"Loading tokenizer from base model...")
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens for labels if not already present
        special_tokens = {"additional_special_tokens": ["<SUPPORTED>", "<REFUTED>", "<NEI>"]}
        existing_special = set(tokenizer.additional_special_tokens or [])
        new_special = [t for t in special_tokens["additional_special_tokens"] if t not in existing_special]
        
        if new_special:
            tokenizer.add_special_tokens({"additional_special_tokens": new_special})
            logger.info(f"Added {len(new_special)} special tokens for labels to checkpoint tokenizer")
        
        # Load base model
        logger.info(f"Loading base model: {config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        # Handle tie_word_embeddings for models like Mistral/LLaMA
        if hasattr(base_model.config, 'tie_word_embeddings') and base_model.config.tie_word_embeddings:
            logger.info("Model has tie_word_embeddings=True, handling for resize...")
            base_model.config.tie_word_embeddings = False
        
        # CRITICAL: Resize embeddings BEFORE enabling gradient checkpointing or loading LoRA
        # Must happen before PEFT wraps the embeddings!
        current_embed_size = base_model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > current_embed_size:
            logger.info(f"Resizing embeddings: {current_embed_size} → {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
        elif len(tokenizer) != current_embed_size:
            logger.warning(f"Tokenizer size ({len(tokenizer)}) != embedding size ({current_embed_size})")
        
        # Enable gradient checkpointing
        base_model.gradient_checkpointing_enable()
        
        # Disable cache when using gradient checkpointing
        base_model.config.use_cache = False
        
        # Load LoRA adapter from checkpoint
        logger.info(f"Loading LoRA adapter from checkpoint...")
        model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=True)
        model.print_trainable_parameters()
        
    else:
        # Create new model from scratch
        if checkpoint_path:
            logger.warning(f"⚠️  Checkpoint path provided but not found: {checkpoint_path}")
            logger.info("Creating new LoRA model from scratch...")
        else:
            logger.info(f"Creating new LoRA model from {config.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens for labels (single-token classification)
        special_tokens = {"additional_special_tokens": ["<SUPPORTED>", "<REFUTED>", "<NEI>"]}
        num_added = tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens for labels")
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        # Handle tie_word_embeddings for models like Mistral/LLaMA
        # Must untie before resize, then resize will handle correctly
        if hasattr(model.config, 'tie_word_embeddings') and model.config.tie_word_embeddings:
            logger.info("Model has tie_word_embeddings=True, handling for resize...")
            model.config.tie_word_embeddings = False  # Temporarily untie
        
        # Resize token embeddings to accommodate new special tokens
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized embeddings to {len(tokenizer)}")
        
        # Enable gradient checkpointing to save VRAM
        model.gradient_checkpointing_enable()
        
        # FIXED: Disable cache when using gradient checkpointing
        model.config.use_cache = False
        
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
        save_steps=200,  # Save checkpoint every 200 steps
        save_total_limit=3,  # Keep 3 checkpoints (best + recent ones for resuming)
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=200,  
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_strategy="steps",
        eval_accumulation_steps=64, 
        max_grad_norm=1.0,
        dataloader_num_workers=0, 
        ddp_find_unused_parameters=False,
    )
    
    from transformers import default_data_collator
    data_collator = default_data_collator
    
    # Get label token IDs for logits-based classification
    label_token_ids = _get_label_token_ids(tokenizer)
    
    # Create compute_metrics function with tokenizer and label_token_ids closure
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, tokenizer, label_token_ids)
    
    # CRITICAL: Preprocess logits to reduce memory usage during evaluation
    # Without this, logits accumulation causes CUDA OOM (22GB+ for 337 samples)
    def preprocess_logits_for_metrics(logits, labels):
        """
        Reduce full logits [batch, seq_len, vocab_size] to label logits [batch, 3]
        BEFORE accumulation to save massive amounts of memory.
        
        For each sample, we extract:
        - Find label position from labels
        - Apply CausalLM shift (pred_pos = label_pos - 1)
        - Extract logits for 3 label tokens only
        
        Memory savings: ~65MB/sample → ~200 bytes/sample (300x reduction!)
        """
        # logits: [batch, seq_len, vocab_size]
        # labels: [batch, seq_len]
        batch_size = logits.shape[0]
        
        # Pre-allocate for label logits only [batch, 3]
        label_logits_batch = torch.zeros((batch_size, 3), device=logits.device, dtype=logits.dtype)
        
        # Extract token IDs for labels
        label_token_id_list = [label_token_ids[label] for label in LABEL_LIST]
        
        for i in range(batch_size):
            # Find label position
            label_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
            
            if len(label_positions) == 0:
                # No label found, use zeros (will be handled in compute_metrics)
                continue
            
            label_pos = label_positions[0].item()
            pred_pos = label_pos - 1
            
            if pred_pos < 0:
                # Can't predict position 0
                continue
            
            # Extract logits at prediction position for label tokens only
            # This is the key: extract ONLY 3 values instead of entire vocab
            for j, token_id in enumerate(label_token_id_list):
                label_logits_batch[i, j] = logits[i, pred_pos, token_id]
        
        # Return reduced logits [batch, 3] instead of [batch, seq_len, vocab_size]
        # This is ~300x smaller!
        return label_logits_batch
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # KEY FIX!
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
    
    # Final evaluation (optional - can skip to save memory)
    if not skip_final_eval:
        logger.info("Running final evaluation...")
        final_metrics = trainer.evaluate()
        logger.info(f"Final metrics: F1={final_metrics.get('eval_f1_macro', 0):.4f}, "
                    f"Precision={final_metrics.get('eval_precision_macro', 0):.4f}, "
                    f"Recall={final_metrics.get('eval_recall_macro', 0):.4f}, "
                    f"Accuracy={final_metrics.get('eval_accuracy', 0):.4f}")
        
        # Clear memory after eval
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("⚠️  Skipping final evaluation to prevent CUDA OOM")
        logger.info("💡 You already have eval metrics from training (check logs above for eval_f1_macro)")
        final_metrics = {}
    
    # Save BEST model (according to F1 metric)
    best_model_dir = config.output_dir
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info(f"✅ Best model (F1={final_metrics.get('eval_f1_macro', 0):.4f}) saved to {best_model_dir}")
    
    logger.info(f"ℹ️  Intermediate checkpoints (e.g., checkpoint-XXX) are saved in {config.output_dir}")
    logger.info(f"ℹ️  To resume training, use the latest checkpoint folder found there.")
    
    return config.output_dir