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

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
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
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    # Prompt template for classification
    prompt_template: str = """You are a crypto claim verification assistant.

Given the following claim and evidence, classify the claim as:
- SUPPORTED: The evidence supports the claim
- REFUTED: The evidence contradicts the claim  
- NEI: Not Enough Information to verify

Claim: {claim}

Evidence:
{evidence}

Classification:"""


def _build_prompt(claim: str, evidence: str, template: str) -> str:
    """Build prompt from claim and evidence."""
    return template.format(claim=claim, evidence=evidence)


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
        # Combine prompt + target for causal LM training
        full_texts = [p + " " + t for p, t in zip(examples["prompt"], examples["target"])]
        
        model_inputs = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
        # For causal LM, labels = input_ids (shifted internally by model)
        # Deep copy to avoid modifying input_ids
        model_inputs["labels"] = [ids.copy() if isinstance(ids, list) else list(ids) for ids in model_inputs["input_ids"]]
        
        # Mask prompt tokens in labels (only compute loss on target)
        prompt_lengths = [
            len(tokenizer(p, truncation=True, max_length=max_length)["input_ids"])
            for p in examples["prompt"]
        ]
        
        for i, prompt_len in enumerate(prompt_lengths):
            # -100 = ignore in loss computation
            model_inputs["labels"][i] = [-100] * prompt_len + model_inputs["labels"][i][prompt_len:]
        
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
    dataset = _prepare_classification_dataset(
        claims, evidences, labels,
        tokenizer, config.max_length, config.prompt_template
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()
    
    # Save
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info(f"LoRA model saved to {config.output_dir}")
    return config.output_dir

