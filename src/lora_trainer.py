"""
LoRA Fine-tuning for LLM (Supervised Classification) - padding=max_length

Stable fixed-length training:
- tokenize with padding="max_length"
- prompt masking uses sum(attention_mask) (NOT len(input_ids))
- use default_data_collator (dataset already fixed-length)
- compute_metrics works with logits or token ids
- preprocess_logits_for_metrics reduces eval RAM
"""

from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

try:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        default_data_collator,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LoRATrainingConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    output_dir: str = "artifacts/lora_llm"
    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 5e-5  # ↓ safer than 2e-4 for fp16
    max_length: int = 256
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    eval_ratio: float = 0.1
    early_stopping_patience: int = 3

    prompt_template: str = """You are a crypto claim verification assistant.

Given the following claim and evidence, classify the claim as:
- SUPPORTED: The evidence supports the claim
- REFUTED: The evidence contradicts the claim
- NEI: Not Enough Information to verify

Claim: {claim}

Evidence:
{evidence}

Classification:"""


LABEL_TO_ID = {"SUPPORTED": 0, "REFUTED": 1, "NEI": 2}


def _build_prompt(claim: str, evidence: str, template: str) -> str:
    return template.format(claim=claim, evidence=evidence)


def _extract_label_from_text(text: str) -> str:
    up = (text or "").upper()
    for lab in ["SUPPORTED", "REFUTED", "NEI"]:
        if lab in up:
            return lab
    return "NEI"


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]

    # logits -> token ids
    if hasattr(predictions, "ndim") and predictions.ndim == 3:
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions

    pred_labels = []
    true_labels = []

    for p_ids, l_ids in zip(pred_ids, labels):
        pred_text = tokenizer.decode(p_ids, skip_special_tokens=True)
        pred_lab = _extract_label_from_text(pred_text)
        pred_labels.append(LABEL_TO_ID[pred_lab])

        valid_label_ids = [int(x) for x in l_ids if int(x) != -100]
        if valid_label_ids:
            true_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
            true_lab = _extract_label_from_text(true_text)
        else:
            true_lab = "NEI"
        true_labels.append(LABEL_TO_ID[true_lab])

    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)

    return {
        "f1_macro": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "precision_macro": precision_score(true_labels, pred_labels, average="macro", zero_division=0),
        "recall_macro": recall_score(true_labels, pred_labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }


def _prepare_classification_dataset(
    claims: List[str],
    evidences: List[str],
    labels: List[str],
    tokenizer,
    max_length: int,
    prompt_template: str,
):
    prompts, targets = [], []

    def normalize_label(label_value) -> str:
        if isinstance(label_value, (int, float)):
            idx = int(label_value)
            return "SUPPORTED" if idx == 0 else "REFUTED" if idx == 1 else "NEI"

        s = str(label_value).upper().strip()
        if s in ["SUPPORTED", "REFUTED", "NEI"]:
            return s
        if s in ["SCAM", "1"]:
            return "REFUTED"
        if s in ["LEGIT", "LEGITIMATE", "0"]:
            return "SUPPORTED"
        return "NEI"

    for c, e, y in zip(claims, evidences, labels):
        prompts.append(_build_prompt(c, e, prompt_template))
        targets.append(normalize_label(y))

    def tokenize_function(examples):
        full_texts = [p + " " + t for p, t in zip(examples["prompt"], examples["target"])]

        model_inputs = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # labels = input_ids copy
        model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]

        # ✅ prompt length = sum(attention_mask), NOT len(input_ids)
        prompt_tok = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        prompt_lengths = [int(sum(m)) for m in prompt_tok["attention_mask"]]

        for i, p_len in enumerate(prompt_lengths):
            p_len = min(p_len, max_length)
            model_inputs["labels"][i][:p_len] = [-100] * p_len

        return model_inputs

    dataset = Dataset.from_dict({"prompt": prompts, "target": targets})
    return dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "target"])


def train_lora_classification(
    claims: List[str],
    evidences: List[str],
    labels: List[str],
    config: Optional[LoRATrainingConfig] = None,
    gradient_accumulation_steps: int = 4,
) -> str:
    if not TORCH_AVAILABLE:
        raise ImportError("torch/transformers/peft/datasets are required for LoRA training.")

    config = config or LoRATrainingConfig()

    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.max_length  # ✅ removes max_length ambiguity

    logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    logger.info(f"Preparing dataset with {len(claims)} samples...")
    full_dataset = _prepare_classification_dataset(
        claims, evidences, labels,
        tokenizer, config.max_length, config.prompt_template
    )

    split_dataset = full_dataset.train_test_split(test_size=config.eval_ratio, seed=42, shuffle=True)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,

        save_steps=100,
        save_total_limit=3,
        save_strategy="steps",

        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        weight_decay=0.01,

        evaluation_strategy="steps",
        eval_steps=100,

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        eval_accumulation_steps=8,
        dataloader_pin_memory=True,

        max_grad_norm=1.0,  # helps prevent fp16 NaN
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,  # ✅ safest for fixed-length
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    logger.info("Starting LoRA fine-tuning with F1 optimization...")
    trainer.train()

    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    logger.info(final_metrics)

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info(f"LoRA model saved to {config.output_dir}")
    return config.output_dir
