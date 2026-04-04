# Script to train LoRA for 3-class claim classification (A=Đúng/B=Sai/C=Thiếu)
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

try:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    np = None
    torch = None
    Dataset = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    accuracy_score = None
    f1_score = None
    precision_score = None
    recall_score = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    DataCollatorForSeq2Seq = None
    EarlyStoppingCallback = None
    Trainer = None
    TrainingArguments = None
    TrainerCallback = object

from src.config import LABEL_LIST, LABEL_TO_ID, PROMPT_TEMPLATE

POSITIVE_LABEL = LABEL_LIST[0]
NEGATIVE_LABEL = LABEL_LIST[1]
NEI_LABEL = LABEL_LIST[2]


@dataclass
class LoRATrainingConfig:
    # Default to a Vietnamese base model to match Vietnamese labels/prompts.
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    output_dir: str = "models/lora_llm"
    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 5e-5
    max_length: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    eval_ratio: float = 0.1
    early_stopping_patience: int = 3
    precision: str = "auto"
    use_flash_attention: bool = True
    use_sdpa: bool = True

    prompt_template: str = PROMPT_TEMPLATE


def _load_tokenizer_for_training(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False, use_fast=True
        )
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)


def _load_causal_lm_for_training(
    model_name: str,
    torch_dtype,
    attn_implementation: str = "flash_attention_2",
):
    kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if attn_implementation and attn_implementation != "eager":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation=attn_implementation,
                **kwargs,
            )
            logger.info(f"✅ Attention implementation: {attn_implementation}")
            return model
        except Exception as e:
            if attn_implementation == "flash_attention_2":
                logger.warning(
                    f"attn_implementation='flash_attention_2' failed: {e}. "
                    "Falling back to 'sdpa' attention."
                )
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        attn_implementation="sdpa",
                        **kwargs,
                    )
                    logger.info("✅ Attention implementation: sdpa")
                    return model
                except Exception as e2:
                    logger.warning(
                        f"attn_implementation='sdpa' not supported: {e2}. "
                        "Falling back to 'eager' attention."
                    )
            else:
                logger.warning(
                    f"attn_implementation='{attn_implementation}' not supported by this model: {e}. "
                    "Falling back to 'eager' attention."
                )
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def _resolve_training_precision(precision: str):
    requested = (precision or "auto").lower().strip()
    valid_precisions = {"auto", "bf16", "fp16", "fp32"}
    if requested not in valid_precisions:
        raise ValueError(
            f"Unsupported precision '{precision}'. "
            f"Expected one of: {sorted(valid_precisions)}."
        )

    if not torch.cuda.is_available():
        if requested in {"bf16", "fp16"}:
            logger.warning(
                f"Requested precision '{requested}' requires CUDA. Falling back to fp32."
            )
        return torch.float32, False, False, "fp32"

    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False, "bf16"
        return torch.float16, False, True, "fp16"

    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "precision='bf16' requested but GPU does not support bfloat16."
            )
        return torch.bfloat16, True, False, "bf16"

    if requested == "fp16":
        return torch.float16, False, True, "fp16"

    return torch.float32, False, False, "fp32"


def _is_linear_like_module(module) -> bool:
    if isinstance(module, torch.nn.Linear):
        return True
    return "linear" in module.__class__.__name__.lower()


def _resolve_lora_target_modules(model) -> List[str]:
    module_suffixes = {name.split(".")[-1] for name, _ in model.named_modules() if name}

    preferred_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
        "Wqkv",
        "out_proj",
        "qkv_proj",
        "W_pack",
        "fc1",
        "fc2",
        "wq",
        "wk",
        "wv",
        "wo",
        "w1",
        "w2",
        "w3",
    ]
    detected = [name for name in preferred_suffixes if name in module_suffixes]
    if detected:
        return detected

    excluded_suffixes = {"lm_head", "embed_out", "classifier", "score", "qa_outputs"}
    excluded_name_fragments = ("embed_tokens", "word_embeddings", "wte")

    linear_suffixes = set()
    for name, module in model.named_modules():
        if not name:
            continue
        suffix = name.split(".")[-1]
        lower_name = name.lower()

        if suffix in excluded_suffixes:
            continue
        if any(fragment in lower_name for fragment in excluded_name_fragments):
            continue
        if _is_linear_like_module(module):
            linear_suffixes.add(suffix)

    if linear_suffixes:
        return sorted(linear_suffixes)

    available_preview = ", ".join(sorted(module_suffixes)[:30])
    raise ValueError(
        "Could not infer LoRA target modules for this model. "
        f"Available module suffixes (first 30): {available_preview}"
    )


def _build_prompt(claim: str, evidence: str, template: str) -> str:
    return template.format(claim=claim, evidence=evidence)


def _normalize_label(label_value) -> str:
    if isinstance(label_value, (int, float)):
        idx = int(label_value)
        if idx == 0:
            return POSITIVE_LABEL
        if idx == 1:
            return NEGATIVE_LABEL
        if idx == 2:
            return NEI_LABEL
        logger.warning(
            f"Unknown integer label '{label_value}'. Defaulting to {NEI_LABEL}."
        )
        return NEI_LABEL

    label_upper = str(label_value).upper().strip()
    # A = Đúng (supported/true)
    if label_upper in [
        POSITIVE_LABEL.upper().strip(),
        "TRUE",
        "SUPPORTED",
        "LEGIT",
        "LEGITIMATE",
        "ĐÚNG",
        "DUNG",
        "0",
    ]:
        return POSITIVE_LABEL
    # B = Sai (refuted/false)
    if label_upper in [
        NEGATIVE_LABEL.upper().strip(),
        "FALSE",
        "REFUTED",
        "SCAM",
        "SAI",
        "1",
    ]:
        return NEGATIVE_LABEL
    # C = Thiếu (not enough info)
    if label_upper in [
        NEI_LABEL.upper().strip(),
        "THIEU",
        "THIẾU",
        "NEI",
        "NOT ENOUGH INFO",
        "NOT ENOUGH INFORMATION",
        "INSUFFICIENT",
        "2",
    ]:
        return NEI_LABEL

    logger.warning(
        f"Unknown label '{label_value}' encountered during dataset prep. Defaulting to {NEI_LABEL}."
    )
    return NEI_LABEL


def _log_label_distribution(labels: List[str], title: str) -> None:
    if not labels:
        logger.warning(f"{title}: empty label list.")
        return

    normalized = [_normalize_label(label) for label in labels]
    counts = Counter(normalized)
    total = sum(counts.values())
    parts = []
    for label in LABEL_LIST:
        count = counts.get(label, 0)
        ratio = (count / total) * 100 if total else 0.0
        parts.append(f"{label}={count} ({ratio:.1f}%)")
    logger.info(f"{title} label distribution: " + ", ".join(parts))

    missing = [label for label in LABEL_LIST if counts.get(label, 0) == 0]
    if missing:
        logger.warning(f"{title} missing classes: {', '.join(missing)}")


def _warn_if_model_label_mismatch(model_name: str) -> None:
    combined = "".join(LABEL_LIST) + PROMPT_TEMPLATE
    has_non_ascii = any(ord(ch) > 127 for ch in combined)
    model_lower = (model_name or "").lower()
    if has_non_ascii and "phogpt" not in model_lower:
        logger.warning(
            "Using Vietnamese labels/prompts with a non-Vietnamese base model. "
            "Consider a Vietnamese base (e.g., PhoGPT) or switch labels/prompts to English."
        )


def _get_label_token_ids(tokenizer, labels: list = None):
    if labels is None:
        labels = LABEL_LIST

    label_token_ids = {}
    for label in labels:
        tokens = tokenizer(label, add_special_tokens=False)["input_ids"]

        if len(tokens) != 1:
            raise ValueError(
                f"Label '{label}' must tokenize to exactly 1 token, got {len(tokens)} tokens: {tokens}. "
                "Update LABEL_LIST in src/config.py to single-token labels for this tokenizer."
            )

        label_token_ids[label] = tokens[0]

        logger.debug(f"Label '{label}' -> token_id {tokens[0]}")

    return label_token_ids


def compute_metrics(eval_pred, tokenizer, label_token_ids):
    logits, labels = eval_pred

    pred_labels = []
    true_labels = []

    for batch_idx in range(len(logits)):
        label_logits_array = logits[batch_idx]

        probs = torch.softmax(
            torch.tensor(label_logits_array, dtype=torch.float32), dim=0
        ).numpy()

        pred_label_idx = np.argmax(probs)
        pred_label = LABEL_LIST[pred_label_idx]
        pred_labels.append(LABEL_TO_ID[pred_label])

        batch_labels = labels[batch_idx]
        label_positions = np.where(batch_labels != -100)[0]

        if len(label_positions) == 0:
            logger.warning(
                f"Sample {batch_idx}: No valid label position found, using {NEI_LABEL} as default"
            )
            true_labels.append(LABEL_TO_ID[NEI_LABEL])
            continue
        valid_label_ids = batch_labels[label_positions]
        true_label_token = valid_label_ids[0]

        true_label = NEI_LABEL
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

    num_classes = len(LABEL_LIST)
    metrics = {
        "f1_macro": f1_score(
            true_labels, pred_labels, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        ),
        "precision_macro": precision_score(
            true_labels, pred_labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            true_labels, pred_labels, average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }
    per_class_f1 = f1_score(
        true_labels,
        pred_labels,
        average=None,
        zero_division=0,
        labels=list(range(num_classes)),
    )
    for idx, label_name in enumerate(LABEL_LIST):
        metrics[f"f1_{label_name.lower()}"] = (
            float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0
        )

    del pred_labels, true_labels, logits, labels

    return metrics


def _prepare_classification_dataset(
    claims: List[str],
    evidences: List[str],
    labels: List[str],
    tokenizer,
    max_length: int,
    prompt_template: str,
):
    targets = []

    for label in labels:
        target = _normalize_label(label)
        targets.append(target)

    if "{evidence}" not in prompt_template:
        raise ValueError("Prompt template must contain {evidence} placeholder")
    if "{claim}" not in prompt_template:
        raise ValueError("Prompt template must contain {claim} placeholder")

    template_start_raw, template_end_raw = prompt_template.split(
        "{evidence}", maxsplit=1
    )
    template_prefix_raw, template_claim_suffix_raw = template_start_raw.split(
        "{claim}", maxsplit=1
    )

    template_prefix_ids = tokenizer(
        template_prefix_raw, add_special_tokens=True, truncation=False
    )["input_ids"]
    template_claim_suffix_ids = tokenizer(
        template_claim_suffix_raw, add_special_tokens=False, truncation=False
    )["input_ids"]
    template_end_ids = tokenizer(
        template_end_raw, add_special_tokens=False, truncation=False
    )["input_ids"]

    truncation_stats = {"claim_truncated": 0, "evidence_truncated": 0}

    def tokenize_function(examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for claim, evidence_raw, target in zip(
            examples["claim"], examples["evidence"], examples["target"]
        ):
            claim_ids = tokenizer(
                str(claim), add_special_tokens=False, truncation=False
            )["input_ids"]

            target_token_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if len(target_token_ids) != 1:
                raise ValueError(
                    f"Training label '{target}' must be exactly 1 token, got {len(target_token_ids)} tokens: {target_token_ids}. "
                    "Update LABEL_LIST in src/config.py to single-token labels for this tokenizer."
                )
            target_ids = list(target_token_ids)
            if tokenizer.eos_token_id is not None:
                target_ids = target_ids + [tokenizer.eos_token_id]

            fixed_without_claim_and_evidence = (
                len(template_prefix_ids)
                + len(template_claim_suffix_ids)
                + len(template_end_ids)
                + len(target_ids)
            )
            available_for_claim = max_length - fixed_without_claim_and_evidence

            if available_for_claim < 0:
                raise ValueError(
                    "Prompt template is too long for max_length after reserving label token. "
                    f"Increase max_length (current={max_length})."
                )

            if len(claim_ids) > available_for_claim:
                claim_ids = claim_ids[:available_for_claim]
                truncation_stats["claim_truncated"] += 1

            start_ids = template_prefix_ids + claim_ids + template_claim_suffix_ids

            fixed_len = len(start_ids) + len(template_end_ids) + len(target_ids)
            available_for_evidence = max_length - fixed_len

            evidence_ids = []
            if available_for_evidence > 0:
                evidence_str = str(evidence_raw)
                if "|||" in evidence_str:
                    evidence_items = evidence_str.split("|||")
                else:
                    evidence_items = evidence_str.split("\n")

                current_evidence_ids = []
                for i, item in enumerate(evidence_items):
                    if not item.strip():
                        continue

                    formatted_item = f"{i + 1}. {item.strip()}\n"
                    item_ids = tokenizer(formatted_item, add_special_tokens=False)[
                        "input_ids"
                    ]

                    if (
                        len(current_evidence_ids) + len(item_ids)
                        <= available_for_evidence
                    ):
                        current_evidence_ids.extend(item_ids)
                    else:
                        truncation_stats["evidence_truncated"] += 1
                        break

                evidence_ids = current_evidence_ids

            full_input_ids = start_ids + evidence_ids + template_end_ids + target_ids

            prompt_len = len(start_ids) + len(evidence_ids) + len(template_end_ids)
            label_token_count = len(target_token_ids)
            labels = (
                [-100] * prompt_len
                + target_ids[:label_token_count]
                + [-100] * (len(target_ids) - label_token_count)
            )

            if not any(token_id != -100 for token_id in labels):
                raise ValueError(
                    "A sample ended up without supervised label tokens. "
                    "Increase max_length or review prompt truncation logic."
                )

            padding_length = max_length - len(full_input_ids)
            if padding_length > 0:
                unpadded_len = len(full_input_ids)
                pad_token_id = (
                    tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                )
                full_input_ids = full_input_ids + [pad_token_id] * padding_length
                labels = labels + [-100] * padding_length
                attention_mask = [1] * unpadded_len + [0] * padding_length
            else:
                full_input_ids = full_input_ids[:max_length]
                labels = labels[:max_length]
                attention_mask = [1] * max_length

            model_inputs["input_ids"].append(full_input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)

        return model_inputs

    dataset = Dataset.from_dict(
        {"claim": claims, "evidence": evidences, "target": targets}
    )

    tokenized = dataset.map(
        tokenize_function, batched=True, remove_columns=["claim", "evidence", "target"]
    )

    if truncation_stats["claim_truncated"] > 0:
        logger.warning(
            f"Claim truncation applied to {truncation_stats['claim_truncated']} samples "
            f"to preserve supervision labels within max_length={max_length}."
        )
    if truncation_stats["evidence_truncated"] > 0:
        logger.info(
            f"Evidence truncation applied to {truncation_stats['evidence_truncated']} samples "
            f"for max_length={max_length}."
        )

    missing_supervision = sum(
        1
        for sample_labels in tokenized["labels"]
        if not any(x != -100 for x in sample_labels)
    )
    if missing_supervision > 0:
        raise ValueError(
            f"Tokenized dataset contains {missing_supervision} samples with no supervised label tokens. "
            "Increase max_length or review prompt template."
        )

    return tokenized


def train_lora_classification(
    claims: List[str],
    evidences: List[str],
    labels: List[str],
    eval_claims: Optional[List[str]] = None,
    eval_evidences: Optional[List[str]] = None,
    eval_labels: Optional[List[str]] = None,
    config: Optional[LoRATrainingConfig] = None,
    gradient_accumulation_steps: int = 1,
    skip_final_eval: bool = True,
    checkpoint_path: Optional[str] = None,
) -> str:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch/transformers/peft/datasets are required for LoRA training."
        )

    config = config or LoRATrainingConfig()
    train_dtype, use_bf16, use_fp16, resolved_precision = _resolve_training_precision(
        config.precision
    )
    logger.info(
        f"Training precision resolved to {resolved_precision} (dtype={train_dtype})."
    )
    _warn_if_model_label_mismatch(config.model_name)

    _use_tf32 = False
    if torch.cuda.is_available():
        _cc_major, _ = torch.cuda.get_device_capability()
        if _cc_major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            _use_tf32 = True

    attn_impl = (
        "flash_attention_2"
        if getattr(config, "use_flash_attention", True)
        else ("sdpa" if config.use_sdpa else "eager")
    )
    logger.info(
        f"🚀 Speed optimisations: Target Attn={attn_impl}, "
        f"TF32={_use_tf32}, 8-bit AdamW, group_by_length=True"
    )

    import os

    from peft import PeftModel

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"🔄 Resuming training from checkpoint: {checkpoint_path}")

        adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.exists(adapter_config):
            raise ValueError(
                f"Invalid checkpoint: missing adapter_config.json in {checkpoint_path}"
            )

        logger.info("Loading tokenizer from base model...")
        tokenizer = _load_tokenizer_for_training(config.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading base model: {config.model_name}")
        base_model = _load_causal_lm_for_training(
            config.model_name, train_dtype, attn_implementation=attn_impl
        )

        base_model.config.use_cache = (
            False  # Must be False during training for correct gradients
        )

        logger.info("Loading LoRA adapter from checkpoint...")
        model = PeftModel.from_pretrained(
            base_model, checkpoint_path, is_trainable=True
        )
        model.print_trainable_parameters()

    else:
        if checkpoint_path:
            logger.warning(
                f"⚠️  Checkpoint path provided but not found: {checkpoint_path}"
            )
            logger.info("Creating new LoRA model from scratch...")
        else:
            logger.info(f"Creating new LoRA model from {config.model_name}")

        tokenizer = _load_tokenizer_for_training(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(
            f"Using existing vocabulary tokens for 3-class labels: {POSITIVE_LABEL}/{NEGATIVE_LABEL}/{NEI_LABEL}"
        )

        model = _load_causal_lm_for_training(
            config.model_name, train_dtype, attn_implementation=attn_impl
        )

        model.config.use_cache = (
            False  # Must be False during training for correct gradients
        )

        target_modules = _resolve_lora_target_modules(model)
        logger.info(
            f"Detected LoRA target modules for {config.model_name}: {target_modules}"
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )

        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if not (len(claims) == len(evidences) == len(labels)):
        raise ValueError(
            "Training inputs must have the same length: claims, evidences, labels."
        )
    if len(claims) == 0:
        raise ValueError("Training dataset is empty.")

    has_explicit_eval = any(
        x is not None for x in (eval_claims, eval_evidences, eval_labels)
    )

    if has_explicit_eval:
        if eval_claims is None or eval_evidences is None or eval_labels is None:
            raise ValueError(
                "When providing eval data, please provide all of: "
                "eval_claims, eval_evidences, eval_labels."
            )
        if not (len(eval_claims) == len(eval_evidences) == len(eval_labels)):
            raise ValueError(
                "Eval inputs must have the same length: "
                "eval_claims, eval_evidences, eval_labels."
            )
        if len(eval_claims) == 0:
            raise ValueError("Eval dataset is empty.")

        _log_label_distribution(labels, "TRAIN (raw)")
        _log_label_distribution(eval_labels, "EVAL (raw)")

        logger.info(f"Preparing TRAIN dataset with {len(claims)} samples...")
        train_dataset = _prepare_classification_dataset(
            claims,
            evidences,
            labels,
            tokenizer,
            config.max_length,
            config.prompt_template,
        )
        logger.info(f"Preparing EVAL dataset with {len(eval_claims)} samples...")
        eval_dataset = _prepare_classification_dataset(
            eval_claims,
            eval_evidences,
            eval_labels,
            tokenizer,
            config.max_length,
            config.prompt_template,
        )
        logger.info("Using provided DEV/EVAL dataset (no automatic train/eval split).")
    else:
        logger.warning(
            "No explicit DEV/EVAL dataset provided; falling back to split from TRAIN."
        )
        _log_label_distribution(labels, "TRAIN (raw)")
        logger.info(f"Preparing dataset with {len(claims)} samples...")
        full_dataset = _prepare_classification_dataset(
            claims,
            evidences,
            labels,
            tokenizer,
            config.max_length,
            config.prompt_template,
        )
        split_dataset = full_dataset.train_test_split(
            test_size=config.eval_ratio, seed=42, shuffle=True
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    logger.info(
        f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}"
    )

    _optim = "adamw_torch_fused"
    logger.info(f"Optimizer: {_optim}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=_use_tf32,
        report_to="none",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim=_optim,
        group_by_length=True,
        logging_nan_inf_filter=False,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_strategy="steps",
        eval_accumulation_steps=64,
        max_grad_norm=1.0,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )

    from transformers import default_data_collator

    data_collator = default_data_collator

    label_token_ids = _get_label_token_ids(tokenizer)

    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, tokenizer, label_token_ids)

    def preprocess_logits_for_metrics(logits, labels):
        """Extract logits at the position that predicts the label token.

        In causal LM, logits[i, t, :] predicts token at position t+1.
        So to predict label at position `label_pos`, we use logits at `label_pos - 1`.
        """
        num_labels = len(LABEL_LIST)
        batch_size = logits.shape[0]

        label_logits_batch = torch.zeros(
            (batch_size, num_labels), device=logits.device, dtype=logits.dtype
        )

        label_token_id_list = [label_token_ids[label] for label in LABEL_LIST]
        skipped = 0

        for i in range(batch_size):
            label_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]

            if len(label_positions) == 0:
                skipped += 1
                continue

            label_pos = label_positions[0].item()
            pred_pos = label_pos - 1

            if pred_pos < 0:
                skipped += 1
                continue

            if pred_pos >= logits.shape[1]:
                skipped += 1
                continue

            for j, token_id in enumerate(label_token_id_list):
                label_logits_batch[i, j] = logits[i, pred_pos, token_id]

        if skipped > 0:
            logger.warning(
                f"preprocess_logits_for_metrics: skipped {skipped}/{batch_size} samples (no valid label position)"
            )

        return label_logits_batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            ),
        ],
    )

    logger.info("Starting LoRA fine-tuning with F1 optimization...")

    trainer.train()

    if not skip_final_eval:
        logger.info("Running final evaluation...")
        final_metrics = trainer.evaluate()
        logger.info(
            f"Final metrics: F1={final_metrics.get('eval_f1_macro', 0):.4f}, "
            f"Precision={final_metrics.get('eval_precision_macro', 0):.4f}, "
            f"Recall={final_metrics.get('eval_recall_macro', 0):.4f}, "
            f"Accuracy={final_metrics.get('eval_accuracy', 0):.4f}"
        )

    else:
        logger.info("⚠️  Skipping final evaluation to prevent CUDA OOM")
        logger.info(
            "💡 You already have eval metrics from training (check logs above for eval_f1_macro)"
        )
        final_metrics = {}

    best_model_dir = config.output_dir
    # Re-enable cache for inference after training
    model.config.use_cache = True
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    best_f1 = final_metrics.get("eval_f1_macro")
    if best_f1 is None:
        best_f1 = trainer.state.best_metric
    best_ckpt = trainer.state.best_model_checkpoint
    logger.info(f"✅ Best model (F1={best_f1 or 0:.4f}) saved to {best_model_dir}")
    if best_ckpt:
        logger.info(f"✅ Best checkpoint: {best_ckpt}")

    logger.info(
        f"ℹ️  Intermediate checkpoints (e.g., checkpoint-XXX) are saved in {config.output_dir}"
    )
    logger.info("ℹ️  To resume training, use the latest checkpoint folder found there.")

    return config.output_dir
