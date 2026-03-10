import gc
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
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    np = None  # type: ignore
    torch = None  # type: ignore
    Dataset = None  # type: ignore
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore
    get_peft_model = None  # type: ignore
    accuracy_score = None  # type: ignore
    f1_score = None  # type: ignore
    precision_score = None  # type: ignore
    recall_score = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore
    DataCollatorForSeq2Seq = None  # type: ignore
    EarlyStoppingCallback = None  # type: ignore
    Trainer = None  # type: ignore
    TrainingArguments = None  # type: ignore
    TrainerCallback = object  # type: ignore[misc,assignment]

from .config import LABEL_LIST, LABEL_TO_ID, PROMPT_TEMPLATE

POSITIVE_LABEL = LABEL_LIST[0]
NEGATIVE_LABEL = LABEL_LIST[1]


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
    eval_ratio: float = 0.1  # Used only when no explicit eval dataset is provided
    early_stopping_patience: int = 3  # Stop if F1 doesn't improve for 3 evals
    precision: str = "auto"  # auto -> bf16 (if supported) else fp16; cpu uses fp32

    # Prompt template for classification
    prompt_template: str = PROMPT_TEMPLATE


def _load_tokenizer_for_training(model_name: str):
    """Load tokenizer with remote code disabled by default.

    Note: use_fast=True (default) is used because newer versions of transformers
    removed slow tokenizer classes for some model families (e.g. BloomTokenizer),
    which causes a ValueError when use_fast=False is forced.
    """
    try:
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False, use_fast=True
        )
    except Exception:
        # Last-resort fallback: let transformers pick the best available tokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)


def _load_causal_lm_for_training(
    model_name: str,
    torch_dtype,
    use_4bit_quant: bool = False,
):
    """Load CausalLM with remote code disabled by default.

    Args:
        model_name: HuggingFace model ID or local path.
        torch_dtype: Compute dtype (e.g. torch.bfloat16).
        use_4bit_quant: If True, load in 4-bit QLoRA mode via BitsAndBytes.
            Reduces base model VRAM from ~20 GB (fp16 LoRA) to ~7-10 GB.
    """
    if use_4bit_quant and torch.cuda.is_available():
        # ── Optimization 3: 4-bit QLoRA quantization (BitsAndBytes) ──
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # nested quant → extra ~0.4 bit/param
        )
        logger.info("Loading model in 4-bit QLoRA mode (BitsAndBytes NF4).")
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )


def _resolve_training_precision(precision: str):
    """
    Resolve compute precision for training/evaluation.

    Returns:
        tuple(torch_dtype, use_bf16, use_fp16, resolved_name)
    """
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
    """Return True for linear-like modules supported by common PEFT backends."""
    if isinstance(module, torch.nn.Linear):
        return True
    return "linear" in module.__class__.__name__.lower()


def _resolve_lora_target_modules(model) -> List[str]:
    """
    Resolve LoRA target modules from the loaded model architecture.

    This avoids hardcoding Llama-only names (q_proj/k_proj/v_proj/o_proj),
    which breaks on models like GPT/PhoGPT/GPT-NeoX variants.
    """
    module_suffixes = {name.split(".")[-1] for name, _ in model.named_modules() if name}

    # Preferred, architecture-aware suffixes (checked in this order).
    preferred_suffixes = [
        # Llama / Mistral / Qwen
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # MLP projections (often useful in LoRA SFT)
        "gate_proj",
        "up_proj",
        "down_proj",
        # GPT-2 / Falcon style
        "c_attn",
        "c_proj",
        "c_fc",
        # GPT-NeoX / Phi style
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
        # Other common naming patterns
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

    # Generic fallback: use all linear-like layer suffixes except known output heads.
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

    # Labels are provided directly as output tokens (e.g., Đúng/Sai).
    # For logits-based classification here, each label MUST be exactly one token.
    label_token_ids = {}
    for label in labels:
        # Use label directly as it's already a word/token candidate
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
    """
    Compute F1, Precision, Recall, Accuracy for binary classification evaluation.

    PAPER-ACCURATE: Extracts pLM(y|q) from logits, not from text generation.

    Args:
        eval_pred: Tuple of (predictions, labels)
            - predictions: preprocessed logits with shape [batch, 2] (label logits only)
            - labels: label IDs with shape [batch, seq_len], -100 for masked positions
        tokenizer: Tokenizer to decode labels
        label_token_ids: Dict mapping label names to their token IDs

    Returns:
        Dict of metrics (F1, precision, recall, accuracy)
    """
    logits, labels = eval_pred

    # logits shape: [batch, 2] - already preprocessed to label logits only!
    # labels shape: [batch, seq_len]

    pred_labels = []
    true_labels = []

    for batch_idx in range(len(logits)):
        # Logits are already extracted for label tokens: [2] = [positive, negative]
        label_logits_array = logits[batch_idx]  # Shape: [2]

        # Apply softmax to get probabilities (pLM)
        exp_logits = np.exp(
            label_logits_array - np.max(label_logits_array)
        )  # numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Choose label with highest probability
        pred_label_idx = np.argmax(probs)
        pred_label = LABEL_LIST[pred_label_idx]
        pred_labels.append(LABEL_TO_ID[pred_label])

        # Extract true label from labels
        batch_labels = labels[batch_idx]
        label_positions = np.where(batch_labels != -100)[0]

        if len(label_positions) == 0:
            logger.warning(
                f"Sample {batch_idx}: No valid label position found, using {NEGATIVE_LABEL} as default"
            )
            true_labels.append(LABEL_TO_ID[NEGATIVE_LABEL])
            continue
        valid_label_ids = batch_labels[label_positions]
        true_label_token = valid_label_ids[0]  # First token of label

        # Map back to label name
        true_label = NEGATIVE_LABEL  # default
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

    # Compute metrics (binary classification)
    metrics = {
        "f1_macro": f1_score(
            true_labels, pred_labels, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        ),
        "f1_binary": f1_score(
            true_labels, pred_labels, average="binary", pos_label=1, zero_division=0
        ),  # negative/refuted class as positive
        "precision_macro": precision_score(
            true_labels, pred_labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            true_labels, pred_labels, average="macro", zero_division=0
        ),
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
    prompt_template: str,
):
    """
    Prepare dataset for supervised classification fine-tuning.
    Format: prompt + label (causal LM style)
    Implements smart truncation: preserves claim/template, truncates evidence from bottom.
    """
    targets = []

    def normalize_label(label_value) -> str:
        """Convert CSV_Loader integer ID to string label for LLM training.

        CSV_Loader Convention:
          - ID 0 = supported/legitimate
          - ID 1 = refuted/fake

        This function converts those IDs to strings for LLM training targets.
        """
        if isinstance(label_value, (int, float)):
            idx = int(label_value)
            # CSV_Loader outputs: 0=positive label, 1=negative label
            if idx == 0:
                return POSITIVE_LABEL
            else:
                return NEGATIVE_LABEL

        # Handle string labels from multiple datasets/languages.
        label_upper = str(label_value).upper().strip()
        if label_upper in [
            POSITIVE_LABEL.upper(),
            "TRUE",
            "SUPPORTED",
            "LEGIT",
            "LEGITIMATE",
            "ĐÚNG",
            "DUNG",
            "0",
        ]:
            return POSITIVE_LABEL
        if label_upper in [
            NEGATIVE_LABEL.upper(),
            "FALSE",
            "REFUTED",
            "SCAM",
            "SAI",
            "1",
        ]:
            return NEGATIVE_LABEL

        logger.warning(
            f"Unknown label '{label_value}' encountered during dataset prep. Defaulting to {NEGATIVE_LABEL}."
        )
        return NEGATIVE_LABEL

    for label in labels:
        target = normalize_label(label)
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

    # Tokenize static template chunks once for efficiency and consistent budgeting.
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

    # Tokenize
    def tokenize_function(examples):
        """
        Build input_ids with smart truncation and numbered evidence.

        Priority order for fitting sequence into max_length:
        1) Always keep label token supervision.
        2) Keep prompt template.
        3) Truncate evidence first.
        4) Truncate claim only when claim+template already exceeds budget.
        """
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for claim, evidence_raw, target in zip(
            examples["claim"], examples["evidence"], examples["target"]
        ):
            claim_ids = tokenizer(
                str(claim), add_special_tokens=False, truncation=False
            )["input_ids"]

            # Tokenize target label token (must be exactly one token).
            target_token_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if len(target_token_ids) != 1:
                raise ValueError(
                    f"Training label '{target}' must be exactly 1 token, got {len(target_token_ids)} tokens: {target_token_ids}. "
                    "Update LABEL_LIST in src/config.py to single-token labels for this tokenizer."
                )
            target_ids = list(target_token_ids)
            if tokenizer.eos_token_id is not None:
                target_ids = target_ids + [tokenizer.eos_token_id]

            # Reserve space for template ending + target first to guarantee supervision.
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

            # If claim alone overflows, truncate claim tail but keep template + label.
            if len(claim_ids) > available_for_claim:
                claim_ids = claim_ids[:available_for_claim]
                truncation_stats["claim_truncated"] += 1

            start_ids = template_prefix_ids + claim_ids + template_claim_suffix_ids

            # Remaining budget goes to evidence.
            fixed_len = len(start_ids) + len(template_end_ids) + len(target_ids)
            available_for_evidence = max_length - fixed_len

            evidence_ids = []
            if available_for_evidence > 0:
                # Support both newline separation (old) and ||| separation (CSV loader)
                evidence_str = str(evidence_raw)
                if "|||" in evidence_str:
                    evidence_items = evidence_str.split("|||")
                else:
                    evidence_items = evidence_str.split("\n")

                current_evidence_ids = []
                for i, item in enumerate(evidence_items):
                    if not item.strip():
                        continue

                    # Format: "1. Evidence text\n"
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
                        # Keep complete evidence items only (no mid-item truncation).
                        truncation_stats["evidence_truncated"] += 1
                        break

                evidence_ids = current_evidence_ids

            # Construct full input
            full_input_ids = start_ids + evidence_ids + template_end_ids + target_ids

            # Create labels: mask prompt, supervise label token only.
            prompt_len = len(start_ids) + len(evidence_ids) + len(template_end_ids)
            label_token_count = len(target_token_ids)
            labels = (
                [-100] * prompt_len
                + target_ids[:label_token_count]
                + [-100] * (len(target_ids) - label_token_count)
            )

            # Safety guard: supervision token must remain after preprocessing.
            if not any(token_id != -100 for token_id in labels):
                raise ValueError(
                    "A sample ended up without supervised label tokens. "
                    "Increase max_length or review prompt truncation logic."
                )

            # Pad to max_length
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
                # Defensive fallback if sequence sizing ever regresses.
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


class MemoryCleanupCallback(TrainerCallback):
    """
    Callback to aggressively clean up memory after evaluation to prevent OOM.
    """

    def on_evaluate(self, args, state, control, **kwargs):
        """Clean up memory after evaluation."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"GPU Memory: {torch.cuda.memory_allocated() / 2048**3:.2f}GB / {torch.cuda.max_memory_allocated() / 2048**3:.2f}GB"
            )

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
    eval_claims: Optional[List[str]] = None,
    eval_evidences: Optional[List[str]] = None,
    eval_labels: Optional[List[str]] = None,
    config: Optional[LoRATrainingConfig] = None,
    # ── Optimization 2: gradient_accumulation_steps=2 with batch_size=1 ─────────
    # Effective batch = batch_size × gradient_accumulation_steps = 1×2 = 2.
    # VRAM is the same as batch_size=1 (only one micro-batch lives on GPU at a time)
    # while gradient quality matches batch_size=2.  Previously this was 4 which
    # means each optimizer step needed 4× forward passes; 2 is a better default
    # for seq=2048 workloads.
    gradient_accumulation_steps: int = 2,
    skip_final_eval: bool = True,  # Skip final eval by default to prevent OOM
    # ── Optimization 3: enable 4-bit QLoRA ──────────────────────────────────────
    # Set use_4bit_quant=True to load the base model in 4-bit (NF4) mode.
    # This drops base-model VRAM from ~20 GB (fp16 LoRA) to ~7-10 GB.
    # Adapter weights are still trained in bf16/fp16.
    use_4bit_quant: bool = False,
    checkpoint_path: Optional[
        str
    ] = None,  # Path to existing checkpoint to resume training
) -> str:
    """
    Train LLM with LoRA for classification task.

    Args:
        claims: List of claims to verify
        evidences: List of retrieved evidence for each claim
        labels: List of ground truth labels (ID or dataset variants)
        eval_claims: Optional list of eval/dev claims. If provided with eval_evidences
            and eval_labels, trainer uses this set directly (no auto split).
        eval_evidences: Optional list of eval/dev evidence.
        eval_labels: Optional list of eval/dev labels.
        config: Training configuration
        gradient_accumulation_steps: Number of gradient accumulation steps
        skip_final_eval: Skip final evaluation to save memory
        checkpoint_path: Path to existing LoRA checkpoint to resume training (optional)

    Returns:
        Path to saved LoRA model
    """
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

    # Check if resuming from checkpoint
    import os

    from peft import PeftModel

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"🔄 Resuming training from checkpoint: {checkpoint_path}")

        # Check if checkpoint has adapter files
        adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.exists(adapter_config):
            raise ValueError(
                f"Invalid checkpoint: missing adapter_config.json in {checkpoint_path}"
            )

        # Always load tokenizer from base model to avoid Peft corrupted tokenizer_config.json issues with Qwen
        logger.info("Loading tokenizer from base model...")
        tokenizer = _load_tokenizer_for_training(config.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # NO special tokens needed - using existing vocabulary labels

        # Load base model
        logger.info(f"Loading base model: {config.model_name}")
        base_model = _load_causal_lm_for_training(
            config.model_name, train_dtype, use_4bit_quant=use_4bit_quant
        )

        # ── Optimization 4: gradient checkpointing (use_reentrant=False) ──
        # use_reentrant=False avoids extra activation copies → saves ~10-15% VRAM.
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # FIXED: Disable cache when using gradient checkpointing
        base_model.config.use_cache = False

        # Load LoRA adapter from checkpoint
        logger.info("Loading LoRA adapter from checkpoint...")
        model = PeftModel.from_pretrained(
            base_model, checkpoint_path, is_trainable=True
        )
        model.print_trainable_parameters()

    else:
        # Create new model from scratch
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

        # NO special tokens needed - using existing vocabulary labels
        logger.info(
            f"Using existing vocabulary tokens for binary labels: {POSITIVE_LABEL}/{NEGATIVE_LABEL}"
        )

        model = _load_causal_lm_for_training(
            config.model_name, train_dtype, use_4bit_quant=use_4bit_quant
        )

        # ── Optimization 4: gradient checkpointing (use_reentrant=False) ──
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # FIXED: Disable cache when using gradient checkpointing
        model.config.use_cache = False

        # Configure LoRA
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
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_nan_inf_filter=False,
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
        Reduce full logits [batch, seq_len, vocab_size] to label logits [batch, 2]
        BEFORE accumulation to save massive amounts of memory.

        For each sample, we extract:
        - Find label position from labels
        - Apply CausalLM shift (pred_pos = label_pos - 1)
        - Extract logits for 2 label tokens only (positive/negative)

        Memory savings: ~65MB/sample → ~100 bytes/sample (650x reduction!)
        """
        # logits: [batch, seq_len, vocab_size]
        # labels: [batch, seq_len]
        batch_size = logits.shape[0]

        # Pre-allocate for label logits only [batch, 2] for binary classification
        label_logits_batch = torch.zeros(
            (batch_size, 2), device=logits.device, dtype=logits.dtype
        )

        # Extract token IDs for labels in LABEL_LIST order
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
            # This is the key: extract ONLY 2 values instead of entire vocab
            for j, token_id in enumerate(label_token_id_list):
                label_logits_batch[i, j] = logits[i, pred_pos, token_id]

        # Return reduced logits [batch, 2] instead of [batch, seq_len, vocab_size]
        # This is ~650x smaller for binary classification!
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
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            ),
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
        logger.info(
            f"Final metrics: F1={final_metrics.get('eval_f1_macro', 0):.4f}, "
            f"Precision={final_metrics.get('eval_precision_macro', 0):.4f}, "
            f"Recall={final_metrics.get('eval_recall_macro', 0):.4f}, "
            f"Accuracy={final_metrics.get('eval_accuracy', 0):.4f}"
        )

        # Clear memory after eval
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("⚠️  Skipping final evaluation to prevent CUDA OOM")
        logger.info(
            "💡 You already have eval metrics from training (check logs above for eval_f1_macro)"
        )
        final_metrics = {}

    # Save BEST model (according to F1 metric)
    best_model_dir = config.output_dir
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info(
        f"✅ Best model (F1={final_metrics.get('eval_f1_macro', 0):.4f}) saved to {best_model_dir}"
    )

    logger.info(
        f"ℹ️  Intermediate checkpoints (e.g., checkpoint-XXX) are saved in {config.output_dir}"
    )
    logger.info("ℹ️  To resume training, use the latest checkpoint folder found there.")

    return config.output_dir
