from typing import Any, List, Optional

from loguru import logger

from .config import LABEL_LIST, PROMPT_TEMPLATE

try:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch/transformers not available. LLMScorer cannot run.")
    torch = None  # type: ignore


# Labels are already words (True/False/Not) - no mapping needed


class LLMScorer:
    """
    Get LM logits for labels using a prompt.
    Uses existing vocabulary tokens: True, False.
    Returns LOGITS, not probabilities (for fusion layer per Eq.2).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 2048,
        labels: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for LLMScorer.")

        self.device = device
        self.max_length = max_length
        # Check if model_name is a LoRA adapter
        import os

        is_lora = os.path.exists(os.path.join(model_name, "adapter_config.json"))

        if is_lora:
            logger.info(
                f"Detected LoRA adapter at {model_name}. Loading base model + adapter..."
            )
            config = PeftConfig.from_pretrained(model_name)
            base_model_path = config.base_model_name_or_path

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, trust_remote_code=True
            )

            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            # Load adapter
            self.model = PeftModel.from_pretrained(self.model, model_name)
        else:
            logger.info(f"Loading standard model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Match training's padding side
        # Training does manual RIGHT padding (append PAD tokens to the right)
        # Must set explicitly as some models default to left padding
        self.tokenizer.padding_side = "right"

        self.model.eval()

        self.labels = labels or LABEL_LIST

        # Get token IDs for labels (already words: True/False)
        self.label_token_ids = {}
        for label in self.labels:
            # Use label directly as it's already a word
            tokens = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(tokens) == 0:
                raise ValueError(f"Label '{label}' tokenized to 0 tokens!")
            self.label_token_ids[label] = tokens[0]
            logger.debug(f"Label '{label}' -> token_id {tokens[0]}")

        self.prompt_template = prompt_template or PROMPT_TEMPLATE
        logger.info(f"LLMScorer initialized with model: {model_name}")
        logger.info(f"Label token IDs: {self.label_token_ids}")

    def _build_prompt(self, text: str, evidence: str = "") -> str:
        return self.prompt_template.format(claim=text, evidence=evidence)

    def score_logits(
        self, texts: List[str], evidences: Optional[List[Any]] = None
    ) -> torch.Tensor:
        """
        Returns LOGITS (not probabilities) for labels in shape [batch, num_labels].
        This is the paper-accurate p_LM(y|q) for Eq.2.

        Implements SMART TRUNCATION:
        - Preserves Claim and Template structure
        - Formats evidence as numbered list: "1. ...\n2. ..."
        - Truncates evidence from bottom to fill max_length
        """
        if evidences is None:
            evidences = [[]] * len(texts)

        # Prepare batch inputs manually to handle smart truncation
        batch_input_ids = []
        batch_attention_mask = []

        # Split template into start (before evidence) and end (after evidence)
        if "{evidence}" not in self.prompt_template:
            raise ValueError("Prompt template must contain {evidence} placeholder")
        template_parts = self.prompt_template.split("{evidence}")
        template_start_raw = template_parts[0]
        template_end_raw = template_parts[1]

        # CRITICAL: Match training's max_length exactly (no reserved tokens)
        # Training: input = [BOS, prompt, label, EOS] fits in max_length
        # Inference: input = [BOS, prompt] should use same available space
        # Prediction position: last non-pad token predicts next token (the label)

        for text, evidence_item in zip(texts, evidences):
            # 1. Prepare fixed parts
            template_start = template_start_raw.format(claim=text)
            template_end = template_end_raw

            # Tokenize fixed parts
            start_ids = self.tokenizer(
                template_start, add_special_tokens=True, truncation=False
            )["input_ids"]
            end_ids = self.tokenizer(
                template_end, add_special_tokens=False, truncation=False
            )["input_ids"]

            # 2. Calculate available space for evidence (match training)
            fixed_len = len(start_ids) + len(end_ids)
            available_for_evidence = self.max_length - fixed_len

            # 3. Process evidence
            evidence_ids = []
            if available_for_evidence > 0:
                # Handle both List[str] (from retriever) and str (from CSV)
                if isinstance(evidence_item, list):
                    evidence_list = evidence_item
                elif isinstance(evidence_item, str):
                    # Support both newline separation (training) and ||| separation (CSV loader)
                    if "|||" in evidence_item:
                        evidence_list = evidence_item.split("|||")
                    else:
                        evidence_list = evidence_item.split("\n")
                else:
                    evidence_list = []

                current_evidence_ids = []
                for i, item in enumerate(evidence_list):
                    if not str(item).strip():
                        continue

                    # Format: "1. Evidence text\n"
                    formatted_item = f"{i + 1}. {str(item).strip()}\n"
                    item_ids = self.tokenizer(formatted_item, add_special_tokens=False)[
                        "input_ids"
                    ]

                    if (
                        len(current_evidence_ids) + len(item_ids)
                        <= available_for_evidence
                    ):
                        current_evidence_ids.extend(item_ids)
                    else:
                        # CRITICAL: Don't truncate mid-token - skip item entirely (match lora_trainer.py)
                        # Truncating evidence mid-word corrupts semantic context for fact-checking
                        break

                evidence_ids = current_evidence_ids

            # 4. Construct full input
            full_input_ids = start_ids + evidence_ids + end_ids

            # Truncate if still too long (shouldn't happen with logic above but safety first)
            if len(full_input_ids) > self.max_length:
                full_input_ids = full_input_ids[: self.max_length]

            batch_input_ids.append(full_input_ids)
            batch_attention_mask.append([1] * len(full_input_ids))

        # Pad batch
        max_batch_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_mask = []

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        for ids, mask in zip(batch_input_ids, batch_attention_mask):
            padding = [pad_id] * (max_batch_len - len(ids))
            mask_padding = [0] * (max_batch_len - len(ids))

            # Right padding (match training)
            padded_input_ids.append(ids + padding)
            padded_attention_mask.append(mask + mask_padding)

        input_tensor = torch.tensor(
            padded_input_ids, dtype=torch.long, device=self.device
        )
        mask_tensor = torch.tensor(
            padded_attention_mask, dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, attention_mask=mask_tensor)
            # outputs.logits: [batch, seq_len, vocab_size]

            # Find the position where we predict the label token
            # This is the last non-pad position
            seq_lengths = mask_tensor.sum(dim=1)
            batch_idx = torch.arange(input_tensor.size(0), device=self.device)
            pred_pos = seq_lengths - 1

            pred_logits = outputs.logits[batch_idx, pred_pos, :]  # [batch, vocab_size]

        # Extract logits only for label tokens
        label_ids = [self.label_token_ids[label] for label in self.labels]
        label_logits = torch.stack([pred_logits[:, idx] for idx in label_ids], dim=-1)

        return label_logits  # [batch, num_labels] - RAW LOGITS

    def score_probs(
        self, texts: List[str], evidences: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Returns softmax PROBABILITIES for labels in shape [batch, num_labels].
        Use this for inference, not for fusion training.
        """
        logits = self.score_logits(texts, evidences)
        return torch.softmax(logits, dim=-1)

    def score_single(self, text: str, evidence: str = "") -> torch.Tensor:
        """Score a single text, returns probabilities for each label."""
        return self.score_probs([text], [evidence])[0]
