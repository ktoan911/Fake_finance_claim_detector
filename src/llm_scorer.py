"""
LLM Scorer - Paper-accurate implementation

Computes p_LM(y|q) logits for classification using a causal LLM.
Returns LOGITS (not probabilities) as per Eq.2: β·pLM + (1-β)·MLP(pret)
"""

from typing import List, Any, Optional
from loguru import logger
from .config import PROMPT_TEMPLATE, LABEL_LIST

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch/transformers not available. LLMScorer cannot run.")
    torch = None  # type: ignore


# Mapping from internal labels to vocab words (matching lora_trainer.py)
LABEL_TO_WORD = {
    "SUPPORTED": "True",
    "REFUTED": "False",
    "NEI": "Unsure",
}


class LLMScorer:
    """
    Get LM logits for labels using a prompt.
    Uses existing vocabulary tokens: True, False, Unsure.
    Returns LOGITS, not probabilities (for fusion layer per Eq.2).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 1024,
        labels: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for LLMScorer.")

        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # CRITICAL: Match training's padding side
        # Training does manual RIGHT padding (append PAD tokens to the right)
        # Must set explicitly as some models default to left padding
        self.tokenizer.padding_side = 'right'
        
        # Check if model_name is a LoRA adapter
        import os
        is_lora = os.path.exists(os.path.join(model_name, "adapter_config.json"))
        
        if is_lora:
            logger.info(f"Detected LoRA adapter at {model_name}. Loading base model + adapter...")
            config = PeftConfig.from_pretrained(model_name)
            base_model_path = config.base_model_name_or_path
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
                low_cpu_mem_usage=True
            )
            # Load adapter
            self.model = PeftModel.from_pretrained(self.model, model_name)
        else:
            logger.info(f"Loading standard model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
                low_cpu_mem_usage=True
            )
            
        self.model.eval()

        self.labels = labels or LABEL_LIST
        
        # Get token IDs for vocab words (True/False/Unsure)
        self.label_token_ids = {}
        for label in self.labels:
            word = LABEL_TO_WORD.get(label, "Unsure")
            tokens = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(tokens) == 0:
                raise ValueError(f"Word '{word}' tokenized to 0 tokens!")
            self.label_token_ids[label] = tokens[0]
            logger.debug(f"Label '{label}' -> '{word}' -> token_id {tokens[0]}")

        self.prompt_template = prompt_template or PROMPT_TEMPLATE
        logger.info(f"LLMScorer initialized with model: {model_name}")
        logger.info(f"Label token IDs: {self.label_token_ids}")

    def _build_prompt(self, text: str, evidence: str = "") -> str:
        return self.prompt_template.format(claim=text, evidence=evidence)

    def score_logits(self, texts: List[str], evidences: Optional[List[str]] = None) -> torch.Tensor:
        """
        Returns LOGITS (not probabilities) for labels in shape [batch, num_labels].
        This is the paper-accurate p_LM(y|q) for Eq.2.
        
        TRAINING ALIGNMENT:
        During training, sequence structure is: [BOS, prompt_tokens, label_token, EOS]
        - Labels mask prompt with -100, keep label_token, mask EOS
        - CausalLM predicts NEXT token, so at position (prompt_end), it predicts label_token
        
        During inference:
        - We need to extract logits from the position that predicts the label token
        - This is the second-to-last non-pad position (before EOS)
        - If no EOS, it's the last position
        """
        if evidences is None:
            evidences = [""] * len(texts)
            
        prompts = [self._build_prompt(t, e) for t, e in zip(texts, evidences)]
        
        # CRITICAL: Reserve space for label tokens to match training truncation
        # Training: truncates prompt to (max_length - target_len) where target_len ~= 2-3
        # Inference: must do the same to ensure sequence length consistency
        # Reserve 3 tokens: 1 for label (True/False/Unsure) + 1 safety + 1 for EOS
        reserved_for_label = 3
        effective_max_length = self.max_length - reserved_for_label
        
        inputs = self.tokenizer(
            prompts,
            add_special_tokens=True,  # Match training - add BOS token
            padding=True,
            truncation=True,
            max_length=effective_max_length,  # Truncate prompt, reserve space for label
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # outputs.logits: [batch, seq_len, vocab_size]
            
            # Find the position where we predict the label token
            # Training: [BOS, prompt_tokens, label_token, EOS]
            #           Model predicts label from position (last_prompt_position)
            # 
            # Inference: [BOS, prompt_tokens, PAD...]
            #            We need logits from last_prompt_position to predict next token (label)
            #            This is simply the last non-pad position (seq_lengths - 1)
            
            attn_mask = inputs["attention_mask"]  # [batch, seq_len]
            seq_lengths = attn_mask.sum(dim=1)    # [batch] - total non-pad tokens
            
            batch_idx = torch.arange(attn_mask.size(0), device=self.model.device)
            
            # The prediction position is the last non-pad token (last prompt token)
            # This position's logits predict the NEXT token, which would be the label
            pred_pos = seq_lengths - 1  # Last non-pad position
            
            pred_logits = outputs.logits[batch_idx, pred_pos, :]  # [batch, vocab_size]

        # Extract logits only for label tokens
        label_ids = [self.label_token_ids[label] for label in self.labels]
        label_logits = torch.stack([pred_logits[:, idx] for idx in label_ids], dim=-1)
        
        return label_logits  # [batch, num_labels] - RAW LOGITS

    def score_probs(self, texts: List[str], evidences: Optional[List[str]] = None) -> torch.Tensor:
        """
        Returns softmax PROBABILITIES for labels in shape [batch, num_labels].
        Use this for inference, not for fusion training.
        """
        logits = self.score_logits(texts, evidences)
        return torch.softmax(logits, dim=-1)

    def score_single(self, text: str, evidence: str = "") -> torch.Tensor:
        """Score a single text, returns probabilities for each label."""
        return self.score_probs([text], [evidence])[0]
