"""
LLM Scorer

Computes p_LM(y|q) for scam vs legitimate using a causal LLM.
This is used in the fusion layer training and inference.
"""

from typing import List, Tuple, Any, Optional
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


class LLMScorer:
    """
    Get LM logits for labels using a prompt.

    Default labels:
      - " scam"
      - " legitimate"
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
        labels: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for LLMScorer.")

        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
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
                device_map=self.device
            )
            # Load adapter
            self.model = PeftModel.from_pretrained(self.model, model_name)
        else:
            logger.info(f"Loading standard model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            
        self.model.eval()

        self.labels = labels or LABEL_LIST
        # Label tokens (use leading space for BPE tokenization)
        self.label_tokens = {label: f" {label}" for label in self.labels}
        self.label_ids = {
            k: self.tokenizer.encode(v, add_special_tokens=False)[0]
            for k, v in self.label_tokens.items()
        }

        self.prompt_template = prompt_template or PROMPT_TEMPLATE

        logger.info(f"LLMScorer initialized with model: {model_name}")

    def _build_prompt(self, text: str, evidence: str = "") -> str:
        return self.prompt_template.format(claim=text, evidence=evidence)

    def score_texts(self, texts: List[str], evidences: Optional[List[str]] = None) -> Any:
        """
        Returns logits for labels in shape [batch, num_labels].
        """
        if evidences is None:
            # If no evidence provided, pass empty string (though model expects evidence)
            evidences = [""] * len(texts)
            
        prompts = [self._build_prompt(t, e) for t, e in zip(texts, evidences)]
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Logits for last token
            last_logits = outputs.logits[:, -1, :]

        label_ids = [self.label_ids[label] for label in self.labels]
        logits = torch.stack([last_logits[:, idx] for idx in label_ids], dim=-1)

        return logits

