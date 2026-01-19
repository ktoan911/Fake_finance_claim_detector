"""
LLM Scorer

Computes p_LM(y|q) for scam vs legitimate using a causal LLM.
This is used in the fusion layer training and inference.
"""

from typing import List, Tuple, Any
from loguru import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
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
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.labels = labels or ["scam", "legitimate"]
        # Label tokens (use leading space for BPE tokenization)
        self.label_tokens = {label: f" {label}" for label in self.labels}
        self.label_ids = {
            k: self.tokenizer.encode(v, add_special_tokens=False)[0]
            for k, v in self.label_tokens.items()
        }

        self.prompt_template = prompt_template or (
            "Classify this reddit post as scam or legitimate.\n"
            "Post: {text}\n"
            "Answer:"
        )

        logger.info(f"LLMScorer initialized with model: {model_name}")

    def _build_prompt(self, text: str) -> str:
        return self.prompt_template.format(text=text)

    def score_texts(self, texts: List[str]) -> Any:
        """
        Returns logits for [legit, scam] in shape [batch, 2].
        """
        prompts = [self._build_prompt(t) for t in texts]
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

