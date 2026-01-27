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
    "NEI": "Not",
}


class LLMScorer:
    """
    Get LM logits for labels using a prompt.
    Uses existing vocabulary tokens: True, False, Not.
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
                device_map="auto",
                low_cpu_mem_usage=True
            )
            # Load adapter
            self.model = PeftModel.from_pretrained(self.model, model_name)
        else:
            logger.info(f"Loading standard model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
        self.model.eval()

        self.labels = labels or LABEL_LIST
        
        # Get token IDs for vocab words (True/False/Not)
        self.label_token_ids = {}
        for label in self.labels:
            word = LABEL_TO_WORD.get(label, "Not")
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

    def score_logits(self, texts: List[str], evidences: Optional[List[Any]] = None) -> torch.Tensor:
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
        
        # Reserve space for label tokens (True/False/Not) + EOS
        # Training uses: [BOS, prompt, label, EOS]
        # Inference needs logits at position of last prompt token to predict label
        # So we need to fit [BOS, prompt] into max_length - 1 (for label prediction space)
        # But wait, we want to match training exactly.
        # Training: input = [BOS, prompt, label, EOS] (length <= max_len)
        # Inference: input = [BOS, prompt] (length <= max_len - 1)
        # We reserve 2 tokens: 1 for potential label generation (though we don't generate), 1 for EOS safety
        reserved_tokens = 2 
        
        for text, evidence_item in zip(texts, evidences):
            # 1. Prepare fixed parts
            template_start = template_start_raw.format(claim=text)
            template_end = template_end_raw
            
            # Tokenize fixed parts
            start_ids = self.tokenizer(template_start, add_special_tokens=True, truncation=False)["input_ids"]
            end_ids = self.tokenizer(template_end, add_special_tokens=False, truncation=False)["input_ids"]
            
            # 2. Calculate available space for evidence
            fixed_len = len(start_ids) + len(end_ids)
            available_for_evidence = self.max_length - fixed_len - reserved_tokens
            
            # 3. Process evidence
            evidence_ids = []
            if available_for_evidence > 0:
                # Handle both List[str] (from retriever) and str (from CSV with \n)
                if isinstance(evidence_item, list):
                    evidence_list = evidence_item
                elif isinstance(evidence_item, str):
                    evidence_list = evidence_item.split('\n')
                else:
                    evidence_list = []
                
                current_evidence_ids = []
                for i, item in enumerate(evidence_list):
                    if not str(item).strip():
                        continue
                    
                    # Format: "1. Evidence text\n"
                    formatted_item = f"{i+1}. {str(item).strip()}\n"
                    item_ids = self.tokenizer(formatted_item, add_special_tokens=False)["input_ids"]
                    
                    if len(current_evidence_ids) + len(item_ids) <= available_for_evidence:
                        current_evidence_ids.extend(item_ids)
                    else:
                        # Truncate current item to fill remaining space
                        remaining = available_for_evidence - len(current_evidence_ids)
                        if remaining > 0:
                            current_evidence_ids.extend(item_ids[:remaining])
                        break
                
                evidence_ids = current_evidence_ids
            
            # 4. Construct full input
            full_input_ids = start_ids + evidence_ids + end_ids
            
            # Truncate if still too long (shouldn't happen with logic above but safety first)
            if len(full_input_ids) > self.max_length:
                full_input_ids = full_input_ids[:self.max_length]
                
            batch_input_ids.append(full_input_ids)
            batch_attention_mask.append([1] * len(full_input_ids))

        # Pad batch
        max_batch_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for ids, mask in zip(batch_input_ids, batch_attention_mask):
            padding = [pad_id] * (max_batch_len - len(ids))
            mask_padding = [0] * (max_batch_len - len(ids))
            
            # Right padding (match training)
            padded_input_ids.append(ids + padding)
            padded_attention_mask.append(mask + mask_padding)
            
        input_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long, device=self.device)

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
