"""
Shared configuration for LoRA training and Fusion training.
Ensures consistency in prompts and labels across the pipeline.
"""

# Label mapping
LABEL_LIST = ["SUPPORTED", "REFUTED", "NEI"]
LABEL_TO_ID = {"SUPPORTED": 0, "REFUTED": 1, "NEI": 2}
ID_TO_LABEL = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}

# Prompt template used for both LoRA fine-tuning and Fusion scoring
# MUST include {claim} and {evidence} placeholders
PROMPT_TEMPLATE = """You are a crypto claim verification assistant.

Given the following claim and evidence, classify the claim as:
- SUPPORTED: The evidence supports the claim
- REFUTED: The evidence contradicts the claim
- NEI: Not Enough Information to verify

Claim: {claim}

Evidence: {evidence}

Classification:"""
