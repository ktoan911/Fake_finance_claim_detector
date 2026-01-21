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
PROMPT_TEMPLATE = """You are an expert fact-checker for financial claims.

Classify the claim based on the evidence:
- SUPPORTED: Evidence confirms the claim
- REFUTED: Evidence contradicts the claim
- NEI: Insufficient evidence

Answer using EXACTLY ONE token from the following list:
<SUPPORTED>
<REFUTED>
<NEI>

Claim: {claim}

Evidences: {evidence}

Verdict:"""
