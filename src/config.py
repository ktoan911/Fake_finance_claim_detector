"""
Shared configuration for LoRA training and Fusion training.
Ensures consistency in prompts and labels across the pipeline.
"""

# Label mapping - Binary classification (True/False only)
LABEL_LIST = ["True", "False"]
LABEL_TO_ID = {"True": 0, "False": 1}
ID_TO_LABEL = {0: "True", 1: "False"}

# Prompt template used for both LoRA fine-tuning and Fusion scoring
# MUST include {claim} and {evidence} placeholders
# Output uses existing vocabulary: True (supported) or False (refuted)
PROMPT_TEMPLATE = """You are an expert fact-checker for financial claims.

Classify the claim based on the evidence:
- True: Evidence confirms the claim
- False: Evidence contradicts the claim or insufficient evidence

Claim: {claim}

Evidence: {evidence}

Verdict:"""
