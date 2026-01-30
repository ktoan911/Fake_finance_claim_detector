"""
Shared configuration for LoRA training and Fusion training.
Ensures consistency in prompts and labels across the pipeline.
"""

# Label mapping - Direct model output labels
LABEL_LIST = ["True", "False", "Not"]
LABEL_TO_ID = {"True": 0, "False": 1, "Not": 2}
ID_TO_LABEL = {0: "True", 1: "False", 2: "Not"}

# Prompt template used for both LoRA fine-tuning and Fusion scoring
# MUST include {claim} and {evidence} placeholders
# Output uses existing vocabulary: True (supported), False (refuted), Unknown (NEI)
PROMPT_TEMPLATE = """You are an expert fact-checker for financial claims.

Classify the claim based on the evidence:
- True: Evidence confirms the claim
- False: Evidence contradicts the claim  
- Not: Insufficient evidence

Claim: {claim}

Evidence: {evidence}

Verdict:"""
