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
# Output uses existing vocabulary: True (supported), False (refuted), Unknown (NEI)
PROMPT_TEMPLATE = """You are an expert fact-checker for financial claims.

Classify the claim based on the evidence:
- True: Evidence confirms the claim
- False: Evidence contradicts the claim  
- Unknown: Insufficient evidence

Claim: {claim}

Evidence: {evidence}

Verdict:"""
