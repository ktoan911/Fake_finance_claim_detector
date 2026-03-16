LABEL_LIST = ["A", "B", "C"]
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}

# Prompt template used for both LoRA fine-tuning and Fusion scoring
# MUST include {claim} and {evidence} placeholders
# Output: A (supported), B (refuted), C (not enough info)
PROMPT_TEMPLATE = """You are an expert fact-checker verifying Vietnamese claims based on the provided evidence.

Classify the claim based on ALL the evidence and answer with ONLY a single letter:
- A: The evidence supports the claim
- B: The evidence contradicts the claim
- C: There is not enough evidence to support or refute the claim

Claim: {claim}

Evidence: {evidence}

Conclusion: """