import datetime

LABEL_LIST = ["A", "B", "C"]
LABEL_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_LABEL = {0: "A", 1: "B", 2: "C"}

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PROMPT_TEMPLATE = f"""You are an expert fact-checker verifying Vietnamese claims based on the provided evidence.

Classify the claim based on ALL the evidence and answer with ONLY a single letter:
- A: The evidence supports the claim
- B: The evidence contradicts the claim
- C: There is not enough evidence to support or refute the claim

Claim: Thời gian hiện tại bây giờ là {current_time}. {{claim}}

Evidence: {{evidence}}

Conclusion: """
