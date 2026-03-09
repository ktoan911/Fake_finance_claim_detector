"""
Shared configuration for LoRA training and Fusion training.
Ensures consistency in prompts and labels across the pipeline.
"""

# Label mapping - Binary classification.
# Keep ID convention stable across pipeline:
#   - ID 0: supported/legitimate
#   - ID 1: refuted/fake
LABEL_LIST = ["Đúng", "Sai"]
LABEL_TO_ID = {"Đúng": 0, "Sai": 1}
ID_TO_LABEL = {0: "Đúng", 1: "Sai"}

# Prompt template used for both LoRA fine-tuning and Fusion scoring
# MUST include {claim} and {evidence} placeholders
# Output tokens follow Vietnamese labels for PhoGPT and Vietnamese prompts.
PROMPT_TEMPLATE = """Bạn là một chuyên gia kiểm chứng thông tin cho các tuyên bố trong lĩnh vực tài chính.

Hãy phân loại tuyên bố dựa trên bằng chứng:
- Đúng: Bằng chứng xác nhận tuyên bố
- Sai: Bằng chứng mâu thuẫn với tuyên bố hoặc không đủ bằng chứng

Tuyên bố: {claim}

Bằng chứng: {evidence}

Kết luận: """
