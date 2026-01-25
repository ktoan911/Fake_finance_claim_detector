
import sys
import os
from transformers import AutoTokenizer
from src.csv_loader import CSVLabeledLoader
from src.lora_trainer import _prepare_classification_dataset
from src.config import PROMPT_TEMPLATE

# Load real data
csv_path = "data/finfact_filtered.csv"
print(f"Loading data from {csv_path}...")
loader = CSVLabeledLoader(csv_path)
df = loader.load()
print(f"Loaded {len(df)} samples.")

# Take first 2 samples
sample_df = df.head(2)
claims = sample_df["text"].tolist()
evidences = sample_df["evidence"].tolist()
labels = sample_df["label"].tolist()

# Mock Tokenizer to avoid downloading models
class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 9999
        self.pad_token_id = 0
    
    def __call__(self, text, add_special_tokens=False, truncation=False):
        # Simple mock: 1 char = 1 token (plus offset to avoid special tokens)
        # This makes length calculation easy to predict
        # We use a simple mapping that preserves characters
        input_ids = [ord(c) for c in text]
        return {"input_ids": input_ids}
    
    def decode(self, token_ids):
        # Filter out special tokens
        valid_ids = [t for t in token_ids if t not in [9999, 0]]
        return "".join([chr(t) for t in valid_ids])

# Initialize tokenizer
tokenizer = MockTokenizer()

# Test with small max_length to force truncation
# Since 1 char = 1 token, we can set max_length in characters
MAX_LENGTH = 500 
print(f"\n--- Testing with max_length={MAX_LENGTH} (Force Truncation) ---")

tokenized = _prepare_classification_dataset(
    claims, evidences, labels, tokenizer, MAX_LENGTH, PROMPT_TEMPLATE
)

for i in range(len(claims)):
    input_ids = tokenized[i]["input_ids"]
    decoded = tokenizer.decode(input_ids)
    print(f"\nSample {i+1}:")
    print(f"Original Evidence Length: {len(evidences[i])} chars")
    print(f"Tokenized Length: {len(input_ids)}")
    print("-" * 40)
    print(decoded)
    print("-" * 40)
    
    # Verify structure
    if "Verdict:" in decoded:
        print("✅ Verdict template present")
    else:
        print("❌ Verdict template MISSING")
        
    if "1." in decoded:
        print("✅ Numbered list present")
    else:
        print("❌ Numbered list MISSING")

