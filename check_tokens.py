
from transformers import AutoTokenizer

model_path = "./lora_llm"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    words = ["Unsure", "UNSURE", "unsure", "true", "TRUE", "True", "false", "FALSE",  "false",  "False"]
    for word in words:
        tokens = tokenizer(word, add_special_tokens=False)["input_ids"]
        print(f"Word: '{word}' -> Tokens: {tokens} (Count: {len(tokens)})")
        for t in tokens:
            print(f"  Token ID {t} -> '{tokenizer.decode([t])}'")

except Exception as e:
    print(f"Error loading tokenizer: {e}")
