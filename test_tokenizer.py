import sys
from transformers import AutoTokenizer

model_name = "vinai/PhoGPT-4B"
if len(sys.argv) > 1:
    model_name = sys.argv[1]

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
except Exception as e:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

for label in ["Đúng", "Sai"]:
    tokens = tokenizer(label, add_special_tokens=False)["input_ids"]
    print(f"Model {model_name} Label '{label}' -> tokens {tokens}")
    for t in tokens:
        print(f"   {t}: {tokenizer.decode([t])}")
