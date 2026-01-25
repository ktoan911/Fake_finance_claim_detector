# 🔍 Crypto Claim Verification (Knowledge-Grounded)

**Goal**: Verify crypto/finance claims using retrieval‑augmented evidence.

**Input**: one post/comment/message (Reddit/Telegram)  
**Output**:  
- `label ∈ {SUPPORTED, REFUTED, NEI}`  
- `evidence`: top‑k retrieved evidence (link/snippet)  
- `confidence`: reliability score

## 📋 System Summary

The system addresses three key limitations of claim verification:
1. **Static knowledge bases** - crypto info evolves rapidly
2. **Unreliable LM outputs** - hallucinations without evidence
3. **Missing evidence handling** - explicit NEI when evidence is insufficient

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Query                                    │
│                  "Is this claim true?"                          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   FAISS       │ │   BM25        │ │   Temporal    │
│   Retriever   │ │   Scoring     │ │   Scoring     │
│   (BGE)       │ │               │ │   (Eq. 1)     │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
              ┌───────────────────────┐
              │  Knowledge-Augmented  │
              │      Retrieval        │
              │    (Section 3.1)      │
              └───────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                                   ▼
┌───────────────┐                   ┌───────────────┐
│   LM-based    │                   │   Retrieval   │
│   Prediction  │                   │   Evidence    │
│   pLM(y|q)    │                   │   pret(y|D)   │
└───────┬───────┘                   └───────┬───────┘
        │                                   │
        └─────────────────┬─────────────────┘
                          ▼
              ┌───────────────────────┐
              │  Confidence-Aware     │
              │      Fusion           │
              │    (Section 3.2)      │
              │  pfinal = σ(β·pLM +   │
              │    (1-β)·MLP(pret))   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
│   Final Prediction    │
│ SUPPORTED/REFUTED/NEI │
              └───────────────────────┘
```

## 🧮 Key Equations

### Equation 1: Temporal Scoring
```
Score(q, di) = α · BM25(q, di) + (1 − α) · Recency(di)
```
where `α = 0.7` and `Recency(di) = e^(-λt)` with `λ = 0.1`

### (Optional) Fusion Training
If you train fusion (MLP + β), it follows the paper’s formulation. See `fusion_trainer.py`.

## 📁 Project Structure

```
paperr_rag/
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
│
├── src/
│   ├── __init__.py
│   ├── retrieval.py        # Knowledge-Augmented Retrieval (Section 3.1)
│   ├── pipeline.py         # Claim verification pipeline
│   ├── mongo_loader.py     # Generic MongoDB loader (trusted + social)
│   ├── llm_scorer.py        # LLM scoring (SUPPORTED/REFUTED/NEI)
│   ├── lora_trainer.py      # LoRA fine-tuning (optional)
│   └── fusion_trainer.py    # Fusion training (optional)
│
└── 27_Knowledge_Grounded_Detectio.txt  # Original paper
```

## 🚀 Quick Start

### Installation

```bash
# Clone/navigate to the project
cd paperr_rag

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Quick demonstration
python main.py --mode demo

# Interactive verification
python main.py --mode interactive
```

## 🧠 Optional Training (LoRA + Fusion MLP + β)

Training is optional and requires GPU + torch/transformers/peft.

### Environment Variables

```bash
export SOCIAL_MONGO_URI="mongodb://localhost:27017"
export SOCIAL_MONGO_DB="social_crawler"
export SOCIAL_MONGO_COLLECTION="reddit_data"
export SOCIAL_SUBREDDITS="CryptoScams,Scams,CryptoCurrency"
export SOCIAL_LIMIT="5000"

# Trusted news/official sources (optional)
export TRUSTED_MONGO_COLLECTION="trusted_news"
export TRUSTED_MONGO_DB="trusted_sources"
export TRUSTED_MONGO_URI="mongodb://localhost:27017"

# LLM / training
export LLM_MODEL_NAME="meta-llama/Llama-3.1-8B"
export DEVICE="cuda"   # or cpu
```

### Labeled CSV (text,evidence,label) for LoRA + Fusion

Training uses a single labeled CSV for BOTH LoRA and Fusion.

```bash
export LABELED_CSV_PATH="/path/to/labeled.csv"
```

CSV must contain columns:
```
text,evidence,label
```
Optional:
```
timestamp
```
Label can be:
```
SUPPORTED / REFUTED / NEI
```
Also accepted:
```
true / false / neutral
```

### Train LoRA Only

```bash
export LABELED_CSV_PATH="/path/to/labeled.csv"
export LORA_OUTPUT_DIR="artifacts/lora_llm"
python train_lora.py
```

### Train Fusion Only

```bash
export LABELED_CSV_PATH="/path/to/labeled.csv"
export LORA_MODEL_PATH="artifacts/lora_llm"   # use LoRA model if available
export FUSION_OUTPUT_PATH="artifacts/fusion_model.pt"
python train_fusion.py
```

### Train All Components (LoRA + Fusion)

Chạy lần lượt 2 bước:
```bash
python train_lora.py
python train_fusion.py
```

### Use LLM at Inference

```bash
export USE_LLM=1
export LLM_MODEL_NAME="meta-llama/Llama-3.1-8B"
export DEVICE="cuda"   # or cpu

python main.py --mode demo
```

## 🔧 Configuration

Key parameters from the paper:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.7 | BM25 vs temporal weight |
| `lambda_decay` | 0.1 | Recency decay factor |
| `gamma` | 0.5 | Recency vs cyclicity mix |
| `top_k_retrieval` | 5 | Evidence items to return |
| `use_llm` | false | Use LLM to output SUPPORTED/REFUTED/NEI |
| `support_threshold` | 0.7 | Threshold for SUPPORTED when LLM disabled |

## 🔬 Components Explained

### 1. Knowledge-Augmented Retrieval (`retrieval.py`)
- **Temporal Scoring**: Prioritizes recent evidence using exponential decay
- **BM25 + BGE**: Combines lexical and semantic matching
- **Cycle-Aware**: Uses FFT to detect repeating scam patterns

### 2. Confidence-Aware Fusion (`fusion.py`)
- **Trainable β**: Learns optimal mix of LM vs retrieval
- **MLP Projection**: Maps retrieval scores to label space
- **Contrastive Training**: Optimizes with InfoNCE-style loss

### 3. Adaptive Threshold (`threshold_optimizer.py`) [Optional]
- Used only if you train a classifier and need threshold tuning

### 4. Semantic Embeddings (`embeddings.py`) [Optional]
- Contrastive fine‑tuning for domain‑specific retrieval

## 🎯 Use Cases

1. **Claim verification** for crypto news and social posts
2. **Compliance**: triage questionable claims for review
3. **Research**: track misinformation trends in crypto finance

## ⚠️ Limitations

- Evidence quality depends on your MongoDB sources
- Full LLM + LoRA requires GPU resources
- NEI requires evidence coverage; missing sources → more NEI

## 🗄️ MongoDB Sources

The system uses **two sources**:
1) **Trusted** news/official sources  
2) **Social** posts (Reddit/Telegram)

### Social Collection Schema (Required Fields)

```json
{
  "_id": "ObjectId(...)",
  "subreddit": "CryptoScams",
  "title": "Lost all ETH after fake mint link",
  "body": "Clicked fake link, wallet drained",
  "created_utc": 1722643200,
  "author": "user123",
  "permalink": "/r/CryptoScams/...",
  "is_deleted": false,
  "is_removed": false
}
```

### Mapping to Paper Variables

```text
document d_i:
  text = title + body
  timestamp = created_utc

Recency:
  t_days = (now - created_utc).days
  recency = exp(-lambda * t_days)

BM25 / Embedding:
  uses ONLY text (title + body)
```

### Trusted Collection Schema (Required Fields)

```json
{
  "_id": "ObjectId(...)",
  "title": "SEC approves spot Bitcoin ETF",
  "content": "The SEC approved ...",
  "published_utc": 1722643200,
  "url": "https://example.com/news/...",
  "source": "Reuters"
}
```

### Example Usage

```python
from src.mongo_loader import MongoSourceLoader
from src.pipeline import CryptoClaimVerificationPipeline, PipelineConfig

social_loader = MongoSourceLoader(
    mongo_uri="mongodb://localhost:27017",
    db_name="social_crawler",
    collection_name="reddit_data"
)

social_docs = social_loader.fetch_documents(
    text_fields=["title", "body", "text"],
    timestamp_field="created_utc",
    source_label="social",
    link_field="permalink",
    limit=5000
)

config = PipelineConfig()
pipeline = CryptoClaimVerificationPipeline(config)
pipeline.build()
pipeline.fit(knowledge_base=social_docs)
```

## 📚 References

```bibtex
@article{li2024knowledge,
  title={Knowledge-Grounded Detection of Cryptocurrency Scams with Retrieval-Augmented LMs},
  author={Li, Zichao},
  journal={Canoakbit Alliance},
  year={2024}
}
```

## 📄 License

This implementation is for educational and research purposes.

---

**Note**: This is a research implementation. For production use, consider:
- Using actual LLM (Llama-3, GPT-4) for claim verification
- Connecting to trusted news/official sources for evidence
- Implementing proper API rate limiting and error handling




👉 Tất cả được gộp lại theo công thức trong paper:

Score(q, d) = α · BM25(q, d) + (1 − α) · Temporal(d)


Trong đó:

Temporal(d) = γ · Recency(d) + (1 − γ) · Cyclicity(d)