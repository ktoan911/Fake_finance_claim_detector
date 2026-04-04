# Fake Crypto Claim Detector

An AI system for detecting fake cryptocurrency claims using RAG + LoRA-finetuned LLM + Confidence-Aware Fusion.

## Project Structure

```
├── src/                    # Core library
│   ├── config.py           # Global config & prompts
│   ├── utils.py            # Shared utilities
│   ├── llm_scorer.py       # LLM inference wrapper
│   ├── models/             # Model architectures & inference
│   │   ├── fusion.py
│   │   └── fusion_inference.py
│   ├── training/           # Training logic
│   │   ├── lora_trainer.py
│   │   └── fusion_trainer.py
│   ├── retrieval/          # Embedding & retrieval
│   │   ├── embeddings.py
│   │   └── retrieval.py
│   ├── database/           # OpenSearch client
│   │   └── opensearch.py
│   └── data/               # Data loading & crawlers
│       ├── csv_loader.py
│       └── crawlers/
│           ├── news_crawler/   # Vietnamese news crawler
│           └── crawl_fb/       # Facebook scraper
│
├── scripts/                # Entry-point scripts
│   ├── cli.py              # Claim verification CLI
│   ├── train_lora.py       # LoRA fine-tuning
│   ├── train_fusion.py     # Fusion model training
│   ├── train_retrieval.py  # Retrieval training
│   └── gen_data.py         # Data generation
│
├── data/                   # Train/dev/test CSVs
├── models/                 # Saved model weights
├── results/                # Crawler outputs & logs
├── tests/                  # Unit & integration tests
├── docs/                   # Papers & documentation
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run claim verification CLI
python scripts/cli.py

# Train LoRA
python scripts/train_lora.py

# Train Fusion model
python scripts/train_fusion.py
```

## Documentation

See [docs/FakeCryptoClaimDetector_Paper.md](docs/FakeCryptoClaimDetector_Paper.md) for the full technical paper.
