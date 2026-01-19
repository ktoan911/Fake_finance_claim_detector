#!/usr/bin/env python3
"""
Crypto Claim Verification (Knowledge-Grounded)

Input: post/comment/message (Reddit/Telegram)
Output:
  - label ∈ {SUPPORTED, REFUTED, NEI}
  - evidence: top-k retrieved evidence (links/snippets)
  - confidence score
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; env vars may still be provided by the shell
    pass


def setup_logging(verbose: bool = True):
    """Configure logging."""
    logger.remove()
    level = "INFO" if verbose else "WARNING"
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def _split_list(value: str) -> list:
    return [v.strip() for v in value.split(",") if v.strip()]


def _load_source_env(prefix: str, default_uri: str, default_db: str, default_collection: str,
                     default_text_fields: str, default_ts_field: str, default_link_field: str):
    mongo_uri = os.getenv(f"{prefix}_MONGO_URI", default_uri)
    db_name = os.getenv(f"{prefix}_MONGO_DB", default_db)
    collection = os.getenv(f"{prefix}_MONGO_COLLECTION", default_collection)
    text_fields = _split_list(os.getenv(f"{prefix}_TEXT_FIELDS", default_text_fields))
    ts_field = os.getenv(f"{prefix}_TIMESTAMP_FIELD", default_ts_field)
    link_field = os.getenv(f"{prefix}_LINK_FIELD", default_link_field)
    limit = int(os.getenv(f"{prefix}_LIMIT", os.getenv("GLOBAL_LIMIT", "5000")))
    return mongo_uri, db_name, collection, text_fields, ts_field, link_field, limit


def _load_knowledge_base():
    from src.mongo_loader import MongoSourceLoader

    # Social source defaults → Reddit
    social_uri, social_db, social_col, social_text_fields, social_ts_field, social_link, social_limit = _load_source_env(
        "SOCIAL",
        "mongodb://localhost:27017",
        "social_crawler",
        "reddit_data",
        "title,body,text",
        "created_utc",
        "permalink",
    )

    social_loader = MongoSourceLoader(social_uri, social_db, social_col)
    social_query = {}
    social_subreddits = os.getenv("SOCIAL_SUBREDDITS", "")
    if social_subreddits:
        social_query = {"subreddit": {"$in": _split_list(social_subreddits)}}
    social_docs = social_loader.fetch_documents(
        text_fields=social_text_fields,
        timestamp_field=social_ts_field,
        source_label="social",
        link_field=social_link,
        limit=social_limit,
        query=social_query,
    )

    # Trusted source (optional)
    trusted_col = os.getenv("TRUSTED_MONGO_COLLECTION")
    trusted_docs = []
    if trusted_col:
        trusted_uri, trusted_db, _, trusted_text_fields, trusted_ts_field, trusted_link, trusted_limit = _load_source_env(
            "TRUSTED",
            "mongodb://localhost:27017",
            "trusted_sources",
            trusted_col,
            "title,body,content,summary",
            "published_utc",
            "url",
        )
        trusted_loader = MongoSourceLoader(trusted_uri, trusted_db, trusted_col)
        trusted_docs = trusted_loader.fetch_documents(
            text_fields=trusted_text_fields,
            timestamp_field=trusted_ts_field,
            source_label="trusted",
            link_field=trusted_link,
            limit=trusted_limit,
        )

    knowledge_base = trusted_docs + social_docs
    return knowledge_base


def run_demo():
    """
    Run a quick demonstration of the claim verification system.
    """
    print("\n" + "=" * 70)
    print("🔍 CRYPTO CLAIM VERIFICATION DEMO")
    print("   SUPPORTED / REFUTED / NEI")
    print("=" * 70 + "\n")
    
    from src.pipeline import CryptoClaimVerificationPipeline, PipelineConfig
    
    # Initialize
    print("📊 Initializing pipeline...")
    config = PipelineConfig(
        alpha=0.7,
        verbose=False
    )
    config.use_llm = os.getenv("USE_LLM", "0") == "1"
    config.llm_model_name = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B")
    config.device = os.getenv("DEVICE", "cpu")
    config.support_threshold = float(os.getenv("SUPPORT_THRESHOLD", "0.7"))

    pipeline = CryptoClaimVerificationPipeline(config)
    pipeline.build()
    
    print("📚 Loading knowledge base from MongoDB sources...")
    knowledge_base = _load_knowledge_base()

    print("⚙️  Building retrieval index...")
    pipeline.fit(knowledge_base)
    
    # Test samples
    test_texts = [
        "SEC approves spot Bitcoin ETF in 2024.",
        "Elon Musk announced a 10x BTC giveaway today.",
        "Binance suffered a major hack and paused withdrawals.",
    ]
    
    print("\n" + "-" * 70)
    print("🎯 PREDICTION RESULTS")
    print("-" * 70)
    
    predictions = pipeline.predict(test_texts)
    
    for pred in predictions:
        print(f"\nLabel: {pred.predicted_label} (Confidence: {pred.confidence:.2f})")
        print(f"Claim: {pred.text}")
        if pred.evidence:
            print("Evidence:")
            for ev in pred.evidence[:2]:
                link = ev.get("link") or "N/A"
                print(f"  - {ev['source']} | score={ev['score']:.3f} | {link}")
    
    print("\n" + "=" * 70)
    print("✨ Demo complete! Use --mode experiment for full results.")
    print("=" * 70 + "\n")


def run_experiment():
    print("\n" + "=" * 70)
    print("🔬 CLAIM VERIFICATION RUN")
    print("=" * 70 + "\n")
    run_demo()


def run_interactive():
    """
    Run interactive mode for testing individual texts.
    """
    print("\n" + "=" * 70)
    print("🔍 INTERACTIVE CLAIM VERIFIER")
    print("   Type claim text, 'quit' to exit")
    print("=" * 70 + "\n")
    
    from src.pipeline import CryptoClaimVerificationPipeline, PipelineConfig
    
    # Initialize
    print("Initializing... (this may take a moment)")
    config = PipelineConfig(verbose=False)
    config.use_llm = os.getenv("USE_LLM", "0") == "1"
    config.llm_model_name = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B")
    config.device = os.getenv("DEVICE", "cpu")
    config.support_threshold = float(os.getenv("SUPPORT_THRESHOLD", "0.7"))

    pipeline = CryptoClaimVerificationPipeline(config)
    pipeline.build()
    knowledge_base = _load_knowledge_base()
    pipeline.fit(knowledge_base)
    print("Ready!\n")
    
    while True:
        try:
            text = input("📝 Enter text to analyze (or 'quit'): ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not text:
                continue
            
            pred = pipeline.predict([text])[0]

            print("\n" + "-" * 50)
            print(f"Label: {pred.predicted_label}")
            print(f"Confidence: {pred.confidence:.2f}")
            print(f"Processing Time: {pred.processing_time_ms:.2f}ms")
            if pred.evidence:
                print("\nEvidence:")
                for i, ev in enumerate(pred.evidence[:2], 1):
                    link = ev.get("link") or "N/A"
                    print(f"  {i}. {ev['source']} | score={ev['score']:.3f} | {link}")
            
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crypto Claim Verification (SUPPORTED/REFUTED/NEI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo          Quick demonstration
  python main.py --mode experiment    Claim verification run
  python main.py --mode interactive   Interactive testing mode
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="demo",
        choices=["demo", "experiment", "interactive"],
        help="Execution mode (default: demo)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "experiment":
        run_experiment()
    elif args.mode == "interactive":
        run_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
