#!/usr/bin/env python3
"""
Fusion Model Inference Script

Performs claim verification using the trained Fusion model with MongoDB knowledge base.

Usage:
    python inference_fusion.py --claim "Bitcoin will hit 100k tomorrow" --mongo_uri "mongodb://localhost:27017/"
"""

import argparse
import os
from loguru import logger
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.mongo_loader import MongoKBLoader
from src.retrieval import KnowledgeAugmentedRetriever
from src.llm_scorer import LLMScorer
from src.fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
from src.config import PROMPT_TEMPLATE, LABEL_LIST


def load_fusion_model(model_path: str, device: str = "cuda"):
    """
    Load trained fusion model from checkpoint.
    
    Args:
        model_path: Path to fusion_model.pt file
        device: Device to load model on
        
    Returns:
        Tuple of (fusion, retrieval_encoder, config)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for inference")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize models
    retrieval_encoder = RetrievalFeatureEncoder(
        num_retrieved=config['top_k'],
        score_features=4,
        hidden_dim=64,
        output_dim=64
    ).to(device)
    
    fusion = ConfidenceAwareFusion(
        retrieval_input_dim=64,
        hidden_dim=128,
        num_classes=config['num_classes'],
        initial_beta=config['initial_beta'],
        lambda_reg=config['lambda_reg']
    ).to(device)
    
    # Load state dicts
    retrieval_encoder.load_state_dict(checkpoint['retrieval_encoder'])
    fusion.load_state_dict(checkpoint['fusion'])
    
    # Set to eval mode
    retrieval_encoder.eval()
    fusion.eval()
    
    logger.info(f"Loaded fusion model from {model_path}")
    logger.info(f"Beta: {fusion.beta.item():.4f}")
    
    return fusion, retrieval_encoder, config


def predict_claim(
    claim: str,
    retriever: KnowledgeAugmentedRetriever,
    llm: LLMScorer,
    fusion,
    retrieval_encoder,
    config: dict,
    device: str = "cuda"
) -> dict:
    """
    Predict label for a single claim.
    
    Args:
        claim: Claim text to verify
        retriever: Initialized retriever with indexed KB
        llm: Initialized LLM scorer
        fusion: Fusion model
        retrieval_encoder: Retrieval feature encoder
        config: Model config
        device: Device
        
    Returns:
        Dictionary with prediction results
    """
    # 1. Retrieve evidence (FAISS filtering: top 500 candidates, then BM25 re-rank)
    results = retriever.retrieve(claim, top_k=config['top_k'], candidate_pool_size=500)
    
    # 2. Extract retrieval features
    features = []
    evidence_texts = []
    for r in results:
        features.append([r.score, r.bm25_score, r.recency_score, r.cyclicity_score])
        evidence_texts.append(r.text)
    
    # Pad if needed
    while len(features) < config['top_k']:
        features.append([0.0, 0.0, 0.0, 0.0])
    
    import numpy as np
    retrieval_features = np.array([features], dtype=np.float32)  # [1, top_k, 4]
    retrieval_tensor = torch.tensor(retrieval_features, device=device)
    
    # 3. Get LLM logits
    with torch.inference_mode():
        llm_logits = llm.score_logits([claim], [evidence_texts])  # [1, num_classes]
        
        # 4. Encode retrieval features
        retrieval_encoded = retrieval_encoder(retrieval_tensor)  # [1, 64]
        
        # 5. Fusion
        output = fusion(llm_logits, retrieval_encoded)
        
        # 6. Get predictions
        probs = output.final_probs[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_label = config.get('label_list', LABEL_LIST)[pred_idx]
        confidence = float(probs[pred_idx])
    
    return {
        'claim': claim,
        'predicted_label': pred_label,
        'confidence': confidence,
        'probabilities': {
            label: float(prob) 
            for label, prob in zip(config.get('label_list', LABEL_LIST), probs)
        },
        'evidence': [
            {
                'text': r.text[:300],
                'score': r.score,
                'url': r.metadata.get('url', ''),
                'timestamp': r.timestamp.isoformat()
            }
            for r in results[:3]  # Top 3
        ],
        'fusion_beta': fusion.beta.item()
    }


def main():
    parser = argparse.ArgumentParser(description="Fusion model inference with MongoDB KB")
    parser.add_argument("--claim", type=str, required=True, help="Claim to verify")
    parser.add_argument("--fusion_model", type=str, default="artifacts/fusion_model.pt", help="Path to fusion model")
    parser.add_argument("--llm_model", type=str, default=os.getenv("LORA_MODEL_PATH", "artifacts/lora_llm"), help="Path to LLM model")
    parser.add_argument("--mongo_uri", type=str, default="mongodb://localhost:27017/", help="MongoDB URI")
    parser.add_argument("--mongo_db", type=str, default="crypto_kb", help="MongoDB database name")
    parser.add_argument("--mongo_collection", type=str, default="posts", help="MongoDB collection name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--kb_limit", type=int, default=None, help="Limit KB documents (None = all)")
    
    args = parser.parse_args()
    
    # Load fusion model
    logger.info("Loading fusion model...")
    fusion, retrieval_encoder, config = load_fusion_model(args.fusion_model, args.device)
    
    # Load MongoDB knowledge base
    logger.info("Loading knowledge base from MongoDB...")
    mongo_loader = MongoKBLoader(
        mongo_uri=args.mongo_uri,
        database=args.mongo_db,
        collection=args.mongo_collection
    )
    kb_docs = mongo_loader.load_documents(limit=args.kb_limit)
    mongo_loader.close()
    
    if not kb_docs:
        logger.error("No documents loaded from MongoDB!")
        return
    
    # Initialize retriever with Cross-Encoder
    logger.info("Indexing knowledge base...")
    retriever = KnowledgeAugmentedRetriever(
        alpha=0.7,
        lambda_decay=0.1,
        gamma=0.5,
        use_query_expansion=True,
        use_cross_encoder=True  # Enable 4-stage pipeline
    )
    retriever.index_documents(kb_docs, text_field='text', id_field='id', timestamp_field='timestamp')
    
    # Initialize LLM
    logger.info("Loading LLM scorer...")
    llm = LLMScorer(
        model_name=args.llm_model,
        device=args.device,
        max_length=2048,
        labels=config.get('label_list', LABEL_LIST),
        prompt_template=PROMPT_TEMPLATE
    )
    
    # Predict
    logger.info(f"\nVerifying claim: {args.claim}")
    result = predict_claim(
        claim=args.claim,
        retriever=retriever,
        llm=llm,
        fusion=fusion,
        retrieval_encoder=retrieval_encoder,
        config=config,
        device=args.device
    )
    
    # Print results
    print("\n" + "="*80)
    print("CLAIM VERIFICATION RESULT")
    print("="*80)
    print(f"Claim: {result['claim']}")
    print(f"\nPrediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nProbabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.2%}")
    
    print(f"\nFusion Beta: {result['fusion_beta']:.4f} (LLM weight)")
    
    print(f"\nTop Evidence:")
    for i, ev in enumerate(result['evidence'], 1):
        print(f"\n{i}. Score: {ev['score']:.4f}")
        print(f"   {ev['text'][:150]}...")
        print(f"   URL: {ev['url']}")
        print(f"   Time: {ev['timestamp']}")
    print("="*80)


if __name__ == "__main__":
    main()
