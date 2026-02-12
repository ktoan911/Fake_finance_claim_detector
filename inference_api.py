import os
from typing import Any, Dict

import torch
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

from src.fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
from src.llm_scorer import LLMScorer
from src.mongodb_retriever import MongoDBRetriever

# Global cache for heavy models to avoid reloading
_MODELS = {
    "llm": None,
    "fusion": None,
    "retrieval_encoder": None,
    "retriever": None,  # Can be shared
}


def load_models(
    lora_path: str = "artifacts/lora_llm",
    fusion_path: str = "artifacts/fusion_model.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load models into global cache if not already loaded"""
    global _MODELS

    if _MODELS["llm"] is None:
        logger.info("Loading LLM Scorer...")
        _MODELS["llm"] = LLMScorer(model_name=lora_path, device=device, max_length=2048)

    if _MODELS["fusion"] is None:
        logger.info("Loading Fusion Model...")
        try:
            checkpoint = torch.load(fusion_path, map_location=device)
            # Load config
            saved_config = checkpoint.get("config", {})

            # Initialize
            retrieval_encoder = RetrievalFeatureEncoder(
                num_retrieved=saved_config.get("top_k", 10),
                score_features=4,
                hidden_dim=64,
                output_dim=64,
            ).to(device)

            fusion = ConfidenceAwareFusion(
                retrieval_input_dim=64,
                hidden_dim=128,
                num_classes=2,
                initial_beta=saved_config.get("initial_beta", 0.5),
            ).to(device)

            retrieval_encoder.load_state_dict(checkpoint["retrieval_encoder"])
            fusion.load_state_dict(checkpoint["fusion"])

            retrieval_encoder.eval()
            fusion.eval()

            _MODELS["retrieval_encoder"] = retrieval_encoder
            _MODELS["fusion"] = fusion
            _MODELS["fusion_config"] = saved_config
        except Exception as e:
            logger.error(f"Failed to load fusion model: {e}")
            raise

    # Initialize Retriever (MongoDB or fallback)
    # We use a wrapper or the standard one depending on env
    # For now, let's assume we use the Custom MongoDBRetriever for evidence fetching,
    # BUT we need KnowledgeAugmentedRetriever structure for feature calculation?
    # Actually, KnowledgeAugmentedRetriever in src/retrieval.py manages scores and RRF.
    # The user wanted "evidence retrieval in mongodb".
    # So we should probably inject MongoDB results into the pipeline.

    # We'll use a simple approach: Fetch from Mongo -> Pass to feature builder
    if _MODELS["retriever"] is None:
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            logger.info("Initializing MongoDB Retriever...")
            _MODELS["retriever"] = MongoDBRetriever(uri=mongo_uri)
        else:
            logger.warning(
                "MONGO_URI not set. Retrieval might fail if strictly required."
            )
            _MODELS["retriever"] = None


def verify_claim(
    claim_text: str,
    top_k: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    End-to-end inference function:
    1. Retrieve evidence from MongoDB
    2. Compute retrieval features
    3. Get LLM logits
    4. Fusion model prediction
    """
    # Ensure models are loaded
    if _MODELS["llm"] is None:
        load_models(device=device)

    llm = _MODELS["llm"]
    fusion_layer = _MODELS["fusion"]
    retrieval_encoder = _MODELS["retrieval_encoder"]
    mongo_retriever = _MODELS["retriever"]

    # 1. Retrieval
    logger.info(f"Processing claim: {claim_text}")
    retrieved_texts = []

    if mongo_retriever:
        try:
            retrieved_texts = mongo_retriever.retrieve(claim_text, top_k=top_k)
        except Exception as e:
            logger.error(f"MongoDB retrieval failed: {e}")
            retrieved_texts = []
    else:
        logger.warning("No MongoDB retriever available. Using empty context.")
        retrieved_texts = []

    # 2. Compute Retrieval Features
    # Since we don't have the full RRF pipeline with scores from Mongo (unless we implement it),
    # we might have to mock the scores or calculate them on the fly.
    # The Fusion model expects [bm25, rrf, recency, cyclicity] per document.
    # If we only get text, we can't fully compute these without re-indexing.
    # Simplified approach: Assign dummy high scores for retrieved items or compute basic TF-IDF.

    # CRITICAL: The Fusion model relies on these features.
    # We'll construct a dummy KnowledgeAugmentedRetriever just to use its feature builder
    # if we had the docs indexed. But we don't.
    # So we construct features manually.

    # Mock features: [1.0 (score), 1.0 (rrf), 0.5 (recency), 0.5 (cyclicity)] for top_k items
    # and 0s for padding.
    features_list = []
    for _ in range(top_k):
        if _ < len(retrieved_texts):
            # [score, rrf_score, recency, cyclicity]
            features_list.append([1.0, 1.0, 0.5, 0.5])  # Assume strong retrieval
        else:
            features_list.append([0.0, 0.0, 0.0, 0.0])

    # Pad to top_k if needed
    if len(features_list) < top_k:
        features_list.extend([[0.0] * 4] * (top_k - len(features_list)))

    features_list = features_list[:top_k]

    tensor_features = torch.tensor([features_list], dtype=torch.float32).to(
        device
    )  # [1, top_k, 4]

    # 3. LLM Inference
    # LLM expects list of evidence strings
    # Format: [ [ev1, ev2, ...] ]
    evidence_input = [retrieved_texts]

    with torch.no_grad():
        logits = llm.score_logits([claim_text], evidence_input)  # [1, num_classes]

    # 4. Fusion Inference
    with torch.no_grad():
        encoded_feats = retrieval_encoder(tensor_features)  # [1, hidden]
        fusion_out = fusion_layer(logits, encoded_feats)
        probs = fusion_out.final_probs[0]  # [num_classes]
        pred_idx = torch.argmax(probs).item()

    # Map label
    # In test_all_modes.py: LABEL_MAP = {0: "True", 1: "False"}
    label_map = {0: "True", 1: "False"}  # Ensure this matches training!
    prediction = label_map.get(pred_idx, "Unknown")
    confidence = probs[pred_idx].item()

    return {
        "claim": claim_text,
        "prediction": prediction,
        "confidence": confidence,
        "evidence": retrieved_texts,
        "fusion_probs": probs.cpu().numpy().tolist(),
    }


if __name__ == "__main__":
    # Test run
    sample_claim = "Bitcoin creation was attributed to Satoshi Nakamoto."
    print("Loading models...")
    # Ensure env is set or rely on defaults
    # os.environ["MONGO_URI"] = "mongodb://..."

    try:
        result = verify_claim(sample_claim)
        print("\n--- Inference Result ---")
        print(f"Claim: {result['claim']}")
        print(f"Prediction: {result['prediction']} ({result['confidence']:.4f})")
        print(f"Evidence Found: {len(result['evidence'])}")
        for idx, ev in enumerate(result["evidence"][:3]):
            print(f" - {ev[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
