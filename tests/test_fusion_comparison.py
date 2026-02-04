#!/usr/bin/env python3
"""
Optimized Model Comparison Test Script
Tests accuracy, precision, recall, and F1 scores for:
1. Fusion Model (retrieved evidence)
2. Fusion Model (gold evidence)
3. LoRA LLM only (retrieved evidence)
4. LoRA LLM only (gold evidence)

OPTIMIZATIONS:
- Load models once, reuse for all tests
- Pre-compute retrieval features and LLM logits once
- Batch processing with larger default sizes
- Optional sampling for quick testing

Usage:
    python tests/test_fusion_comparison.py
    python tests/test_fusion_comparison.py --sample 100  # Test on 100 samples only
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
from tqdm import tqdm

from src.fusion_trainer import FusionTrainingConfig, _normalize_label, _build_retrieval_features
from src.retrieval import KnowledgeAugmentedRetriever
from src.llm_scorer import LLMScorer
from src.fusion import ConfidenceAwareFusion, RetrievalFeatureEncoder
from src.config import PROMPT_TEMPLATE, LABEL_LIST, LABEL_TO_ID


def load_fusion_model(model_path: str, device: str = "cuda") -> Tuple:
    """Load trained fusion model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fusion model not found at {model_path}")
    
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
    
    logger.info(f"✓ Loaded fusion model from {model_path}")
    logger.info(f"  Beta: {fusion.beta.item():.4f}")
    
    return fusion, retrieval_encoder, config


def precompute_all_data(
    test_df: pd.DataFrame,
    knowledge_base: List[Dict],
    llm: LLMScorer,
    retriever: KnowledgeAugmentedRetriever,
    device: str = "cuda",
    batch_size: int = 8,
    llm_batch_size: int = 8,
    top_k: int = 5
) -> Dict:
    """
    Pre-compute ALL data needed for all tests in one pass.
    This is much faster than computing separately for each test.
    
    Returns:
        Dictionary with pre-computed data for all test configurations
    """
    logger.info(f"\n{'='*80}")
    logger.info("PRE-COMPUTING ALL DATA (this runs once for all 4 tests)")
    logger.info(f"{'='*80}")
    
    # Prepare data
    if "claim" in test_df.columns:
        texts = test_df["claim"].tolist()
    else:
        texts = test_df["text"].tolist()
    gold_evidences = test_df["evidence"].tolist()
    true_labels = [_normalize_label(l) for l in test_df["label"].tolist()]
    
    logger.info(f"Processing {len(texts)} samples...")
    
    # Storage for all pre-computed data
    all_retrieval_features = []
    all_retrieved_evidences = []
    all_llm_logits_retrieved = []
    all_llm_logits_gold = []
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Pre-computing data"):
        batch_texts = texts[i:i + batch_size]
        batch_gold_evidences = gold_evidences[i:i + batch_size]
        
        # 1. Retrieval (once per sample) - FAISS filtering enabled
        batch_retrieval_features = []
        batch_retrieved_evidences = []
        for t in batch_texts:
            # Note: _build_retrieval_features now uses candidate_pool_size=100 internally
            feats, retrieved_evidence = _build_retrieval_features(retriever, t, top_k)
            batch_retrieval_features.append(feats)
            batch_retrieved_evidences.append(retrieved_evidence)
        
        all_retrieval_features.extend(batch_retrieval_features)
        all_retrieved_evidences.extend(batch_retrieved_evidences)
        
        # 2. LLM logits for RETRIEVED evidence
        batch_logits_retrieved = []
        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
                for j in range(0, len(batch_texts), llm_batch_size):
                    sub_texts = batch_texts[j:j + llm_batch_size]
                    sub_evidences = batch_retrieved_evidences[j:j + llm_batch_size]
                    logits = llm.score_logits(sub_texts, sub_evidences)
                    batch_logits_retrieved.append(logits.cpu())
        
        all_llm_logits_retrieved.append(torch.cat(batch_logits_retrieved, dim=0))
        
        # 3. LLM logits for GOLD evidence
        batch_logits_gold = []
        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
                for j in range(0, len(batch_texts), llm_batch_size):
                    sub_texts = batch_texts[j:j + llm_batch_size]
                    sub_evidences = batch_gold_evidences[j:j + llm_batch_size]
                    logits = llm.score_logits(sub_texts, sub_evidences)
                    batch_logits_gold.append(logits.cpu())
        
        all_llm_logits_gold.append(torch.cat(batch_logits_gold, dim=0))
    
    # Convert to tensors
    retrieval_features_tensor = torch.tensor(np.array(all_retrieval_features), dtype=torch.float32)
    llm_logits_retrieved_tensor = torch.cat(all_llm_logits_retrieved, dim=0)
    llm_logits_gold_tensor = torch.cat(all_llm_logits_gold, dim=0)
    labels_tensor = torch.tensor(true_labels, dtype=torch.long)
    
    logger.info(f"✓ Pre-computation complete!")
    logger.info(f"  Retrieval features: {retrieval_features_tensor.shape}")
    logger.info(f"  LLM logits (retrieved): {llm_logits_retrieved_tensor.shape}")
    logger.info(f"  LLM logits (gold): {llm_logits_gold_tensor.shape}")
    
    return {
        "retrieval_features": retrieval_features_tensor,
        "llm_logits_retrieved": llm_logits_retrieved_tensor,
        "llm_logits_gold": llm_logits_gold_tensor,
        "labels": labels_tensor,
        "texts": texts,
        "true_labels": true_labels
    }


def test_fusion_fast(
    precomputed_data: Dict,
    fusion,
    retrieval_encoder,
    evidence_mode: str,
    device: str = "cuda",
    batch_size: int = 32
) -> Dict:
    """
    Fast test using pre-computed data.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing FUSION MODEL with {evidence_mode.upper()} evidence")
    logger.info(f"{'='*80}")
    
    retrieval_features = precomputed_data["retrieval_features"]
    llm_logits = (precomputed_data["llm_logits_retrieved"] if evidence_mode == "retrieved" 
                  else precomputed_data["llm_logits_gold"])
    labels = precomputed_data["labels"]
    
    all_predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(labels), batch_size), desc="Testing Fusion"):
        batch_retrieval = retrieval_features[i:i + batch_size].to(device)
        batch_logits = llm_logits[i:i + batch_size].to(device)
        
        with torch.inference_mode():
            retrieval_encoded = retrieval_encoder(batch_retrieval)
            output = fusion(batch_logits, retrieval_encoded)
            # Use final_probs for predictions (works for both binary and multi-class)
            predictions = torch.argmax(output.final_probs, dim=-1).cpu().numpy()
        
        all_predictions.extend(predictions.tolist())
    
    # Calculate metrics
    true_labels = precomputed_data["true_labels"]
    accuracy = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, all_predictions, average='macro', zero_division=0)
    
    logger.info(f"✓ Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    
    return {
        "model_type": "Fusion Model",
        "evidence_mode": evidence_mode,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": all_predictions,
        "true_labels": true_labels
    }


def test_lora_fast(
    precomputed_data: Dict,
    evidence_mode: str,
    batch_size: int = 32
) -> Dict:
    """
    Fast test using pre-computed LLM logits only (no fusion).
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing LORA LLM ONLY with {evidence_mode.upper()} evidence")
    logger.info(f"{'='*80}")
    
    llm_logits = (precomputed_data["llm_logits_retrieved"] if evidence_mode == "retrieved" 
                  else precomputed_data["llm_logits_gold"])
    labels = precomputed_data["labels"]
    
    all_predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(labels), batch_size), desc="Testing LoRA"):
        batch_logits = llm_logits[i:i + batch_size]
        
        with torch.inference_mode():
            predictions = torch.argmax(batch_logits, dim=-1).numpy()
        
        all_predictions.extend(predictions.tolist())
    
    # Calculate metrics
    true_labels = precomputed_data["true_labels"]
    accuracy = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, all_predictions, average='macro', zero_division=0)
    
    logger.info(f"✓ Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    
    return {
        "model_type": "LoRA LLM Only",
        "evidence_mode": evidence_mode,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": all_predictions,
        "true_labels": true_labels
    }


def create_comparison_table(results_list: List[Dict]) -> pd.DataFrame:
    """Create a comparison table from test results."""
    comparison_data = []
    
    for result in results_list:
        comparison_data.append({
            "Model": result["model_type"],
            "Evidence Mode": result["evidence_mode"].capitalize(),
            "Accuracy": f"{result['accuracy']:.4f}",
            "Precision": f"{result['precision']:.4f}",
            "Recall": f"{result['recall']:.4f}",
            "F1 Score": f"{result['f1_score']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def print_detailed_comparison(results_list: List[Dict]):
    """Print detailed comparison including per-class metrics."""
    logger.info(f"\n{'='*80}")
    logger.info(f"DETAILED COMPARISON")
    logger.info(f"{'='*80}\n")
    
    for result in results_list:
        logger.info(f"\n{result['model_type']} - {result['evidence_mode'].upper()} Evidence")
        logger.info("-" * 80)
        
        # Per-class metrics
        report = classification_report(
            result['true_labels'],
            result['predictions'],
            target_names=LABEL_LIST,
            digits=4,
            zero_division=0
        )
        logger.info("\n" + report)


def main():
    """Main testing function."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compare Fusion Model and LoRA LLM performance on test dataset (OPTIMIZED)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--test_data",
        type=str,
        default="/media/DATA/Python/social_media_crypto/Fake_Crypto_Claim_Detector/data/finfact.csv",
        help="Path to test dataset (CSV file with claim/text, evidence, label columns)"
    )
    parser.add_argument(
        "--fusion_model",
        type=str,
        default="artifacts/fusion_model.pt",
        help="Path to trained fusion model checkpoint"
    )
    parser.add_argument(
        "--lora_model",
        type=str,
        default="artifacts/lora_llm",
        help="Path to trained LoRA LLM model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for pre-computing (increased from 4 for speed)"
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=8,
        help="Batch size for LLM inference (increased from 4 for speed)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for fast evaluation (larger since no LLM inference)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N examples for quick testing (default: use all data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/model_comparison_results.csv",
        help="Path to save comparison results CSV"
    )
    
    args = parser.parse_args()
    
    # Configuration from args
    TEST_DATA_PATH = args.test_data
    FUSION_MODEL_PATH = args.fusion_model
    LORA_MODEL_PATH = args.lora_model
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    LLM_BATCH_SIZE = args.llm_batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size
    TOP_K = args.top_k
    SAMPLE_SIZE = args.sample
    OUTPUT_PATH = args.output
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"# OPTIMIZED MODEL COMPARISON TEST SUITE")
    logger.info(f"{'#'*80}")
    logger.info(f"Test Data: {TEST_DATA_PATH}")
    logger.info(f"Fusion Model: {FUSION_MODEL_PATH}")
    logger.info(f"LoRA Model: {LORA_MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"LLM Batch Size: {LLM_BATCH_SIZE}")
    logger.info(f"Eval Batch Size: {EVAL_BATCH_SIZE}")
    logger.info(f"Top K: {TOP_K}")
    if SAMPLE_SIZE:
        logger.info(f"Sample Size: {SAMPLE_SIZE} (quick test mode)")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"{'#'*80}\n")
    
    # Load test data
    logger.info("Loading test dataset...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Sample if requested
    if SAMPLE_SIZE and SAMPLE_SIZE < len(test_df):
        logger.info(f"🔸 Sampling {SAMPLE_SIZE} examples from {len(test_df)} total")
        test_df = test_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    
    logger.info(f"✓ Loaded {len(test_df)} test samples")
    logger.info(f"  Columns: {list(test_df.columns)}")
    
    # Build knowledge base
    from datetime import datetime, timezone
    knowledge_base = []
    for idx, row in test_df.iterrows():
        # Use incrementing timestamps to avoid all being same time
        # Make timezone-aware (UTC) to match retrieval system expectations
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        # Add idx hours to spread timestamps out
        timestamp = base_time.replace(hour=idx % 24, day=1 + (idx // 24) % 28)
        
        knowledge_base.append({
            "text": str(row["claim"]) if "claim" in test_df.columns else str(row["text"]),
            "timestamp": timestamp
        })
    
    # ==========================================
    # STEP 1: Load all models ONCE
    # ==========================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 1: Loading models (once for all tests)")
    logger.info(f"{'='*80}")
    
    # Load fusion model
    fusion, retrieval_encoder, config = load_fusion_model(FUSION_MODEL_PATH, DEVICE)
    
    # Initialize retriever with Cross-Encoder
    retriever = KnowledgeAugmentedRetriever(
        alpha=0.7,
        lambda_decay=0.1,
        gamma=0.5,
        use_query_expansion=True,
        use_cross_encoder=True  # Enable 4-stage pipeline for testing
    )
    retriever.index_documents(knowledge_base, text_field="text", timestamp_field="timestamp")
    logger.info(f"✓ Indexed {len(knowledge_base)} documents")
    
    # Initialize LLM
    llm = LLMScorer(
        model_name=LORA_MODEL_PATH,
        device=DEVICE,
        max_length=1024,
        labels=LABEL_LIST,
        prompt_template=PROMPT_TEMPLATE,
    )
    llm.model.eval()
    for p in llm.model.parameters():
        p.requires_grad_(False)
    logger.info(f"✓ Loaded LLM scorer")
    
    # ==========================================
    # STEP 2: Pre-compute ALL data ONCE
    # ==========================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 2: Pre-computing all data (retrieval + LLM logits)")
    logger.info(f"{'='*80}")
    
    precomputed_data = precompute_all_data(
        test_df=test_df,
        knowledge_base=knowledge_base,
        llm=llm,
        retriever=retriever,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        llm_batch_size=LLM_BATCH_SIZE,
        top_k=TOP_K
    )
    
    # Free LLM memory
    del llm
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # ==========================================
    # STEP 3: Run all 4 tests FAST
    # ==========================================
    logger.info(f"\n{'='*80}")
    logger.info("STEP 3: Running all 4 tests (FAST - using pre-computed data)")
    logger.info(f"{'='*80}")
    
    all_results = []
    
    # Test 1: Fusion + Retrieved
    result = test_fusion_fast(
        precomputed_data, fusion, retrieval_encoder, 
        "retrieved", DEVICE, EVAL_BATCH_SIZE
    )
    all_results.append(result)
    
    # Test 2: Fusion + Gold
    result = test_fusion_fast(
        precomputed_data, fusion, retrieval_encoder, 
        "gold", DEVICE, EVAL_BATCH_SIZE
    )
    all_results.append(result)
    
    # Test 3: LoRA + Retrieved
    result = test_lora_fast(
        precomputed_data, "retrieved", EVAL_BATCH_SIZE
    )
    all_results.append(result)
    
    # Test 4: LoRA + Gold
    result = test_lora_fast(
        precomputed_data, "gold", EVAL_BATCH_SIZE
    )
    all_results.append(result)
    
    # ==========================================
    # STEP 4: Generate report
    # ==========================================
    logger.info(f"\n\n{'#'*80}")
    logger.info(f"# FINAL COMPARISON TABLE")
    logger.info(f"{'#'*80}\n")
    
    comparison_df = create_comparison_table(all_results)
    logger.info("\n" + str(comparison_df.to_string(index=False)))
    
    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    comparison_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\n✓ Saved comparison table to: {OUTPUT_PATH}")
    
    # Print detailed comparison
    print_detailed_comparison(all_results)
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"# TEST SUITE COMPLETE")
    logger.info(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
