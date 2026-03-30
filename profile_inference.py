import time

from src.fusion_inference import FusionClaimVerifier

# We will patch perf_counter to explicitly track load times
load_times = {}

print("====================================")
print("    MODEL LOADING SPEED PROFILE     ")
print("====================================")

t0 = time.time()
print("1. Initializing FusionClaimVerifier ... (This loads ALL models)")

verifier = FusionClaimVerifier(
    fusion_model_path="artifacts/fusion_model.pt",
    opensearch_index="news_kb",  # OPENSEARCH_INDEX_NAME
    llm_model_path="artifacts/lora_llm",
    retriever_model_path="AITeamVN/Vietnamese_Embedding",
    device="cpu",  # Test full CPU
    llm_evidence_top_k=5,
    debug=True,
)
init_time = time.time() - t0
print(f"[LOAD] FusionClaimVerifier Init Time: {init_time:.2f} seconds")

# Explicitly measure components for one prediction
print("\n====================================")
print("    INFERENCE SPEED PROFILE         ")
print("====================================")

claim_text = "giá vàng sẽ tăng 200% vào ngày mai"

t_start = time.time()

# 1. Query Expansion & Retrieval
print("\n1. Running Retrieval (BM25 + Semantic Vector Search)")
t_retr_0 = time.time()
docs = verifier.retriever.retrieve(claim_text, top_k=verifier.top_k)
t_retr_1 = time.time()
print(f"   [INFER] Retrieval Time: {t_retr_1 - t_retr_0:.2f} seconds")

# 2. LLM Inference
print("\n2. Running LLM Inference (Prompt + Context -> Logits)")
evidences = [d.text for d in docs[: verifier.llm_evidence_top_k]]
t_llm_0 = time.time()
import torch

with torch.inference_mode():
    llm_logits = verifier.llm.score_logits([claim_text], [evidences]).to(
        verifier.device
    )
t_llm_1 = time.time()
print(f"   [INFER] LLM Inference Time: {t_llm_1 - t_llm_0:.2f} seconds")

# 3. Fusion Layer
print("\n3. Running Fusion Layer")
from src.fusion_inference import _build_retrieval_features_train_compatible

# We rebuild the features just for fusion measurement
retrieval_features_np, _, _ = _build_retrieval_features_train_compatible(
    verifier.retriever, claim_text, verifier.top_k
)
t_fuse_0 = time.time()
with torch.inference_mode():
    ret_feat = torch.tensor(
        retrieval_features_np, dtype=torch.float32, device=verifier.device
    ).unsqueeze(0)
    retrieval_encoded = verifier.retrieval_encoder(ret_feat)
    fusion_output = verifier.fusion(llm_logits, retrieval_encoded)
t_fuse_1 = time.time()
print(f"   [INFER] Fusion Time: {t_fuse_1 - t_fuse_0:.2f} seconds")

print("\n====================================")
print(f"TOTAL PREDICTION TIME: {t_fuse_1 - t_start:.2f} seconds")
print("====================================\n")
