import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Phải thêm đường dẫn project vào sys.path để có thể import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.fusion_inference import verify_claim_true_false

app = FastAPI(title="Fake Crypto Claim Detector API")

# Cấu hình CORS để Extension (background script) có thể gọi được API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClaimRequest(BaseModel):
    claim: str


@app.post("/verify")
def verify_claim(request: ClaimRequest):
    try:
        verdict = verify_claim_true_false(
            claim=request.claim,
            fusion_model_path="models/fusion_model.pt",
            llm_model_path="models/lora_llm",
            retriever_model_path="AITeamVN/Vietnamese_Embedding",
            opensearch_index="news_kb",
            device="cuda",
            llm_evidence_top_k=3,
            debug=False,
        )
        return {"verdict": verdict, "status": "success"}
    except Exception as e:
        return {"verdict": "Lỗi xử lý", "status": "error", "error": str(e)}
