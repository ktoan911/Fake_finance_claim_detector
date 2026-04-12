import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.fusion_inference import FusionClaimVerifier, _resolve_fusion_model_path

# ── Global verifier (pre-warmed at startup) ─────────────────────────────────
_verifier: FusionClaimVerifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at container startup so requests never cold-start."""
    global _verifier
    logger.info("[startup] Pre-warming FusionClaimVerifier …")
    try:
        fusion_path = _resolve_fusion_model_path(
            os.getenv("FUSION_MODEL", "models/fusion_model.pt")
        )
        _verifier = FusionClaimVerifier(
            fusion_model_path=fusion_path,
            opensearch_index=os.getenv("OPENSEARCH_INDEX_NAME")
            or os.getenv("OP_KB_NAME", "news_kb"),
            llm_model_path=os.getenv("LLM_FINETUNE"),
            retriever_model_path=os.getenv(
                "RETRIEVER_MODEL", "Qwen/Qwen3-Embedding-4B"
            ),
            device=os.getenv("DEVICE", "cpu"),
            llm_evidence_top_k=int(os.getenv("FUSION_LLM_EVIDENCE_TOP_K", "3")),
            debug=True,
        )
        logger.info("[startup] FusionClaimVerifier ready ✓")
    except Exception:
        import traceback

        logger.error(f"[startup] Failed to load verifier:\n{traceback.format_exc()}")
        # Keep _verifier = None; requests will return a clear error instead of hanging.
    yield
    # shutdown: nothing to clean up
    logger.info("[shutdown] API server stopping.")


app = FastAPI(title="Fake Crypto Claim Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClaimRequest(BaseModel):
    claim: str


@app.get("/health")
def health(request: Request):
    logger.info(f"[health] domain={request.headers.get('host', 'unknown')}")
    return {"status": "ok", "model_loaded": _verifier is not None}


@app.post("/verify")
def verify_claim(request: ClaimRequest, http_request: Request):
    if _verifier is None:
        import traceback

        return {
            "verdict": "Lỗi xử lý",
            "status": "error",
            "error": "Verifier chưa được khởi tạo (xem log startup để biết lý do).",
        }
    try:
        domain = http_request.headers.get("host", "unknown")
        logger.info(f"[verify] domain={domain} claim={request.claim!r}")
        prediction = _verifier.predict(request.claim)
        return {
            "verdict": prediction.verdict,
            "status": "success",
            "evidence": prediction.evidence,
            "source_links": prediction.source_links,
            "confidence": prediction.confidence,
        }
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        print(f"API Error: {error_traceback}", flush=True)
        return {
            "verdict": "Lỗi xử lý",
            "status": "error",
            "error": str(e),
            "traceback": error_traceback,
        }
