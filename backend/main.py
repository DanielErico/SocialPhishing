"""
main.py
=======
FastAPI backend for the SocialPhishing URL classifier.

Endpoints
---------
POST /classify  -- Classify a URL as Safe, Suspicious, or Phishing
GET  /health    -- Health check / liveness probe
GET  /          -- Root info response

Label Logic
-----------
The underlying model is a binary Random Forest (Phishing vs Safe).
A three-tier label is applied at inference time using confidence thresholds:

  P(Phishing) >= 0.70          -->  "Phishing"    (high confidence threat)
  P(Phishing) in [0.40, 0.70)  -->  "Suspicious"  (uncertain, proceed with caution)
  P(Phishing) < 0.40           -->  "Safe"         (likely legitimate)

Rate Limiting
-------------
Maximum 100 requests per minute per IP address (via slowapi).
Clients that exceed the limit receive HTTP 429 Too Many Requests.

Security
--------
- No request data is stored persistently (in-memory only during request lifecycle)
- CORS is configured to allow Chrome extension origins
- Input URLs are validated before processing

Usage (local)
-------------
    cd url-classifier/backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

    # Test:
    curl -X POST http://localhost:8000/classify \
         -H "Content-Type: application/json" \
         -d '{"url": "https://login-verify.suspicious-bank.com/account/update"}'
"""

import io
import os
import sys
import time
import logging
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.feature_extractor import extract_features, FEATURE_COLUMNS, features_to_vector

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("socialphishing")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# The model is looked up in this priority order:
#   1. MODEL_PATH environment variable (set on Render)
#   2. ../model/phishing_model.pkl  (local development)
_default_model_path = Path(__file__).parent.parent / "model" / "phishing_model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_default_model_path)))

# Global model artifact loaded at startup
_model_artifact: dict | None = None


def load_model() -> dict:
    """Load and return the joblib model artifact. Raises RuntimeError if not found."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Set the MODEL_PATH environment variable or run train_model.py first."
        )
    logger.info(f"Loading model from: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    logger.info(
        f"Model loaded | type={type(artifact['classifier']).__name__} "
        f"| stored_accuracy={artifact.get('accuracy', 'N/A')}"
    )
    return artifact


# ---------------------------------------------------------------------------
# Confidence thresholds for three-tier labelling
# ---------------------------------------------------------------------------

PHISHING_THRESHOLD = 0.70   # P(Phishing) >= this  -->  Phishing
SUSPICIOUS_THRESHOLD = 0.40 # P(Phishing) >= this  -->  Suspicious
                             # P(Phishing) <  this  -->  Safe


def _confidence_to_label(phishing_prob: float) -> tuple[str, float]:
    """
    Convert a raw P(Phishing) score to a human-readable label and a
    normalised confidence value (0.0 - 1.0) representing certainty.

    Returns
    -------
    (label, confidence)
        label      : "Phishing" | "Suspicious" | "Safe"
        confidence : probability rounded to 4 d.p.
    """
    p = float(phishing_prob)
    if p >= PHISHING_THRESHOLD:
        return "Phishing", round(p, 4)
    elif p >= SUSPICIOUS_THRESHOLD:
        return "Suspicious", round(p, 4)
    else:
        # For Safe URLs, report confidence as P(Safe) = 1 - P(Phishing)
        return "Safe", round(1.0 - p, 4)


# ---------------------------------------------------------------------------
# URL validation helpers
# ---------------------------------------------------------------------------

def _is_valid_url(url: str) -> bool:
    """
    Return True if *url* has a parseable scheme and netloc.
    Accepts http:// and https:// only (no data:, javascript:, etc.).
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rate limiter (slowapi)
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SocialPhishing URL Classifier API",
    description=(
        "Classifies URLs as Safe, Suspicious, or Phishing using a "
        "Random Forest model trained on PhishTank data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS -- allow Chrome extension origins and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1",
        # Chrome extensions send requests from chrome-extension:// origins
        # The wildcard below allows all; restrict to specific extension ID in production
        "*",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load the ML model into memory when the server starts."""
    global _model_artifact
    try:
        _model_artifact = load_model()
        logger.info("Startup complete -- model is ready.")
    except RuntimeError as exc:
        logger.error(f"STARTUP FAILED: {exc}")
        # Allow the server to start but return 503 on classify requests
        _model_artifact = None


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down SocialPhishing API.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class HtmlFeatures(BaseModel):
    """HTML structural features extracted locally by the client browser."""
    has_password_input: int
    has_form: int
    external_links_ratio: float
    brand_mismatch: int

class ClassifyRequest(BaseModel):
    """Request body for POST /classify."""
    url: str
    html_features: HtmlFeatures | None = None

    @field_validator("url")
    @classmethod
    def url_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("url must not be empty")
        return v.strip()


class ClassifyResponse(BaseModel):
    """Response body for POST /classify."""
    label: str          # "Safe" | "Suspicious" | "Phishing"
    confidence: float   # 0.0 - 1.0 (certainty of the assigned label)
    phishing_probability: float  # Raw P(Phishing) from the model
    features: dict      # The 8 extracted features (for transparency / debugging)
    latency_ms: float   # Server-side processing time in milliseconds


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint -- returns API info and status."""
    model_ready = _model_artifact is not None
    return {
        "service": "SocialPhishing URL Classifier API",
        "version": "1.0.0",
        "status": "ready" if model_ready else "model_not_loaded",
        "endpoints": {
            "classify": "POST /classify",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """
    Liveness probe for deployment platforms (Render, Railway, etc.).
    Returns HTTP 200 if the model is loaded and ready.
    Returns HTTP 503 if the model failed to load at startup.
    """
    if _model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )
    return {
        "status": "healthy",
        "model_accuracy": _model_artifact.get("accuracy"),
        "model_classes": _model_artifact.get("class_names"),
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
@limiter.limit("100/minute")
async def classify_url(request: Request, body: ClassifyRequest):
    """
    Classify a URL as Safe, Suspicious, or Phishing.

    Parameters
    ----------
    body : ClassifyRequest
        JSON body with a single field: ``url`` (string, required)

    Returns
    -------
    ClassifyResponse
        - ``label``               : "Safe" | "Suspicious" | "Phishing"
        - ``confidence``          : 0.0 - 1.0 certainty of the assigned label
        - ``phishing_probability``: raw P(Phishing) from the Random Forest
        - ``features``            : dict of the 8 extracted URL features
        - ``latency_ms``          : server processing time in milliseconds

    Raises
    ------
    HTTP 400  -- URL is empty, malformed, or has an unsupported scheme
    HTTP 429  -- Rate limit exceeded (>100 requests/minute from this IP)
    HTTP 503  -- Model not loaded (startup failure)
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Guard: model must be loaded
    # ------------------------------------------------------------------
    if _model_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. The server may be starting up."
        )

    # ------------------------------------------------------------------
    # 2. Input validation
    # ------------------------------------------------------------------
    url = body.url

    if not _is_valid_url(url):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid URL: '{url}'. "
                "URL must start with http:// or https:// and contain a valid hostname."
            ),
        )

    # ------------------------------------------------------------------
    # 2b. Whitelist check (Industry standard to prevent false positives on trusted sites)
    # ------------------------------------------------------------------
    WHITELIST_DOMAINS = {
        "google.com", "google.co.uk", "bing.com", "yahoo.com", "duckduckgo.com",
        "amazon.com", "amazon.co.uk", "ebay.com", "etsy.com", "shopify.com",
        "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
        "reddit.com", "pinterest.com", "tiktok.com", "github.com",
        "stackoverflow.com", "microsoft.com", "apple.com", "wikipedia.org",
        "d-internconnect.com", "in-hub.d-internconnect.com"
    }

    try:
        parsed_url = urlparse(url)
        hostname = (parsed_url.hostname or "").lower()
        base_hostname = hostname[4:] if hostname.startswith("www.") else hostname
        
        if hostname in WHITELIST_DOMAINS or base_hostname in WHITELIST_DOMAINS:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            logger.info(f"classify | whitelisted | hostname={hostname} | latency={latency_ms}ms")
            return ClassifyResponse(
                label="Safe",
                confidence=1.0,
                phishing_probability=0.0,
                features=extract_features(url),
                latency_ms=latency_ms,
            )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------------
    features = extract_features(url)
    feature_vector = np.array([features_to_vector(features)], dtype=float)

    # ------------------------------------------------------------------
    # 4. Model inference
    # ------------------------------------------------------------------
    clf = _model_artifact["classifier"]
    le = _model_artifact["label_encoder"]

    # If the model has a scaler, scale the input vector
    if "scaler" in _model_artifact:
        feature_vector = _model_artifact["scaler"].transform(feature_vector)

    # predict_proba returns shape (1, n_classes)
    # Class order is determined by LabelEncoder (alphabetical: Phishing=0, Safe=1)
    proba = clf.predict_proba(feature_vector)[0]

    # P(Phishing) is at index 0 (alphabetically first)
    class_names = _model_artifact["class_names"]  # ['Phishing', 'Safe']
    phishing_idx = class_names.index("Phishing")
    phishing_prob = float(proba[phishing_idx])

    # ------------------------------------------------------------------
    # 5. Apply Decision Fusion Heuristics (if HTML features are provided)
    # ------------------------------------------------------------------
    adjusted_prob = phishing_prob
    if body.html_features is not None:
        hf = body.html_features
        # Heuristic 1: If no password inputs and no form elements exist, significantly reduce risk
        if hf.has_password_input == 0 and hf.has_form == 0:
            adjusted_prob -= 0.35
        
        # Heuristic 2: If there is a password form AND a brand mismatch in the title, promote risk
        if hf.brand_mismatch == 1 and hf.has_password_input == 1:
            adjusted_prob += 0.35
            
        # Heuristic 3: If a high percentage of page assets are loaded from external sites (possible spoofing)
        if hf.external_links_ratio > 0.60:
            adjusted_prob += 0.15
        elif hf.external_links_ratio < 0.10:
            adjusted_prob -= 0.10
            
        # Clamp probability to range [0.0, 1.0]
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))
        logger.info(f"classify | decision fusion | original={phishing_prob:.4f} | adjusted={adjusted_prob:.4f}")

    # ------------------------------------------------------------------
    # 6. Apply three-tier label logic
    # ------------------------------------------------------------------
    label, confidence = _confidence_to_label(adjusted_prob)

    # ------------------------------------------------------------------
    # 7. Logging (no URL data is persisted -- log only the label)
    # ------------------------------------------------------------------
    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(
        f"classify | label={label} | confidence={confidence:.4f} "
        f"| p_phishing={adjusted_prob:.4f} | latency={latency_ms}ms"
    )

    return ClassifyResponse(
        label=label,
        confidence=confidence,
        phishing_probability=round(adjusted_prob, 4),
        features=features,
        latency_ms=latency_ms,
    )
