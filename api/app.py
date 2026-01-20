from __future__ import annotations

import math
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_logreg.joblib"

# Monitoring / request logging (kept outside git via .gitignore)
MONITORING_DATA = PROJECT_ROOT / "monitoring" / "data"
MONITORING_DATA.mkdir(parents=True, exist_ok=True)
REQUEST_LOG_PARQUET = MONITORING_DATA / "api_requests.parquet"
REQUEST_LOG_CSV = MONITORING_DATA / "api_requests.csv"

app = FastAPI(
    title="FinTech Fraud Detection Platform API",
    version="1.0.0",
    description="Real-time scoring API for an imbalanced fraud detection model (LogReg pipeline).",
)

# Loaded on startup
model = None
FEATURES: List[str] = []


# -----------------------------
# Schemas
# -----------------------------
class PredictRequest(BaseModel):
    # Option A: provide vector in correct feature order
    vector: Optional[List[float]] = Field(
        default=None,
        description="Feature vector in the exact order returned by /schema/features",
        example=[0.0] * 31,
    )

    # Option B: provide named features (order not required)
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature mapping by name (keys must match /schema/features)",
        example={
            "V1": 0.0,
            "V2": 0.0,
            "Time": 0.0,
            "Amount": 0.0,
            "Amount_log": 0.0,
        },
    )

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    proba_fraud: float
    predicted_label: int
    threshold: float
    features_used: List[str]


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: Any) -> float:
    """Defensive conversion (handles ints, floats, numpy types)."""
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            raise ValueError("NaN/Inf not allowed")
        return xf
    except Exception as e:
        raise ValueError(f"Invalid numeric value: {x!r}") from e


def build_dataframe_from_request(req: PredictRequest) -> pd.DataFrame:
    """
    Build a single-row DataFrame with columns exactly matching FEATURES order.
    Accepts either vector (ordered) or features dict (named).
    """
    if req.vector is None and req.features is None:
        raise HTTPException(status_code=400, detail="Provide either 'vector' or 'features'.")

    if req.vector is not None and req.features is not None:
        raise HTTPException(status_code=400, detail="Provide only one of 'vector' or 'features', not both.")

    if req.vector is not None:
        if len(req.vector) != len(FEATURES):
            raise HTTPException(
                status_code=400,
                detail=f"Vector length {len(req.vector)} does not match expected {len(FEATURES)}.",
            )
        row = [_safe_float(v) for v in req.vector]
        return pd.DataFrame([row], columns=FEATURES)

    feat_map = req.features or {}
    missing = [f for f in FEATURES if f not in feat_map]
    extra = [k for k in feat_map.keys() if k not in FEATURES]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    if extra:
        raise HTTPException(status_code=400, detail=f"Unexpected features: {extra}")

    row = [_safe_float(feat_map[f]) for f in FEATURES]
    return pd.DataFrame([row], columns=FEATURES)


def log_request(
    x_df: pd.DataFrame,
    threshold: float,
    proba: float,
    predicted_label: int,
) -> None:
    """
    Append a single prediction request to monitoring logs.

    - Prefers parquet (single growing table).
    - Falls back to CSV append if parquet engine isn't available.
    - Never raises (API must not break on logging).
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        row = {
            "timestamp_utc": ts,
            "threshold": float(threshold),
            "proba_fraud": float(proba),
            "predicted_label": int(predicted_label),
        }

        # Add feature columns
        # x_df has one row, with columns == FEATURES
        for f in FEATURES:
            row[f] = float(x_df.iloc[0][f])

        df_row = pd.DataFrame([row])

        # Prefer parquet
        try:
            if REQUEST_LOG_PARQUET.exists():
                df_old = pd.read_parquet(REQUEST_LOG_PARQUET)
                df_new = pd.concat([df_old, df_row], ignore_index=True)
            else:
                df_new = df_row
            df_new.to_parquet(REQUEST_LOG_PARQUET, index=False)
        except Exception:
            # CSV fallback (append mode)
            header = not REQUEST_LOG_CSV.exists()
            df_row.to_csv(REQUEST_LOG_CSV, mode="a", header=header, index=False)

    except Exception:
        # Never break scoring if logging fails
        return


# -----------------------------
# Startup: load model + feature names
# -----------------------------
@app.on_event("startup")
def load_model() -> None:
    global model, FEATURES

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # Pull feature order directly from trained pipeline
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        n = getattr(model, "n_features_in_", None)
        if n is None:
            raise RuntimeError("Model has no feature_names_in_ or n_features_in_. Cannot determine schema.")
        FEATURES = [f"f{i}" for i in range(int(n))]
    else:
        FEATURES = [str(x) for x in list(names)]


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "num_features": len(FEATURES),
        "request_log_parquet": str(REQUEST_LOG_PARQUET),
        "request_log_csv": str(REQUEST_LOG_CSV),
    }


@app.get("/schema/features")
def schema_features() -> Dict[str, Any]:
    return {"features": FEATURES, "count": len(FEATURES)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    x_df = build_dataframe_from_request(req)

    # Predict probability of fraud (class 1)
    try:
        proba = float(model.predict_proba(x_df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    label = int(proba >= req.threshold)

    # Monitoring log (does not affect response)
    log_request(x_df=x_df, threshold=req.threshold, proba=proba, predicted_label=label)

    return PredictResponse(
        proba_fraud=proba,
        predicted_label=label,
        threshold=req.threshold,
        features_used=FEATURES,
    )
