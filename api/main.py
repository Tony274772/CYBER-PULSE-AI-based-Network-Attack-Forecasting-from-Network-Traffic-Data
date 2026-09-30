"""Optional FastAPI wrapper (Section 7.12).

Single endpoint ``POST /predict`` accepting a file upload, calling the same
``inference.predict.predict()`` function, returning its dict as JSON.

NOT on the demo critical path — exists to tick the "API-style interface" box.

Usage:
    uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

app = FastAPI(
    title="GT-RSSM Network Attack Forecasting API",
    description=(
        "AI-based network attack forecasting from network traffic data. "
        "Upload a PCAP or CIC-IDS2017-style CSV to get per-window infiltration "
        "probabilities, MITRE ATT&CK stage predictions, and K-step forecasts."
    ),
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "model": "GT-RSSM"}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(..., description="PCAP or CSV file"),
    k_steps: str = Query("5,10,20", description="Comma-separated forecast horizons"),
):
    """Run GT-RSSM inference on the uploaded file.

    Returns per-window infiltration probabilities, MITRE stage predictions,
    K-step imagination forecasts, KL surprise values, and attention/SHAP
    explainability data.
    """
    try:
        k_tuple = tuple(int(k.strip()) for k in k_steps.split(","))
    except ValueError:
        return JSONResponse(status_code=400,
                            content={"error": "k_steps must be comma-separated integers"})

    # Save upload to temp file
    suffix = Path(file.filename).suffix if file.filename else ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.inference.predict import predict
        result = predict(tmp_path, k_steps=k_tuple)

        # Convert any remaining numpy types for JSON serialization
        def _sanitize(obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        return JSONResponse(content=_sanitize(result))
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": str(e)})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/stages")
async def stages():
    """Return the MITRE ATT&CK stage order used by the model."""
    try:
        from src.config import Config
        cfg = Config.load("configs/default.yaml")
        return {"stage_order": cfg.stage_order}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
