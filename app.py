# =======================
# Librairies nécessaires
# =======================
import pandas as pd
import joblib
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import gradio as gr

from huggingface_hub import hf_hub_download
from sqlalchemy.orm import Session


# ======================
# Import des modules internes
# ======================
from database.create_db import (
    ClientInputDB,
    PredictionResultDB,
    RequestLogDB,
    ApiResponseDB,
    SessionLocal,
    Base,
    engine
)

# ======================
# Logger JSON structuré
# ======================
logger = logging.getLogger("ml_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

# ======================
# Initialisation de la base
# ======================
print("Initialisation des tables si absentes...")
Base.metadata.create_all(bind=engine)
print("Tables prêtes")

# =========================
# Chargement des artefacts (local / HF)
# =========================
MODEL_REPO = "FlorianSC/homecredit-scoring-artifacts"
CACHE_DIR = "models_cache"

def get_artifact(path: str) -> str:
    local_path = Path(path)
    if local_path.exists():
        return str(local_path)
    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=path,
        cache_dir=CACHE_DIR
    )

# =========================
# Initialisation FastAPI
# =========================
app = FastAPI(title="HomeCredit Scoring API")

# =========================
# Modèle d'entrée API
# =========================
class PredictRequest(BaseModel):
    client_id: int

# =========================
# Chargement modèle & données
# =========================
model = joblib.load(get_artifact("pipeline_lightgbm.pkl"))
imputer = joblib.load(get_artifact("imputer_numeric.pkl"))

with open("models/expected_columns.json") as f:
    EXPECTED_COLS = json.load(f)

with open("models/imputer_columns.json") as f:
    IMPUTER_COLS = json.load(f)

THRESHOLD = float(open("models/threshold.txt").read().strip())
REFERENCE_DATA = pd.read_csv("data/example_input.csv")

# =========================
# Endpoint Health
# =========================
@app.get("/health")
def health_check():
    db = SessionLocal()
    start_time = time.perf_counter()

    latency_ms = (time.perf_counter() - start_time) * 1000

    db.add(RequestLogDB(
        endpoint="/health",
        user_id="ml_api_user",
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc)
    ))
    db.commit()
    db.close()

    logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "health_check",
        "endpoint": "/health",
        "latency_ms": latency_ms,
        "status_code": 200,
        "error": None
    }))

    return {"status": "OK", "message": "API opérationnelle"}

# =========================
# Préparation client
# =========================
def prepare_client(client_id: int) -> pd.DataFrame:
    row = REFERENCE_DATA[REFERENCE_DATA["SK_ID_CURR"] == client_id]
    if row.empty:
        raise HTTPException(404, f"Client {client_id} introuvable")

    df = row.copy()
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[EXPECTED_COLS]

    if "is_train" not in df.columns:
        df["is_train"] = 0.0

    df[IMPUTER_COLS] = imputer.transform(df[IMPUTER_COLS])
    return df

# =========================
# Endpoint Predict
# =========================
@app.post("/predict")
def predict(request: PredictRequest):
    db = SessionLocal()
    start_time = time.perf_counter()
    cpu_start = time.process_time()

    try:
        # Préparation
        df = prepare_client(request.client_id)

        # Inference pure
        infer_start = time.perf_counter()
        proba = float(model.predict_proba(df)[0][1])
        inference_ms = (time.perf_counter() - infer_start) * 1000

        decision = "approved" if proba > THRESHOLD else "rejected"

        # Sauvegarde input
        client_db = ClientInputDB(
            client_id=request.client_id,
            features=df.to_dict(orient="records")[0]
        )
        db.add(client_db)
        db.commit()
        db.refresh(client_db)

        # Sauvegarde prediction
        prediction_db = PredictionResultDB(
            client_input_id=client_db.id,
            client_id=request.client_id,
            probability=proba,
            decision=decision,
            threshold=THRESHOLD
        )
        db.add(prediction_db)
        db.commit()
        db.refresh(prediction_db)

        # Log requête
        latency_ms = (time.perf_counter() - start_time) * 1000
        cpu_time_ms = (time.process_time() - cpu_start) * 1000
        req_log = RequestLogDB(
            endpoint="/predict",
            client_input_id=client_db.id,
            client_id=request.client_id,
            user_id="ml_api_user",
            latency_ms=latency_ms,
            inference_ms=inference_ms,
            cpu_time_ms=cpu_time_ms,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(req_log)
        db.commit()
        db.refresh(req_log)

        # Log réponse
        db.add(ApiResponseDB(
            request_id=req_log.id,
            prediction_id=prediction_db.id,
            status_code=200,
            message=decision
        ))
        db.commit()

        # Logger JSON
        logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "predict",
            "endpoint": "/predict",
            "client_id": request.client_id,
            "probability": round(proba, 4),
            "decision": decision,
            "latency_ms": latency_ms,
            "inference_ms": inference_ms,
            "cpu_time_ms": cpu_time_ms,
            "status_code": 200,
            "error": None
        }))

        return {
            "client_id": request.client_id,
            "probability": round(proba, 4),
            "decision": decision,
            "threshold": THRESHOLD
        }

    except Exception as e:
        logger.error(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "predict_error",
            "endpoint": "/predict",
            "client_id": request.client_id,
            "status_code": 500,
            "error": str(e)
        }))
        raise

    finally:
        db.close()

# =========================
# Fonctions Gradio
# =========================
def gradio_predict(client_id):
    return predict(PredictRequest(client_id=int(client_id)))

def gradio_health():
    return health_check()

def gradio_model_info():
    return {
        "model_type": type(model).__name__,
        "threshold": THRESHOLD,
        "n_features": len(EXPECTED_COLS)
    }

# =========================
# UI Gradio
# =========================
columns_list = REFERENCE_DATA["SK_ID_CURR"]

with gr.Blocks(title="Home Credit Scoring – API UI") as gradio_app:
    gr.Markdown("# 📊 Home Credit Scoring API")

    with gr.Tab("🔮 Prédiction"):
        client_id_input = gr.Dropdown(
            label="SK_ID_CURR du client",
            choices=columns_list.to_list()
        )
        predict_btn = gr.Button("Prédire")
        predict_output = gr.JSON()
        predict_btn.click(gradio_predict, client_id_input, predict_output)

    with gr.Tab("❤️ Health Check"):
        gr.Button("Vérifier").click(gradio_health, outputs=gr.JSON())

    with gr.Tab("ℹ️ Infos Modèle"):
        gr.Button("Infos").click(gradio_model_info, outputs=gr.JSON())

app = gr.mount_gradio_app(app, gradio_app, path="/")

# =========================
# Lancement local
# =========================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)