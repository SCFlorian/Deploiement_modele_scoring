# =======================
# Librairies nécessaires
# =======================
import pandas as pd
import joblib
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import time

# ======================
# Import des modules internes
from database.create_db import (ClientInputDB, PredictionResultDB, RequestLogDB, ApiResponseDB, SessionLocal)

# ======================
# Initialisation de la base

from database.create_db import Base, engine

print("Initialisation des tables si absentes...")
Base.metadata.create_all(bind=engine)
print("Tables prêtes")

# =========================
# Chargement des artefacts (local / HF)
# =========================

MODEL_REPO = "FlorianSC/homecredit-scoring-artifacts"
CACHE_DIR = "models_cache"

def get_artifact(path: str) -> str:
    """
    - En local / CI : charge le fichier s'il existe
    - Sur Hugging Face : télécharge depuis le Model Hub
    """
    local_path = Path(path)
    if local_path.exists():
        return str(local_path)

    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=path,
        cache_dir=CACHE_DIR
    )
# =========================
# Initialisation
# =========================
app = FastAPI(title="HomeCredit Scoring API")


# =========================
# Modèle d'entrée API
# =========================
class PredictRequest(BaseModel):
    client_id: int


# =========================
# Chargement du modèle & données
# =========================

model = joblib.load(get_artifact("pipeline_lightgbm.pkl"))
imputer = joblib.load(get_artifact("imputer_numeric.pkl"))

with open("models/expected_columns.json") as f:
    EXPECTED_COLS = json.load(f)

with open("models/imputer_columns.json") as f:
    IMPUTER_COLS = json.load(f)

THRESHOLD = float(open("models/threshold.txt").read().strip())

# Fichier contenant plusieurs clients (ton ancien example_input mais multi-lignes)
REFERENCE_DATA = pd.read_csv("data/example_input.csv")


# =========================
# Routes simples
# =========================

@app.get("/health")
def health_check():
    db = SessionLocal()
    start_time = time.perf_counter()

    response = {"status": "OK", "message": "API opérationnelle"}

    latency_ms = (time.perf_counter() - start_time) * 1000

    new_request = RequestLogDB(
        endpoint="/health",
        user_id="ml_api_user",
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc)
    )
    db.add(new_request)
    db.commit()
    db.close()

    return response

# =========================
# Fonction de préparation d'un client
# =========================
def prepare_client(client_id: int):
    row = REFERENCE_DATA[REFERENCE_DATA["SK_ID_CURR"] == client_id]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Client {client_id} introuvable"
        )

    df = row.copy()

    # aligner colonnes
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[EXPECTED_COLS]

    # colonne nécessaire à l’imputer
    if "is_train" not in df.columns:
        df["is_train"] = 0.0

    # imputation
    df[IMPUTER_COLS] = imputer.transform(df[IMPUTER_COLS])

    return df


# =========================
# Endpoint API
# =========================
@app.post("/predict")
def predict(request: PredictRequest):
    db = SessionLocal()
    start_time = time.perf_counter()

    try:
        # 1) Préparation / inférence
        df = prepare_client(request.client_id)
        proba = float(model.predict_proba(df)[0][1])
        decision = "approved" if proba > THRESHOLD else "rejected"

        # 2) Sauvegarde input
        client_db = ClientInputDB(
            client_id=request.client_id,
            features=df.to_dict(orient="records")[0]
        )
        db.add(client_db)
        db.commit()
        db.refresh(client_db)

        # 3) Sauvegarde prédiction
        prediction_db = PredictionResultDB(
            client_id=client_db.id,   # FK vers client_input.id (comme dans ton create_db)
            probability=proba,
            decision=decision,
            threshold=THRESHOLD
        )
        db.add(prediction_db)
        db.commit()
        db.refresh(prediction_db)

        # 4) Log requête
        latency_ms = (time.perf_counter() - start_time) * 1000
        req_log = RequestLogDB(
            endpoint="/predict",
            client_id=client_db.id,
            user_id="ml_api_user",
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(req_log)
        db.commit()
        db.refresh(req_log)

        # 5) Log réponse
        resp_log = ApiResponseDB(
            request_id=req_log.id,
            prediction_id=prediction_db.id,
            status_code=200,
            message=decision
        )
        db.add(resp_log)
        db.commit()

        return {
            "client_id": request.client_id,
            "probability": round(proba, 4),
            "decision": decision,
            "threshold": THRESHOLD
        }

    except HTTPException as e:
        # erreur "fonctionnelle" (client introuvable)
        latency_ms = (time.perf_counter() - start_time) * 1000

        req_log = RequestLogDB(
            endpoint="/predict",
            client_id=None,
            user_id="ml_api_user",
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(req_log)
        db.commit()
        db.refresh(req_log)

        resp_log = ApiResponseDB(
            request_id=req_log.id,
            prediction_id=None,
            status_code=e.status_code,
            message=str(e.detail)
        )
        db.add(resp_log)
        db.commit()

        raise

    except Exception as e:
        # erreur technique
        latency_ms = (time.perf_counter() - start_time) * 1000

        req_log = RequestLogDB(
            endpoint="/predict",
            client_id=None,
            user_id="ml_api_user",
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(req_log)
        db.commit()
        db.refresh(req_log)

        resp_log = ApiResponseDB(
            request_id=req_log.id,
            prediction_id=None,
            status_code=500,
            message=str(e)
        )
        db.add(resp_log)
        db.commit()

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()

# =========================
# Fonctions Gradio (wrappers)
# =========================

def gradio_predict(client_id):
    try:
        client_id = int(client_id)
        df = prepare_client(client_id)

        proba = float(model.predict_proba(df)[0][1])
        decision = "approved" if proba > THRESHOLD else "rejected"

        return {
            "client_id": client_id,
            "probability": round(proba, 4),
            "decision": decision,
            "threshold": THRESHOLD
        }
    except Exception as e:
        return {"error": str(e)}


def gradio_health():
    return health_check()


def gradio_model_info():
    return {
        "model_type": type(model).__name__,
        "threshold": THRESHOLD,
        "n_features": len(EXPECTED_COLS)
    }

columns_list = REFERENCE_DATA["SK_ID_CURR"]

# =========================
# UI GRADIO MULTI-ENDPOINTS
# =========================

with gr.Blocks(title="Home Credit Scoring – API UI") as gradio_app:

    gr.Markdown("# 📊 Home Credit Scoring API")
    gr.Markdown("### L'objectif ici est de choisir un nouveau client pour connaître sa situation")

    with gr.Tab("🔮 Prédiction"):
        client_id_input = gr.Dropdown(label="SK_ID_CURR du client (à choisir dans la liste)", choices=columns_list.to_list())
        predict_btn = gr.Button("Prédire")
        predict_output = gr.JSON()

        predict_btn.click(
            fn=gradio_predict,
            inputs=client_id_input,
            outputs=predict_output
        )

    with gr.Tab("❤️ Health Check"):
        health_btn = gr.Button("Vérifier l'état de l'API")
        health_output = gr.JSON()

        health_btn.click(
            fn=gradio_health,
            outputs=health_output
        )

    with gr.Tab("ℹ️ Infos Modèle"):
        info_btn = gr.Button("Afficher les infos du modèle")
        info_output = gr.JSON()

        info_btn.click(
            fn=gradio_model_info,
            outputs=info_output
        )

app = gr.mount_gradio_app(app, gradio_app, path="/")


# =========================
# Lancement local
# =========================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)