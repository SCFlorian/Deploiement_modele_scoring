import pandas as pd
import joblib
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download

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
@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API opérationnelle"}


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

    df = prepare_client(request.client_id)

    proba = float(model.predict_proba(df)[0][1])
    decision = "approved" if proba > THRESHOLD else "rejected"

    return {
        "client_id": request.client_id,
        "probability": round(proba, 4),
        "decision": decision,
        "threshold": THRESHOLD
    }


# =========================
# UI GRADIO
# =========================
def gradio_predict(client_id):
    try:
        client_id = int(client_id)
        df = prepare_client(client_id)

        proba = float(model.predict_proba(df)[0][1])
        decision = "approved" if proba > THRESHOLD else "rejected"

        return f"Client: {client_id}\nProbabilité: {proba:.4f}\nDécision: {decision}"
    except Exception as e:
        return f"Erreur : {str(e)}"


gradio_app = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Number(label="SK_ID_CURR du client"),
    outputs="text",
    title="Home Credit Scoring – UI"
)

app = gr.mount_gradio_app(app, gradio_app, path="/ui")


# =========================
# Lancement local
# =========================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)