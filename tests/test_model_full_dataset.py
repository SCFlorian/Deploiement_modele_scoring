# ====================================================
# L'objectif ici va être de tester le modèle lui-même
# ====================================================

import os
import json
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# ==== Chemins de base ====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

EXAMPLE_CSV_PATH = os.path.join(BASE_DIR, "data", "example_input.csv")
COLS_PATH = os.path.join(BASE_DIR, "models", "expected_columns.json")
IMPUTER_COLS_PATH = os.path.join(BASE_DIR, "models", "imputer_columns.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

# ==== Repo HF des artefacts ====
MODEL_REPO = "FlorianSC/homecredit-scoring-artifacts"


def get_artifact(filename: str) -> str:
    """
    Télécharge l'artefact depuis Hugging Face
    (cache automatique en local / CI)
    """
    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename
    )


def test_model_full_dataset():
    """
    Reproduit exactement la logique de l'API,
    en testant le modèle directement sur un CSV utilisateur.
    """

    # === Chargement des objets ===
    model = joblib.load(get_artifact("pipeline_lightgbm.pkl"))
    imputer = joblib.load(get_artifact("imputer_numeric.pkl"))

    with open(COLS_PATH, "r") as f:
        EXPECTED_COLS = json.load(f)

    with open(IMPUTER_COLS_PATH, "r") as f:
        IMPUTER_COLS = json.load(f)

    THRESHOLD = float(open(THRESHOLD_PATH).read().strip())

    # === Chargement du CSV ===
    df = pd.read_csv(EXAMPLE_CSV_PATH)

    assert df.shape[0] != 0, "Le fichier test doit contenir des lignes"
    assert df.dtypes.apply(
        lambda t: t in ["float64", "int64"]
    ).all(), "Toutes les variables doivent être numériques"

    # === Alignement colonnes ===
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[EXPECTED_COLS]

    # === Fix imputer ===
    if "is_train" not in df.columns:
        df["is_train"] = 0.0

    # === Imputation ===
    df[IMPUTER_COLS] = imputer.transform(df[IMPUTER_COLS])

    # === Prédiction ===
    proba = model.predict_proba(df)[0][1]
    decision = "approved" if proba > THRESHOLD else "rejected"

    # === Assertions ===
    assert 0.0 <= proba <= 1.0, "Proba hors bornes (0-1)"
    assert decision in ["approved", "rejected"], "Décision incorrecte"