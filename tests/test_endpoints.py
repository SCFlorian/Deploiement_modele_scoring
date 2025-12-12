# ===========================================================
# L'objectif ici va être de tester les différents endpoints
# ===========================================================

# Ce dont nous avons besoin
from fastapi.testclient import TestClient
import pytest
from app import app
import pandas as pd
import os
import io

# Classe pour tester l'application FastAPI
client = TestClient(app)

# Détection du chemin pour récupérer le fichier csv
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EXEMPLE_CSV_PATH = os.path.join(BASE_DIR,"data", "example_input.csv")

# ==========================
# Test 1 - Endpoint /health

def test_health_endpoint():
    """ Vérifie que le endpoint /health renvoie bien 200 et un message OK """
    response = client.get("/health")
    assert response.status_code == 200, "Le endpoint /health doit répondre 200"
    data = response.json()
    assert "status" in data, "Les JSON doit contenir 'status'"
    assert data["status"].lower() == "ok"

# ==========================
# Test 2 - Endpoint /predict

def test_predict_endpoint():
    # Charger un client existant depuis le fichier de référence
    df = pd.read_csv("data/example_input.csv")
    test_client_id = int(df["SK_ID_CURR"].iloc[0])

    # Appel au endpoint
    response = client.post("/predict", json={"client_id": test_client_id})

    assert response.status_code == 200, "Le endpoint /predict doit répondre 200"
    result = response.json()
    assert "probability" in result, "La réponse doit contenir la clé 'probability'"
    assert 0 <= result["probability"] <=1 , "La probabilté doit être entre 0 et 1"
