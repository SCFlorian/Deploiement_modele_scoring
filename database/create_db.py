# =======================
# Librairies nécessaires
# =======================
from sqlalchemy import (create_engine, Column, Integer, Float, String, DateTime, Text, ForeignKey, func, JSON)
import os
from dotenv import load_dotenv
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ========================
# Afin de charger le .env
load_dotenv()

# ==========================
# Savoir quelle BDD prendre
IS_HF = os.getenv("SPACE_ID") is not None
DB_URL = os.getenv("DATABASE_URL")

if IS_HF:
    print("Mode Hugging Face détecté — utilisation de SQLite (temporaire).")

# ==========================================================
# Utiliser /tmp (seul dossier en écriture sur Hugging Face)
    DB_DIR = "/tmp"
    DB_PATH = os.path.join(DB_DIR, "hf_temp.db")

# ==============================
# Crée le fichier si nécessaire
    if not os.path.exists(DB_PATH):
        open(DB_PATH, "a").close()

    DB_URL = f"sqlite:///{DB_PATH}"

elif DB_URL:
    print("Mode local — connexion PostgreSQL utilisée.")
else:
    raise ValueError("DATABASE_URL introuvable dans .env (nécessaire en local).")

# =====================
# Connexion SQLAlchemy
connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# =========================================
# Table des inputs d'un client selectionné 
# =========================================

class ClientInputDB(Base):
    __tablename__ = "client_input"

    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, index=True)
    features = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())

    # Relation
    predictions = relationship("PredictionResultDB", back_populates="client")
    requests = relationship("RequestLogDB", back_populates="client")

# ===================================
# Table des résultats de prédiction 
# ===================================

class PredictionResultDB(Base):
    __tablename__ = "prediction_results"
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey("client_input.id"), index=True)
    probability = Column(Float)
    decision = Column(String)
    threshold = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

    # Relation
    client = relationship("ClientInputDB", back_populates="predictions")

# ============================================
# Table de la journalisation des requêtes API
# ============================================

class RequestLogDB(Base):
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String)
    client_id = Column(Integer, ForeignKey("client_input.id"))
    user_id = Column(String, default="ml_api_user")
    latency_ms = Column(Float)
    timestamp = Column(DateTime, server_default=func.now())

    # Relation
    client = relationship("ClientInputDB", back_populates="requests")
    responses = relationship("ApiResponseDB", back_populates="request")

# =========================================
# Table de journalisation des réponses API
# =========================================

class ApiResponseDB(Base):
    __tablename__ = "api_responses"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("requests.id"))
    prediction_id = Column(Integer, ForeignKey("prediction_results.id"))
    status_code = Column(Integer)
    message = Column(String)
    timestamp = Column(DateTime, server_default=func.now())

    # Relation
    request = relationship("RequestLogDB", back_populates="responses")