"""API REST FastAPI — Maintenance Prédictive Industrielle.

Permet à d'autres systèmes (ERP, supervision industrielle, appli mobile)
d'interroger le modèle de maintenance prédictive via des requêtes HTTP.
"""
from __future__ import annotations

# Lancement : uvicorn api.main:app --reload --port 8000
# Swagger   : http://localhost:8000/docs
# Endpoints : POST /predict · GET /health · GET /model-info

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Chemin racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    SensorInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    RiskLevel,
)

# ---------------------------------------------------------------------------
# Initialisation de l'application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Maintenance Prédictive Industrielle — API",
    description=(
        "API de prédiction de pannes industrielles. "
        "Envoie des données capteurs, reçois une probabilité de panne dans 24h. "
        "EFREI M1 Data Engineering — Bloc 2 RNCP40875."
    ),
    version="1.0.0",
    contact={"name": "EFREI Data Engineering", "email": "contact@efrei.net"},
)

# CORS — permet les requêtes cross-origin (dashboard, ERP, appli mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Chargement des artefacts au démarrage
# ---------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"

_model = None
_pipeline = None
_model_name = "Non chargé"
_threshold = 0.5
_feature_names = []


def load_model_artifacts():
    """Charge le modèle et le pipeline depuis le dossier models/ au démarrage.

    Utilise une variable globale pour éviter de recharger à chaque requête.
    Si les artefacts sont absents, l'API reste opérationnelle mais /predict
    retourne une erreur 503.
    """
    global _model, _pipeline, _model_name, _threshold, _feature_names

    best_model_path = MODELS_DIR / "best_model.pkl"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.pkl"

    if (MODELS_DIR / "best_model_name.pkl").exists():
        _model_name = joblib.load(MODELS_DIR / "best_model_name.pkl")

    if best_model_path.exists():
        # Le MLP PyTorch est sauvegardé avec torch.save, pas joblib
        if _model_name == "MLP (PyTorch)":
            from src.models.mlp import load_model as mlp_load
            _model = mlp_load(best_model_path)
        else:
            _model = joblib.load(best_model_path)
        print(f"[API] Modèle chargé : {_model_name}")
    else:
        print("[API] WARN: best_model.pkl introuvable. Lancez python train.py")

    if pipeline_path.exists():
        _pipeline = joblib.load(pipeline_path)
        print(f"[API] Pipeline chargé : {pipeline_path}")

    if (MODELS_DIR / "best_threshold.pkl").exists():
        _threshold = float(joblib.load(MODELS_DIR / "best_threshold.pkl"))

    if (MODELS_DIR / "feature_names.pkl").exists():
        _feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")


@app.on_event("startup")
async def startup_event():
    load_model_artifacts()


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _get_risk_level(proba: float) -> RiskLevel:
    if proba < 0.3:
        return RiskLevel.low
    elif proba < 0.6:
        return RiskLevel.moderate
    elif proba < 0.8:
        return RiskLevel.high
    else:
        return RiskLevel.critical


def _get_recommendation(risk_level: RiskLevel) -> str:
    recommendations = {
        RiskLevel.low:      "Surveillance standard. Aucune action immédiate requise.",
        RiskLevel.moderate: "Planifier une inspection préventive dans les 48h.",
        RiskLevel.high:     "Intervention urgente requise. Alerter le technicien de maintenance.",
        RiskLevel.critical: "ARRÊT MACHINE RECOMMANDÉ. Intervention immédiate nécessaire.",
    }
    return recommendations[risk_level]


def _predict(sensor_input: SensorInput) -> tuple[float, int]:
    """Effectue la prédiction à partir des données capteurs validées.

    Args:
        sensor_input: Données capteurs validées par Pydantic.

    Returns:
        Tuple (probabilité, classe_prédite).
    """
    input_dict = {
        "machine_type":           sensor_input.machine_type.value,
        "operating_mode":         sensor_input.operating_mode.value,
        "vibration_rms":          sensor_input.vibration_rms,
        "temperature_motor":      sensor_input.temperature_motor,
        "current_phase_avg":      sensor_input.current_phase_avg,
        "pressure_level":         sensor_input.pressure_level,
        "rpm":                    sensor_input.rpm,
        "hours_since_maintenance": sensor_input.hours_since_maintenance,
        "ambient_temp":           sensor_input.ambient_temp,
    }

    df_input = pd.DataFrame([input_dict])
    X_transformed = _pipeline.transform(df_input)

    model_type = type(_model).__name__

    if model_type == "MaintenanceMLP":
        import torch
        _model.eval()
        X_tensor = torch.from_numpy(X_transformed.astype("float32"))
        with torch.no_grad():
            logit = _model(X_tensor)
            proba = torch.sigmoid(logit).item()
    else:
        proba = float(_model.predict_proba(X_transformed)[0, 1])

    predicted_class = int(proba >= _threshold)
    return proba, predicted_class


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédiction de panne à partir des données capteurs",
    description=(
        "Reçoit les valeurs de capteurs d'une machine industrielle et retourne "
        "la probabilité de panne dans les 24 prochaines heures, ainsi qu'une "
        "recommandation d'action pour le responsable maintenance."
    ),
    responses={
        503: {"description": "Modèle non disponible — lancer python train.py"},
        422: {"description": "Données invalides — champs manquants ou hors plage"},
    },
)
async def predict(sensor_input: SensorInput) -> PredictionResponse:
    """Endpoint principal : prédiction de panne industrielle.

    Args:
        sensor_input: Données capteurs validées automatiquement par Pydantic.

    Returns:
        PredictionResponse avec probabilité, classe, niveau de risque et recommandation.

    Raises:
        HTTPException 503: Si le modèle n'est pas chargé.
        HTTPException 500: En cas d'erreur interne de prédiction.
    """
    if _model is None or _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Lancez d'abord : python train.py",
        )

    try:
        proba, predicted_class = _predict(sensor_input)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}",
        )

    risk_level = _get_risk_level(proba)
    recommendation = _get_recommendation(risk_level)

    return PredictionResponse(
        failure_probability=round(proba, 4),
        predicted_class=predicted_class,
        risk_level=risk_level,
        threshold_used=_threshold,
        model_name=_model_name,
        recommendation=recommendation,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="État du service",
    description="Vérifie que l'API est opérationnelle et que le modèle est chargé.",
)
async def health() -> HealthResponse:
    """Endpoint de santé — utilisé par les systèmes de monitoring et CI/CD."""
    model_loaded = _model is not None and _pipeline is not None

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model=_model_name,
        version="1.0",
        model_loaded=model_loaded,
        threshold=_threshold,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Informations sur le modèle",
    description="Retourne les features attendues, leurs plages de valeurs, et les métadonnées du modèle.",
)
async def model_info() -> ModelInfoResponse:
    """Informations sur le modèle et les features — utile pour intégration client."""
    feature_ranges = {
        "vibration_rms":          {"min": 0.35, "max": 10.0, "unit": "m/s²"},
        "temperature_motor":      {"min": 28.0, "max": 95.0, "unit": "°C"},
        "current_phase_avg":      {"min": 0.0,  "max": 20.0, "unit": "A"},
        "pressure_level":         {"min": 0.0,  "max": 50.0, "unit": "bar"},
        "rpm":                    {"min": 0.0,  "max": 3000.0, "unit": "RPM"},
        "hours_since_maintenance":{"min": 0.0,  "max": 600.0, "unit": "heures"},
        "ambient_temp":           {"min": 8.0,  "max": 18.0, "unit": "°C"},
        "machine_type":           {"values": ["CNC", "Pump", "Compressor", "Robotic Arm"]},
        "operating_mode":         {"values": ["idle", "normal", "peak"]},
    }

    return ModelInfoResponse(
        model_name=_model_name,
        features=_feature_names if _feature_names else list(feature_ranges.keys()),
        feature_ranges=feature_ranges,
    )
