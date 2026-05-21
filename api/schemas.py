"""Schémas Pydantic pour la validation des données de l'API FastAPI.

Pydantic v2 valide automatiquement les types, les plages, et les valeurs
autorisées — retourne un 422 Unprocessable Entity avec message clair si invalide.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class MachineType(str, Enum):
    cnc = "CNC"
    pump = "Pump"
    compressor = "Compressor"
    robotic_arm = "Robotic Arm"


class OperatingMode(str, Enum):
    idle = "idle"
    normal = "normal"
    peak = "peak"


class RiskLevel(str, Enum):
    low = "Faible"
    moderate = "Modéré"
    high = "Élevé"
    critical = "Critique"


class SensorInput(BaseModel):
    """Données capteurs envoyées par le client pour obtenir une prédiction.

    Toutes les plages correspondent aux valeurs observées dans le dataset
    predictive_maintenance_v3.csv.
    """

    machine_type: MachineType = Field(
        description="Type de machine (CNC, Pump, Compressor, Robotic Arm)"
    )
    operating_mode: OperatingMode = Field(
        description="Mode opératoire de la machine"
    )
    vibration_rms: float = Field(
        ge=0.35, le=10.0,
        description="Vibration RMS mesurée par l'accéléromètre (m/s²)",
    )
    temperature_motor: float = Field(
        ge=28.0, le=95.0,
        description="Température du moteur (°C)",
    )
    current_phase_avg: float = Field(
        ge=0.0, le=20.0,
        description="Courant de phase moyen (A)",
    )
    pressure_level: float = Field(
        ge=0.0, le=50.0,
        description="Niveau de pression hydraulique (bar)",
    )
    rpm: float = Field(
        ge=0.0, le=3000.0,
        description="Vitesse de rotation (tours/minute)",
    )
    hours_since_maintenance: float = Field(
        ge=0.0, le=600.0,
        description="Heures écoulées depuis la dernière maintenance",
    )
    ambient_temp: float = Field(
        ge=8.0, le=18.0,
        description="Température ambiante de l'atelier (°C)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "machine_type": "CNC",
                "operating_mode": "normal",
                "vibration_rms": 3.5,
                "temperature_motor": 72.0,
                "current_phase_avg": 12.0,
                "pressure_level": 30.0,
                "rpm": 1500.0,
                "hours_since_maintenance": 250.0,
                "ambient_temp": 15.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Réponse de l'endpoint /predict."""

    failure_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probabilité de panne dans les 24 prochaines heures",
    )
    predicted_class: Literal[0, 1] = Field(
        description="0 = pas de panne, 1 = panne probable",
    )
    risk_level: RiskLevel = Field(
        description="Niveau de risque : Faible / Modéré / Élevé / Critique",
    )
    threshold_used: float = Field(
        description="Seuil de décision utilisé pour la classification",
    )
    model_name: str = Field(
        description="Nom du modèle ayant effectué la prédiction",
    )
    recommendation: str = Field(
        description="Recommandation d'action pour le responsable maintenance",
    )


class HealthResponse(BaseModel):
    """Réponse de l'endpoint /health."""

    status: Literal["ok", "degraded"] = "ok"
    model: str
    version: str = "1.0"
    model_loaded: bool
    threshold: float


class ModelInfoResponse(BaseModel):
    """Réponse de l'endpoint /model-info."""

    model_name: str
    features: list[str]
    feature_ranges: dict[str, dict]
    target: str = "failure_within_24h"
    task: str = "binary_classification"
    imbalance_ratio: float = 5.8
    class_labels: dict[str, str] = {"0": "Pas de panne", "1": "Panne dans 24h"}
