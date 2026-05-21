"""Pipeline de préparation des données pour la maintenance prédictive."""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Chemin par défaut du dataset (relatif à la racine du projet)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "predictive_maintenance_v3.csv"

# Colonnes features numériques utilisées pour la classification
NUMERIC_FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
]

# Colonnes features catégorielles
CATEGORICAL_FEATURES = ["machine_type", "operating_mode"]

# Variable cible principale
TARGET = "failure_within_24h"

# Colonnes à exclure (non-features)
DROP_COLS = ["timestamp", "machine_id", "rul_hours", "failure_type", "estimated_repair_cost"]


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut et retourne un DataFrame.

    Args:
        path: Chemin vers le fichier CSV.

    Returns:
        DataFrame brut avec toutes les colonnes.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def build_pipeline() -> Pipeline:
    """Construit le pipeline sklearn complet (imputation + encodage + standardisation).

    Le pipeline utilise un ColumnTransformer pour traiter séparément :
    - Les variables numériques : imputation médiane + StandardScaler
    - Les variables catégorielles : imputation mode + OneHotEncoder

    Aucun fit ne doit être effectué ici — le pipeline est renvoyé non-fitté.

    Returns:
        Pipeline sklearn prêt à être fitté sur X_train.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
    ])

    return pipeline


def get_train_test_split(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """Sépare les données en train/test et retourne les données transformées.

    Stratégie :
    - Stratification sur y pour conserver la distribution des classes (5.8:1)
    - Fit du pipeline uniquement sur X_train (pas de data leakage)
    - Transform appliqué sur train et test séparément

    Args:
        df: DataFrame brut chargé par load_data().
        target: Nom de la colonne cible.
        test_size: Proportion du jeu de test.
        random_state: Graine pour la reproductibilité.

    Returns:
        Tuple (X_train, X_test, y_train, y_test, pipeline_fitted)
        X_train et X_test sont des arrays numpy transformés.
        y_train et y_test sont des Series pandas.
        pipeline_fitted est le pipeline sklearn fitté sur X_train.
    """
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # Split stratifié pour préserver le ratio 5.8:1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = build_pipeline()

    # Fit UNIQUEMENT sur le train set — jamais sur le test
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train, y_test, pipeline


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Retourne les noms des features après transformation du ColumnTransformer.

    Args:
        pipeline: Pipeline fitté.

    Returns:
        Liste des noms de features (numériques + catégorielles encodées).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + cat_feature_names
