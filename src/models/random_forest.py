"""Modèle Random Forest avec GridSearchCV.

Choix justifié : le Random Forest est robuste aux outliers, capture les non-linéarités
et fournit une feature importance native. C'est souvent le meilleur compromis
performance / interprétabilité sur des données tabulaires industrielles.

GridSearchCV explore systématiquement l'espace d'hyperparamètres pour trouver
la combinaison optimisant le Recall sur validation croisée Stratified K-Fold.
"""
from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "random_forest.pkl"

# Grille d'hyperparamètres selon le cahier des charges
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Entraîne un Random Forest via GridSearchCV (Stratified K-Fold, k=5).

    Le scoring principal est le Recall — priorité absolue dans un contexte industriel
    où un faux négatif (panne non détectée) génère un arrêt machine non planifié.

    Args:
        X_train: Features d'entraînement transformées (array numpy).
        y_train: Labels d'entraînement.

    Returns:
        Meilleur RandomForestClassifier trouvé par GridSearchCV, fitté sur X_train complet.
    """
    y_train = np.array(y_train)

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=skf,
        scoring="recall",  # Optimiser le Recall en priorité
        refit=True,        # Refit sur l'ensemble du train avec les meilleurs params
        n_jobs=-1,
        verbose=1,
    )

    print("  [Random Forest] GridSearchCV en cours...")
    grid_search.fit(X_train, y_train)

    print(f"  Meilleurs hyperparamètres : {grid_search.best_params_}")
    print(f"  Meilleur Recall CV        : {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_


def evaluate(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Évalue le modèle sur le jeu de test avec métriques complètes.

    Args:
        model: RandomForestClassifier fitté.
        X_test: Features de test transformées.
        y_test: Labels de test réels.
        threshold: Seuil de décision probabiliste.

    Returns:
        Dictionnaire des métriques : accuracy, precision, recall, f1, roc_auc, pr_auc.
    """
    y_test = np.array(y_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "model": "Random Forest",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "threshold": threshold,
    }


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 15,
) -> dict:
    """Extrait l'importance des features natives du Random Forest.

    L'importance est calculée par diminution moyenne de l'impureté (MDI) —
    rapide mais biaisée vers les features à haute cardinalité.

    Args:
        model: RandomForestClassifier fitté.
        feature_names: Noms des features après transformation.
        top_n: Nombre de features à retourner.

    Returns:
        Dictionnaire {feature_name: importance_score} trié par importance décroissante.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return {feature_names[i]: float(importances[i]) for i in indices}


def save_model(model: RandomForestClassifier, path: Path = MODEL_PATH) -> None:
    """Sérialise le modèle avec joblib.

    Args:
        model: Modèle fitté à sauvegarder.
        path: Chemin de sauvegarde.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Modèle sauvegardé : {path}")
