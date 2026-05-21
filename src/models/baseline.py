"""Modèle baseline : Régression Logistique.

Choix justifié : la régression logistique sert de référence interprétable.
Elle offre une baseline solide grâce à sa simplicité et sa robustesse sur des
données tabulaires normalisées. Toute complexité supplémentaire doit apporter
un gain mesurable par rapport à ce baseline.

class_weight="balanced" : corrige le déséquilibre 5.8:1 en pénalisant davantage
les erreurs sur la classe minoritaire (panne = 1).
"""

import time
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "baseline_lr.pkl"


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Entraîne une Régression Logistique avec validation croisée Stratified K-Fold.

    Args:
        X_train: Features d'entraînement transformées (array numpy).
        y_train: Labels d'entraînement.
        C: Inverse de la force de régularisation (plus C est grand, moins régularisé).
        max_iter: Nombre maximum d'itérations du solveur.

    Returns:
        Modèle LogisticRegression fitté sur l'intégralité de X_train.
    """
    y_train = np.array(y_train)

    # saga : plus stable que lbfgs avec class_weight="balanced" sur données déséquilibrées
    model = LogisticRegression(
        C=C,
        max_iter=5000,
        class_weight="balanced",
        solver="saga",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Validation croisée pour estimer la stabilité avant fit final
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=["recall", "f1", "roc_auc", "average_precision"],
        return_train_score=False,
    )

    print("  [Logistic Regression] CV Stratified K-Fold (k=5):")
    print(f"    Recall  : {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}")
    print(f"    F1      : {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
    print(f"    ROC-AUC : {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")
    print(f"    PR-AUC  : {cv_results['test_average_precision'].mean():.3f}")

    # Fit final sur l'ensemble du train set
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Évalue le modèle sur le jeu de test avec métriques complètes.

    Args:
        model: Modèle LogisticRegression fitté.
        X_test: Features de test transformées.
        y_test: Labels de test réels.
        threshold: Seuil de décision (par défaut 0.5, ajustable pour optimiser le Recall).

    Returns:
        Dictionnaire des métriques : accuracy, precision, recall, f1, roc_auc, pr_auc.
    """
    y_test = np.array(y_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "threshold": threshold,
    }


def save_model(model: LogisticRegression, path: Path = MODEL_PATH) -> None:
    """Sérialise le modèle avec joblib.

    Args:
        model: Modèle fitté à sauvegarder.
        path: Chemin de sauvegarde.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Modèle sauvegardé : {path}")
