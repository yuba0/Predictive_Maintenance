"""Modèle Gradient Boosting : XGBoost avec RandomizedSearchCV.

Choix justifié : XGBoost est généralement supérieur au Random Forest sur les
données tabulaires déséquilibrées grâce à son paramètre `scale_pos_weight`
qui pondère directement la classe minoritaire dans la fonction de perte.
L'early stopping évite l'overfitting sans tuning exhaustif du nombre d'arbres.

scale_pos_weight = ratio majoritaire / minoritaire ≈ 5.8 — indique à XGBoost
de pénaliser 5.8× plus fortement les faux négatifs que les faux positifs.
"""

import numpy as np
import joblib
from pathlib import Path
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "gradient_boosting.pkl"

# Ratio classes pour scale_pos_weight (20482 / 3560 ≈ 5.75)
SCALE_POS_WEIGHT = 20482 / 3560

# Espace de recherche RandomizedSearchCV
PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 500),
    "learning_rate": uniform(0.01, 0.3),
    "max_depth": randint(3, 10),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0, 0.5),
}


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 30,
) -> XGBClassifier:
    """Entraîne XGBoost via RandomizedSearchCV avec early stopping.

    scale_pos_weight corrige le déséquilibre de classes directement dans la loss.
    L'early stopping sur la validation interne arrête l'entraînement dès que
    la performance sur le fold de validation ne s'améliore plus.

    Args:
        X_train: Features d'entraînement transformées (array numpy).
        y_train: Labels d'entraînement.
        n_iter: Nombre de combinaisons d'hyperparamètres à tester.

    Returns:
        Meilleur XGBClassifier trouvé, fitté sur X_train complet.
    """
    y_train = np.array(y_train)

    base_model = XGBClassifier(
        scale_pos_weight=SCALE_POS_WEIGHT,  # Gestion déséquilibre dans la loss
        eval_metric="aucpr",                # PR-AUC comme métrique d'évaluation
        early_stopping_rounds=20,           # Arrêt si pas d'amélioration sur 20 rounds
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=skf,
        scoring="recall",
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print("  [XGBoost] RandomizedSearchCV en cours...")
    # eval_set pour early stopping : utiliser 10% du train comme set de validation interne
    split_idx = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # RandomizedSearchCV sans early stopping pour la comparaison des hyperparamètres
    base_model_no_es = XGBClassifier(
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )
    random_search_no_es = RandomizedSearchCV(
        estimator=base_model_no_es,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=skf,
        scoring="recall",
        refit=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    random_search_no_es.fit(X_train, y_train)
    best_params = random_search_no_es.best_params_

    print(f"  Meilleurs hyperparamètres : {best_params}")

    # Fit final avec early stopping sur un val set interne
    best_model = XGBClassifier(
        **best_params,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=0,
    )
    best_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"  Nombre d'arbres optimal (early stopping) : {best_model.best_iteration}")
    return best_model


def evaluate(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Évalue le modèle sur le jeu de test avec métriques complètes.

    Args:
        model: XGBClassifier fitté.
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
        "model": "XGBoost",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "threshold": threshold,
    }


def save_model(model: XGBClassifier, path: Path = MODEL_PATH) -> None:
    """Sérialise le modèle avec joblib.

    Args:
        model: Modèle fitté à sauvegarder.
        path: Chemin de sauvegarde.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Modèle sauvegardé : {path}")
