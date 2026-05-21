"""Comparaison des techniques de gestion du déséquilibre de classes.

Déséquilibre observé : 5.8:1 (20 482 "non-panne" vs 3 560 "panne").
Contexte industriel : un faux négatif (panne non détectée) est bien plus coûteux
qu'un faux positif — la métrique prioritaire est le Recall.

Méthodes comparées :
1. Baseline (aucun rééquilibrage)
2. Random Over-Sampling (ROS)
3. SMOTE
4. Random Under-Sampling (RUS)
5. class_weight="balanced" dans le modèle

Validation : Stratified K-Fold (k=5) sur le train set uniquement.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    make_scorer,
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 42
N_SPLITS = 5

# Scorers pour la validation croisée
SCORING = {
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
    "pr_auc": make_scorer(average_precision_score, needs_proba=True),
}


def _run_cv_with_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampler,
    model: RandomForestClassifier,
    n_splits: int = N_SPLITS,
) -> dict:
    """Exécute une validation croisée Stratified K-Fold avec rééchantillonnage.

    Le rééchantillonnage est appliqué uniquement sur le fold d'entraînement,
    jamais sur le fold de validation — évite toute fuite de données.

    Args:
        X_train: Features d'entraînement (array numpy transformé).
        y_train: Labels d'entraînement.
        sampler: Instance d'un sampler imblearn (ou None pour baseline).
        model: Classifieur Random Forest (non fitté).
        n_splits: Nombre de folds.

    Returns:
        Dictionnaire avec les métriques moyennées sur les k folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    recalls, f1s, roc_aucs, pr_aucs, matrices = [], [], [], [], []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Rééchantillonnage uniquement sur le fold d'entraînement
        if sampler is not None:
            X_fold_train, y_fold_train = sampler.fit_resample(X_fold_train, y_fold_train)

        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        y_proba = model.predict_proba(X_fold_val)[:, 1]

        recalls.append(recall_score(y_fold_val, y_pred))
        f1s.append(f1_score(y_fold_val, y_pred))
        roc_aucs.append(roc_auc_score(y_fold_val, y_proba))
        pr_aucs.append(average_precision_score(y_fold_val, y_proba))
        matrices.append(confusion_matrix(y_fold_val, y_pred))

    avg_cm = np.mean(matrices, axis=0).astype(int)

    return {
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "roc_auc_mean": np.mean(roc_aucs),
        "roc_auc_std": np.std(roc_aucs),
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_std": np.std(pr_aucs),
        "confusion_matrix_avg": avg_cm,
    }


def compare_imbalance_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    """Compare les 5 méthodes de gestion du déséquilibre sur Random Forest.

    Chaque méthode est évaluée par Stratified K-Fold (k=5) sur le train set.
    Le test set n'est jamais utilisé ici — il est réservé à l'évaluation finale.

    Args:
        X_train: Features d'entraînement transformées par le pipeline sklearn.
        y_train: Labels d'entraînement (0 ou 1).

    Returns:
        DataFrame pandas avec une ligne par méthode et les métriques comparatives.
    """
    y_train = np.array(y_train)

    methods = {
        "Baseline (no resampling)": (None, False),
        "Random Over-Sampling": (RandomOverSampler(random_state=RANDOM_STATE), False),
        "SMOTE": (SMOTE(random_state=RANDOM_STATE), False),
        "Random Under-Sampling": (RandomUnderSampler(random_state=RANDOM_STATE), False),
        "class_weight=balanced": (None, True),
    }

    results = []

    for method_name, (sampler, use_class_weight) in methods.items():
        print(f"  → Évaluation : {method_name}...")
        t0 = time.time()

        class_weight = "balanced" if use_class_weight else None
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        metrics = _run_cv_with_resampling(X_train, y_train, sampler, model)
        elapsed = time.time() - t0

        results.append({
            "Méthode": method_name,
            "Recall": f"{metrics['recall_mean']:.3f} ± {metrics['recall_std']:.3f}",
            "F1-Score": f"{metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}",
            "ROC-AUC": f"{metrics['roc_auc_mean']:.3f} ± {metrics['roc_auc_std']:.3f}",
            "PR-AUC": f"{metrics['pr_auc_mean']:.3f} ± {metrics['pr_auc_std']:.3f}",
            "Recall_val": metrics["recall_mean"],
            "F1_val": metrics["f1_mean"],
            "Temps (s)": round(elapsed, 1),
        })

        print(f"     Recall={metrics['recall_mean']:.3f}, F1={metrics['f1_mean']:.3f}, "
              f"PR-AUC={metrics['pr_auc_mean']:.3f}")

    df_results = pd.DataFrame(results)
    return df_results


def get_best_sampler(df_results: pd.DataFrame):
    """Sélectionne la meilleure méthode selon le Recall moyen.

    Dans un contexte de maintenance industrielle, le Recall prime sur la Precision :
    un faux négatif (panne manquée) entraîne un arrêt machine non planifié coûteux.

    Args:
        df_results: DataFrame retourné par compare_imbalance_methods().

    Returns:
        Tuple (method_name, sampler) de la meilleure méthode.
    """
    best_row = df_results.loc[df_results["Recall_val"].idxmax()]
    best_method = best_row["Méthode"]

    samplers = {
        "Baseline (no resampling)": None,
        "Random Over-Sampling": RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "Random Under-Sampling": RandomUnderSampler(random_state=RANDOM_STATE),
        "class_weight=balanced": None,
    }

    print(f"\n  Meilleure méthode selon Recall : {best_method}")
    return best_method, samplers.get(best_method)
