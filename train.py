"""Script d'entraînement principal — Maintenance Prédictive Industrielle.

Usage :
    python train.py                    # Entraîne tous les modèles
    python train.py --skip-imbalance   # Ignore la comparaison des méthodes de rééquilibrage
    python train.py --model rf         # Entraîne uniquement Random Forest
    python train.py --threshold-metric recall  # Optimise le seuil sur le Recall

Sortie :
    models/best_model.pkl            → Modèle final sérialisé
    models/comparison_report.csv     → Tableau comparatif des 4 modèles
    models/threshold_analysis.csv    → Analyse du seuil de décision
"""
from __future__ import annotations

import argparse
import time
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# Racine du projet (répertoire contenant ce fichier)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_data, get_train_test_split, get_feature_names
from src.data.imbalance import compare_imbalance_methods
from src.models import baseline, random_forest, gradient_boosting, mlp
from src.evaluation.metrics import (
    compare_models,
    find_optimal_threshold,
    compute_shap,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne les modèles de maintenance prédictive")
    parser.add_argument("--skip-imbalance", action="store_true",
                        help="Ignorer la comparaison des méthodes de rééquilibrage")
    parser.add_argument("--model", choices=["all", "lr", "rf", "xgb", "mlp"], default="all",
                        help="Modèle(s) à entraîner (default: all)")
    parser.add_argument("--threshold-metric", choices=["f1", "recall"], default="f1",
                        help="Métrique à optimiser pour le seuil de décision")
    parser.add_argument("--xgb-iter", type=int, default=30,
                        help="Nombre d'itérations RandomizedSearchCV pour XGBoost")
    return parser.parse_args()


def step_preprocessing() -> tuple:
    """Étape 1 : Chargement et préparation des données."""
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — Chargement et préparation des données")
    print("=" * 60)

    df = load_data()
    print(f"  Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    class_counts = df["failure_within_24h"].value_counts()
    ratio = class_counts[0] / class_counts[1]
    print(f"  Classe 0 (pas de panne) : {class_counts[0]:,}")
    print(f"  Classe 1 (panne)        : {class_counts[1]:,}")
    print(f"  Ratio déséquilibre      : {ratio:.1f}:1")

    X_train, X_test, y_train, y_test, pipeline = get_train_test_split(df)
    feature_names = get_feature_names(pipeline)

    print(f"  Train set : {X_train.shape[0]:,} exemples")
    print(f"  Test set  : {X_test.shape[0]:,} exemples")
    print(f"  Features  : {len(feature_names)} (après encodage OHE)")

    return X_train, X_test, y_train, y_test, pipeline, feature_names


def step_imbalance(X_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    """Étape 2 : Comparaison des méthodes de rééquilibrage."""
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Comparaison des méthodes de rééquilibrage")
    print("=" * 60)
    print("  (Validation croisée Stratified K-Fold, k=5)")

    df_imbalance = compare_imbalance_methods(X_train, np.array(y_train))

    print("\n  Résultats comparatifs :")
    print(df_imbalance[["Méthode", "Recall", "F1-Score", "PR-AUC"]].to_string(index=False))
    return df_imbalance


def step_train_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_filter: str = "all",
    xgb_iter: int = 30,
    threshold_metric: str = "f1",
) -> tuple[dict, dict, dict]:
    """Étapes 3-4 : Entraînement et évaluation des 4 modèles."""
    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Entraînement des modèles")
    print("=" * 60)

    trained_models = {}
    results = {}
    thresholds = {}

    model_configs = {
        "lr":  ("Logistic Regression", baseline),
        "rf":  ("Random Forest",        random_forest),
        "xgb": ("XGBoost",              gradient_boosting),
        "mlp": ("MLP (PyTorch)",        mlp),
    }

    for key, (name, module) in model_configs.items():
        if model_filter not in ("all", key):
            continue

        print(f"\n  --- {name} ---")
        t0 = time.time()

        if key == "xgb":
            model = module.train(X_train, y_train, n_iter=xgb_iter)
        elif key == "mlp":
            model, history = module.train(X_train, y_train)
            # Sauvegarde de l'historique d'entraînement
            pd.DataFrame(history).to_csv(MODELS_DIR / "mlp_training_history.csv", index=False)
        else:
            model = module.train(X_train, y_train)

        train_time = time.time() - t0

        # Ajustement du seuil de décision
        from src.evaluation.metrics import _get_probabilities
        y_proba = _get_probabilities(model, X_test)
        optimal_threshold, df_thresh = find_optimal_threshold(
            np.array(y_test), y_proba, metric=threshold_metric
        )
        thresholds[name] = optimal_threshold
        print(f"  Seuil optimal ({threshold_metric}) : {optimal_threshold}")

        # Évaluation avec le seuil optimal
        metrics = module.evaluate(model, X_test, y_test, threshold=optimal_threshold)
        metrics["train_time"] = train_time
        results[name] = metrics

        print(f"  Recall={metrics['recall']:.3f} | F1={metrics['f1']:.3f} | "
              f"ROC-AUC={metrics['roc_auc']:.3f} | PR-AUC={metrics['pr_auc']:.3f}")

        # Sauvegarde individuelle
        module.save_model(model)
        trained_models[name] = model

    return trained_models, results, thresholds


def step_select_best_model(
    trained_models: dict,
    results: dict,
) -> tuple[str, object]:
    """Étape 5 : Sélection du meilleur modèle selon Recall et F1.

    Critère de sélection :
    Le modèle final n'est pas forcément celui avec le score le plus élevé.
    On favorise le compromis Recall / stabilité / interprétabilité.
    En cas d'égalité sur le Recall, le F1 départage.

    Returns:
        Tuple (nom_du_modèle, modèle_sélectionné).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Sélection du meilleur modèle")
    print("=" * 60)

    # Score composite : 70% Recall + 30% F1
    best_name = max(
        results.keys(),
        key=lambda m: 0.7 * results[m]["recall"] + 0.3 * results[m]["f1"],
    )
    best_metrics = results[best_name]

    print(f"\n  Modèle sélectionné : {best_name}")
    print(f"  Recall  : {best_metrics['recall']:.3f}")
    print(f"  F1      : {best_metrics['f1']:.3f}")
    print(f"  PR-AUC  : {best_metrics['pr_auc']:.3f}")
    print(f"  Justification : meilleur compromis Recall/F1 (priorité industrielle)")

    return best_name, trained_models[best_name]


def step_shap(
    best_model,
    best_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """Étape 6 : Analyse SHAP du modèle final."""
    print("\n" + "=" * 60)
    print("ÉTAPE 6 — Interprétabilité SHAP")
    print("=" * 60)

    print(f"  Calcul des SHAP values pour : {best_name}")
    shap_results = compute_shap(
        best_model, X_train, X_test, feature_names, model_name=best_name
    )

    top_features = shap_results["feature_importance"].head(5)
    print("\n  Top 5 features (impact moyen) :")
    for _, row in top_features.iterrows():
        print(f"    {row['feature']:35s} : {row['mean_abs_shap']:.4f}")

    # Sauvegarde de l'importance SHAP
    shap_results["feature_importance"].to_csv(
        MODELS_DIR / "shap_feature_importance.csv", index=False
    )

    # Sauvegarde des figures SHAP
    shap_results["fig_summary"].savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    shap_results["fig_bar"].savefig(MODELS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    print("  Figures SHAP sauvegardées dans models/")


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  MAINTENANCE PRÉDICTIVE — PIPELINE D'ENTRAÎNEMENT")
    print("  EFREI M1 DE — Bloc 2 RNCP40875")
    print("=" * 60)

    # --- Étape 1 : Preprocessing ---
    X_train, X_test, y_train, y_test, pipeline, feature_names = step_preprocessing()

    # Sauvegarder le pipeline de preprocessing pour le dashboard/API
    joblib.dump(pipeline, MODELS_DIR / "preprocessing_pipeline.pkl")
    print("  Pipeline preprocessing sauvegardé.")

    # --- Étape 2 : Comparaison méthodes de rééquilibrage ---
    if not args.skip_imbalance:
        df_imbalance = step_imbalance(X_train, y_train)
        df_imbalance.to_csv(MODELS_DIR / "imbalance_comparison.csv", index=False)
        print("  Résultats sauvegardés : models/imbalance_comparison.csv")

    # --- Étape 3-4 : Entraînement et évaluation ---
    trained_models, results, thresholds = step_train_models(
        X_train, X_test, y_train, y_test,
        feature_names=feature_names,
        model_filter=args.model,
        xgb_iter=args.xgb_iter,
        threshold_metric=args.threshold_metric,
    )

    # --- Rapport comparatif ---
    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Tableau comparatif des modèles")
    print("=" * 60)
    df_comparison = compare_models(results)
    print(df_comparison.to_string(index=False))
    df_comparison.to_csv(MODELS_DIR / "comparison_report.csv", index=False)
    print("  Rapport sauvegardé : models/comparison_report.csv")

    # --- Étape 5 : Sélection du meilleur modèle ---
    if len(trained_models) > 0:
        best_name, best_model = step_select_best_model(trained_models, results)

        # Sauvegarder le meilleur modèle (chargé par le dashboard et l'API)
        best_model_path = MODELS_DIR / "best_model.pkl"
        if best_name == "MLP (PyTorch)":
            # torch.save pour le MLP (joblib ne gère pas bien les modèles PyTorch)
            from src.models.mlp import save_model as mlp_save
            mlp_save(best_model, best_model_path)
        else:
            joblib.dump(best_model, best_model_path)
        joblib.dump(best_name, MODELS_DIR / "best_model_name.pkl")
        joblib.dump(thresholds[best_name], MODELS_DIR / "best_threshold.pkl")
        joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
        print(f"\n  Meilleur modèle sauvegardé : {best_model_path}")

        # --- Étape 6 : SHAP ---
        try:
            step_shap(best_model, best_name, X_train, X_test, feature_names)
        except Exception as e:
            print(f"  [WARN] SHAP indisponible pour ce modèle : {e}")

    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT TERMINÉ")
    print("  → Lancer le dashboard : streamlit run dashboard/app.py")
    print("  → Lancer l'API        : uvicorn api.main:app --reload")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
