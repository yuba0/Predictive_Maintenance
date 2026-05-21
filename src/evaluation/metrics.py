"""Module d'évaluation, comparaison des modèles et interprétabilité SHAP.

Métriques prioritaires dans ce contexte industriel :
- Recall   : minimiser les faux négatifs (pannes non détectées → arrêt machine)
- PR-AUC   : robuste au déséquilibre de classes (plus informatif que ROC-AUC)
- F1-Score : compromis Precision / Recall

L'accuracy est reportée mais ne sert pas de critère de sélection du modèle
(déséquilibre 5.8:1 → un modèle qui prédit toujours 0 aurait 85% d'accuracy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    recall_score,
    average_precision_score,
)

# Palette de couleurs par modèle pour cohérence visuelle dans le dashboard
MODEL_COLORS = {
    "Logistic Regression": "#636EFA",
    "Random Forest":        "#EF553B",
    "XGBoost":              "#00CC96",
    "MLP (PyTorch)":        "#AB63FA",
}


# ---------------------------------------------------------------------------
# Ajustement du seuil de décision
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Teste les seuils de 0.1 à 0.9 et retourne le seuil optimal.

    Dans un contexte de maintenance industrielle, le choix du seuil doit
    arbitrer entre Recall (minimiser pannes manquées) et Precision (éviter
    trop d'alertes inutiles). Le seuil par défaut (0.5) est rarement optimal
    sur des données déséquilibrées.

    Seuil retenu : maximisation du F1 ou du Recall selon le contexte.
    Si le coût d'une panne non détectée est très élevé → favoriser Recall.

    Args:
        y_true: Labels réels.
        y_proba: Probabilités prédites par le modèle.
        metric: Métrique à optimiser ("f1" ou "recall").
        thresholds: Array de seuils à tester (par défaut 0.1 à 0.9, pas 0.05).

    Returns:
        Tuple (seuil_optimal, DataFrame avec F1/Recall/Precision par seuil).
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    records = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        records.append({
            "threshold": round(t, 2),
            "precision": float(np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-9)),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    df_thresholds = pd.DataFrame(records)

    if metric == "recall":
        best_idx = df_thresholds["recall"].idxmax()
    else:
        best_idx = df_thresholds["f1"].idxmax()

    optimal_threshold = df_thresholds.loc[best_idx, "threshold"]
    return float(optimal_threshold), df_thresholds


def plot_threshold_curve(df_thresholds: pd.DataFrame, optimal_threshold: float) -> go.Figure:
    """Trace la courbe F1 / Recall / Precision en fonction du seuil (Plotly).

    Args:
        df_thresholds: DataFrame retourné par find_optimal_threshold().
        optimal_threshold: Seuil optimal à mettre en évidence.

    Returns:
        Figure Plotly interactive.
    """
    fig = go.Figure()

    for metric, color in [("f1", "#EF553B"), ("recall", "#00CC96"), ("precision", "#636EFA")]:
        fig.add_trace(go.Scatter(
            x=df_thresholds["threshold"],
            y=df_thresholds[metric],
            name=metric.capitalize(),
            line=dict(color=color, width=2),
        ))

    fig.add_vline(
        x=optimal_threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Seuil optimal : {optimal_threshold}",
        annotation_position="top right",
    )

    fig.update_layout(
        title="F1 / Recall / Precision selon le seuil de décision",
        xaxis_title="Seuil",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Comparaison des modèles
# ---------------------------------------------------------------------------

def compare_models(results_dict: dict) -> pd.DataFrame:
    """Génère un tableau comparatif des modèles avec toutes les métriques.

    Args:
        results_dict: Dictionnaire {model_name: metrics_dict} où metrics_dict
                      est retourné par les fonctions evaluate() de chaque modèle.
                      Doit inclure : accuracy, precision, recall, f1,
                                     roc_auc, pr_auc, train_time.

    Returns:
        DataFrame pandas trié par Recall décroissant.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Modèle":      model_name,
            "Accuracy":    round(metrics.get("accuracy", 0), 4),
            "Precision":   round(metrics.get("precision", 0), 4),
            "Recall ↑":    round(metrics.get("recall", 0), 4),
            "F1-Score":    round(metrics.get("f1", 0), 4),
            "ROC-AUC":     round(metrics.get("roc_auc", 0), 4),
            "PR-AUC ↑":    round(metrics.get("pr_auc", 0), 4),
            "Seuil":       metrics.get("threshold", 0.5),
            "Temps (s)":   round(metrics.get("train_time", 0), 1),
        })

    df = pd.DataFrame(rows).sort_values("Recall ↑", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Matrices de confusion
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> go.Figure:
    """Trace la matrice de confusion interactive (Plotly).

    Args:
        y_true: Labels réels.
        y_pred: Labels prédits.
        model_name: Nom du modèle (pour le titre).

    Returns:
        Figure Plotly avec heatmap annotée.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Pas de panne (0)", "Panne (1)"]

    # Normalisation pour les pourcentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"{cm[i, j]}<br>({cm_pct[i, j]:.1f}%)",
                showarrow=False,
                font=dict(size=14, color="white" if (i == j and cm[i, j] > cm.max() / 2) else "black"),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title=f"Matrice de confusion — {model_name}",
        xaxis_title="Prédiction",
        yaxis_title="Réalité",
        annotations=annotations,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Courbes ROC et Precision-Recall
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> go.Figure:
    """Trace toutes les courbes ROC sur un même graphe Plotly.

    Args:
        models_dict: Dictionnaire {model_name: model_object}.
                     Le modèle doit exposer predict_proba() ou forward() (MLP).
        X_test: Features de test transformées.
        y_test: Labels de test réels.

    Returns:
        Figure Plotly avec toutes les courbes ROC superposées.
    """
    fig = go.Figure()

    # Ligne de référence aléatoire
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Aléatoire (AUC=0.5)",
    ))

    for model_name, model in models_dict.items():
        y_proba = _get_probabilities(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(model_name, "#000000")

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{model_name} (AUC={roc_auc:.3f})",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Courbes ROC — Comparaison des modèles",
        xaxis_title="Taux de faux positifs (FPR)",
        yaxis_title="Taux de vrais positifs (Recall)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        template="plotly_white",
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_pr_curves(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> go.Figure:
    """Trace toutes les courbes Precision-Recall sur un même graphe Plotly.

    La courbe PR est plus informative que la ROC sur des données déséquilibrées :
    elle focus sur la performance sur la classe minoritaire (pannes).

    Args:
        models_dict: Dictionnaire {model_name: model_object}.
        X_test: Features de test transformées.
        y_test: Labels de test réels.

    Returns:
        Figure Plotly avec toutes les courbes PR superposées.
    """
    y_test = np.array(y_test)
    baseline_precision = y_test.mean()  # Precision d'un classifieur aléatoire

    fig = go.Figure()

    # Baseline (classifieur aléatoire)
    fig.add_hline(
        y=baseline_precision,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Baseline ({baseline_precision:.2f})",
    )

    for model_name, model in models_dict.items():
        y_proba = _get_probabilities(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        color = MODEL_COLORS.get(model_name, "#000000")

        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name=f"{model_name} (AP={pr_auc:.3f})",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Courbes Precision-Recall — Comparaison des modèles",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Interprétabilité SHAP
# ---------------------------------------------------------------------------

def compute_shap(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str = "model",
    max_display: int = 10,
) -> dict:
    """Calcule les SHAP values et génère les visualisations d'interprétabilité.

    Stratégie par type de modèle :
    - Tree-based (RF, XGBoost) : TreeExplainer (exact, rapide)
    - Logistic Regression : LinearExplainer
    - MLP PyTorch : KernelExplainer sur un échantillon (coûteux en calcul)

    Les SHAP values mesurent la contribution marginale de chaque feature à
    la prédiction — contrairement à la feature importance du RF (MDI),
    elles sont cohérentes et fidèles à n'importe quel modèle.

    Args:
        model: Modèle fitté (sklearn ou PyTorch).
        X_train: Features d'entraînement (pour l'explainer background).
        X_test: Features de test (pour les SHAP values).
        feature_names: Noms des features après transformation.
        model_name: Nom du modèle pour les titres.
        max_display: Nombre de features à afficher.

    Returns:
        Dictionnaire avec les SHAP values et figures matplotlib.
    """
    import warnings
    warnings.filterwarnings("ignore")

    model_type = type(model).__name__

    # Sélectionner l'explainer approprié selon le type de modèle
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # Pour les classifieurs binaires sklearn, shap_values est une liste [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_train, feature_names=feature_names)
        shap_values = explainer.shap_values(X_test)

    else:
        # MLP PyTorch → KernelExplainer sur un sous-échantillon du train
        # (KernelExplainer est model-agnostic mais lent)
        background_sample = shap.sample(X_train, 100)

        def predict_fn(X):
            import torch
            tensor = torch.from_numpy(X.astype(np.float32))
            with torch.no_grad():
                logits = model(tensor)
                return torch.sigmoid(logits).numpy()

        explainer = shap.KernelExplainer(predict_fn, background_sample)
        X_test_sample = shap.sample(X_test, 200)
        shap_values = explainer.shap_values(X_test_sample, silent=True)
        X_test = X_test_sample  # Utiliser le sous-échantillon pour les plots

    # --- Summary plot global (importance + direction de l'effet) ---
    fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Summary Plot — {model_name}", fontsize=13)
    plt.tight_layout()

    # --- Bar plot (importance globale — valeur absolue moyenne) ---
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13)
    plt.tight_layout()

    # --- Force plot pour un exemple individuel (premier exemple du test) ---
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0] if len(expected_value) > 0 else 0.0

    force_plot = shap.force_plot(
        expected_value,
        shap_values[0],
        X_test[0],
        feature_names=feature_names,
        matplotlib=False,
        show=False,
    )

    # Importance moyenne par feature (pour le dashboard Streamlit)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).head(max_display)

    return {
        "shap_values": shap_values,
        "explainer": explainer,
        "feature_importance": feature_importance_df,
        "fig_summary": fig_summary,
        "fig_bar": fig_bar,
        "force_plot": force_plot,
    }


def plot_shap_bar_plotly(feature_importance_df: pd.DataFrame, model_name: str) -> go.Figure:
    """Trace un bar chart SHAP interactif avec Plotly pour le dashboard.

    Args:
        feature_importance_df: DataFrame retourné par compute_shap()["feature_importance"].
        model_name: Nom du modèle.

    Returns:
        Figure Plotly bar chart triée par importance décroissante.
    """
    df = feature_importance_df.sort_values("mean_abs_shap", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"],
        y=df["feature"],
        orientation="h",
        marker_color="#EF553B",
    ))
    fig.update_layout(
        title=f"Top features — Impact sur la prédiction ({model_name})",
        xaxis_title="SHAP moyen (impact sur la prédiction)",
        yaxis_title="Variable",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _get_probabilities(model, X_test: np.ndarray) -> np.ndarray:
    """Extrait les probabilités de la classe positive pour n'importe quel modèle.

    Supporte les modèles sklearn (predict_proba) et PyTorch (forward + sigmoid).

    Args:
        model: Modèle entraîné.
        X_test: Features de test.

    Returns:
        Array numpy des probabilités de la classe 1.
    """
    model_type = type(model).__name__

    if model_type == "MaintenanceMLP":
        import torch
        model.eval()
        # Envoyer l'input sur le même device que le modèle (MPS, CUDA ou CPU)
        device = next(model.parameters()).device
        X_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32)).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            return torch.sigmoid(logits).cpu().numpy()
    else:
        return model.predict_proba(X_test)[:, 1]
