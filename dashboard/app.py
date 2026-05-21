"""Dashboard Streamlit — Maintenance Prédictive Industrielle.

Interface destinée au responsable maintenance, non au data scientist.
4 pages accessibles via la sidebar :
  1. Vue d'ensemble  — KPIs, distribution pannes, timeline alertes
  2. Simulation      — Formulaire capteurs → probabilité de panne
  3. Modèles         — Comparaison des 4 modèles, ROC, PR, matrices de confusion
  4. Interprétabilité — SHAP global et top features

Contraintes :
- Le modèle est chargé depuis models/best_model.pkl (jamais ré-entraîné)
- Toutes les visualisations sont en Plotly (interactif)
- Responsive et lisible sans connaissance en ML
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Chemin racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "predictive_maintenance_v3.csv"

# ---------------------------------------------------------------------------
# Configuration Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Maintenance Prédictive | EFREI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Chargement des artefacts (modèle + pipeline + métadonnées)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modèle...")
def load_artifacts():
    """Charge le modèle final et ses métadonnées depuis le dossier models/.

    Le cache Streamlit évite de recharger les artefacts à chaque interaction.
    """
    artifacts = {}

    best_model_path = MODELS_DIR / "best_model.pkl"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.pkl"

    if best_model_path.exists():
        model_name_path = MODELS_DIR / "best_model_name.pkl"
        model_name = joblib.load(model_name_path) if model_name_path.exists() else "Modèle"
        # Le MLP PyTorch est sauvegardé avec torch.save, pas joblib
        if model_name == "MLP (PyTorch)":
            import torch
            from src.models.mlp import load_model as mlp_load
            artifacts["model"] = mlp_load(best_model_path)
        else:
            artifacts["model"] = joblib.load(best_model_path)
        artifacts["model_name"] = model_name
        artifacts["threshold"] = joblib.load(MODELS_DIR / "best_threshold.pkl") \
            if (MODELS_DIR / "best_threshold.pkl").exists() else 0.5
        artifacts["feature_names"] = joblib.load(MODELS_DIR / "feature_names.pkl") \
            if (MODELS_DIR / "feature_names.pkl").exists() else []
    else:
        artifacts["model"] = None
        artifacts["model_name"] = "Non entraîné"
        artifacts["threshold"] = 0.5
        artifacts["feature_names"] = []

    if pipeline_path.exists():
        artifacts["pipeline"] = joblib.load(pipeline_path)
    else:
        artifacts["pipeline"] = None

    if (MODELS_DIR / "comparison_report.csv").exists():
        artifacts["comparison_df"] = pd.read_csv(MODELS_DIR / "comparison_report.csv")
    else:
        artifacts["comparison_df"] = pd.DataFrame()

    if (MODELS_DIR / "shap_feature_importance.csv").exists():
        artifacts["shap_df"] = pd.read_csv(MODELS_DIR / "shap_feature_importance.csv")
    else:
        artifacts["shap_df"] = pd.DataFrame()

    return artifacts


@st.cache_data(show_spinner="Chargement des données...")
def load_dataset() -> pd.DataFrame:
    """Charge et retourne le dataset brut."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Utilitaires de prédiction
# ---------------------------------------------------------------------------

def predict_failure(artifacts: dict, input_data: dict) -> tuple[float, int, str]:
    """Effectue une prédiction à partir des valeurs saisies par l'utilisateur.

    Args:
        artifacts: Dictionnaire des artefacts chargés.
        input_data: Dictionnaire des valeurs capteurs saisies dans le formulaire.

    Returns:
        Tuple (probabilité, classe_prédite, niveau_risque).
    """
    model = artifacts.get("model")
    pipeline = artifacts.get("pipeline")
    threshold = artifacts.get("threshold", 0.5)

    if model is None or pipeline is None:
        return 0.0, 0, "Modèle non disponible"

    df_input = pd.DataFrame([input_data])
    X_transformed = pipeline.transform(df_input)

    model_type = type(model).__name__

    if model_type == "MaintenanceMLP":
        import torch
        model.eval()
        X_tensor = __import__("torch").from_numpy(X_transformed.astype("float32"))
        with __import__("torch").no_grad():
            logit = model(X_tensor)
            proba = __import__("torch").sigmoid(logit).item()
    else:
        proba = model.predict_proba(X_transformed)[0, 1]

    predicted_class = int(proba >= threshold)

    if proba < 0.3:
        risk_level = "Faible"
    elif proba < 0.6:
        risk_level = "Modéré"
    elif proba < 0.8:
        risk_level = "Élevé"
    else:
        risk_level = "Critique"

    return proba, predicted_class, risk_level


def get_risk_color(risk_level: str) -> str:
    colors = {"Faible": "#00CC96", "Modéré": "#FFA500", "Élevé": "#EF553B", "Critique": "#7B0000"}
    return colors.get(risk_level, "#636EFA")


# ---------------------------------------------------------------------------
# Page 1 — Vue d'ensemble
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame, artifacts: dict):
    st.title("🏭 Vue d'ensemble — Maintenance Prédictive")
    st.markdown("---")

    if df.empty:
        st.warning("Dataset non disponible. Vérifiez que le fichier CSV est présent dans `data/`.")
        return

    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    n_failures = df["failure_within_24h"].sum()
    failure_rate = n_failures / total * 100
    n_machines = df["machine_id"].nunique()
    avg_rul = df["rul_hours"].mean()

    with col1:
        st.metric("Taux de panne global", f"{failure_rate:.1f}%",
                  delta=f"{n_failures} pannes / {total:,} observations")
    with col2:
        st.metric("Machines surveillées", str(n_machines),
                  delta="CNC · Pump · Compressor · Robotic Arm")
    with col3:
        st.metric("RUL moyen (heures)", f"{avg_rul:.0f} h",
                  delta="Durée de vie restante estimée")
    with col4:
        machines_at_risk = int(df[df["failure_within_24h"] == 1]["machine_id"].nunique())
        st.metric("Machines à risque", str(machines_at_risk),
                  delta="ayant eu une panne dans les 24h")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Distribution des pannes par type de machine
    with col_left:
        st.subheader("Pannes par type de machine")
        failure_by_type = df.groupby("machine_type")["failure_within_24h"].agg(
            total="count", failures="sum"
        ).reset_index()
        failure_by_type["taux"] = failure_by_type["failures"] / failure_by_type["total"] * 100

        fig = px.bar(
            failure_by_type, x="machine_type", y="taux",
            color="machine_type",
            labels={"machine_type": "Type de machine", "taux": "Taux de panne (%)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Distribution des types de pannes
    with col_right:
        st.subheader("Types de défaillances")
        failure_types = df[df["failure_within_24h"] == 1]["failure_type"].value_counts().reset_index()
        failure_types.columns = ["Type", "Nombre"]

        fig = px.pie(
            failure_types, names="Type", values="Nombre",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Timeline des alertes (agrégé par jour)
    st.subheader("Timeline des alertes de panne")
    if "timestamp" in df.columns:
        df_timeline = df.copy()
        df_timeline["date"] = df_timeline["timestamp"].dt.date
        timeline = df_timeline.groupby("date")["failure_within_24h"].sum().reset_index()
        timeline.columns = ["Date", "Nombre de pannes"]

        fig = px.area(
            timeline, x="Date", y="Nombre de pannes",
            labels={"Nombre de pannes": "Alertes de panne"},
            color_discrete_sequence=["#EF553B"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Distribution des capteurs
    st.subheader("Distribution des capteurs par état (panne / pas de panne)")
    sensor_col = st.selectbox(
        "Sélectionner un capteur",
        ["vibration_rms", "temperature_motor", "current_phase_avg",
         "pressure_level", "rpm", "hours_since_maintenance"],
    )
    fig = px.histogram(
        df.dropna(subset=[sensor_col]),
        x=sensor_col,
        color="failure_within_24h",
        barmode="overlay",
        color_discrete_map={0: "#636EFA", 1: "#EF553B"},
        labels={"failure_within_24h": "Panne dans 24h", sensor_col: sensor_col},
        opacity=0.7,
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Simulation / Prédiction
# ---------------------------------------------------------------------------

def page_simulation(artifacts: dict):
    st.title("🔮 Simulation — Prédiction de panne")
    st.markdown("Renseignez les valeurs des capteurs pour obtenir une prédiction en temps réel.")
    st.markdown("---")

    model_available = artifacts.get("model") is not None and artifacts.get("pipeline") is not None

    if not model_available:
        st.error(
            "Modèle non disponible. Lancez d'abord l'entraînement :\n```\npython train.py\n```"
        )
        return

    st.info(
        f"**Modèle actif :** {artifacts['model_name']} | "
        f"**Seuil de décision :** {artifacts['threshold']:.2f}"
    )

    # Formulaire de saisie des capteurs
    with st.form("prediction_form"):
        st.subheader("Paramètres machine")

        col1, col2 = st.columns(2)
        with col1:
            machine_type = st.selectbox("Type de machine", ["CNC", "Pump", "Compressor", "Robotic Arm"])
            operating_mode = st.selectbox("Mode opératoire", ["idle", "normal", "peak"])
            vibration_rms = st.slider("Vibration RMS", 0.35, 10.0, 1.5, step=0.05)
            temperature_motor = st.slider("Température moteur (°C)", 28.0, 95.0, 55.0, step=0.5)
            current_phase_avg = st.slider("Courant phase moyen (A)", 0.0, 20.0, 7.0, step=0.1)

        with col2:
            pressure_level = st.slider("Niveau de pression (bar)", 0.0, 50.0, 25.0, step=0.5)
            rpm = st.slider("Vitesse de rotation (RPM)", 0, 3000, 1200, step=10)
            hours_since_maintenance = st.slider("Heures depuis maintenance", 0, 600, 150, step=5)
            ambient_temp = st.slider("Température ambiante (°C)", 8.0, 18.0, 13.0, step=0.1)

        submitted = st.form_submit_button("Analyser", use_container_width=True, type="primary")

    if submitted:
        input_data = {
            "machine_type": machine_type,
            "operating_mode": operating_mode,
            "vibration_rms": vibration_rms,
            "temperature_motor": temperature_motor,
            "current_phase_avg": current_phase_avg,
            "pressure_level": pressure_level,
            "rpm": float(rpm),
            "hours_since_maintenance": hours_since_maintenance,
            "ambient_temp": ambient_temp,
        }

        proba, predicted_class, risk_level = predict_failure(artifacts, input_data)
        color = get_risk_color(risk_level)

        st.markdown("---")
        st.subheader("Résultat de l'analyse")

        col_gauge, col_result = st.columns([1, 1])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Probabilité de panne dans 24h"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30],  "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 80], "color": "#f8d7da"},
                        {"range": [80, 100], "color": "#7B0000"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": artifacts["threshold"] * 100,
                    },
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_result:
            st.markdown(f"### Niveau de risque : :{color}[**{risk_level}**]")

            if predicted_class == 1:
                st.error(f"⚠️ **ALERTE : Panne probable dans les 24 heures**")
                recommendations = {
                    "Faible":    "Surveillance accrue recommandée. Prévoir une inspection.",
                    "Modéré":    "Planifier une intervention préventive dans les 48h.",
                    "Élevé":     "Intervention urgente requise. Alerter le technicien.",
                    "Critique":  "ARRÊT MACHINE RECOMMANDÉ. Intervention immédiate.",
                }
                st.warning(f"**Recommandation :** {recommendations.get(risk_level, '')}")
            else:
                st.success("✅ Aucune panne détectée dans les 24 prochaines heures.")
                st.info("Continuer la surveillance selon le plan de maintenance habituel.")

            st.markdown(f"""
            | Paramètre | Valeur |
            |-----------|--------|
            | Probabilité | `{proba:.1%}` |
            | Seuil décision | `{artifacts['threshold']:.2f}` |
            | Classe prédite | `{'Panne (1)' if predicted_class else 'Normal (0)'}` |
            | Machine | `{machine_type}` |
            | Mode | `{operating_mode}` |
            """)


# ---------------------------------------------------------------------------
# Page 3 — Comparaison des modèles
# ---------------------------------------------------------------------------

def page_models(df: pd.DataFrame, artifacts: dict):
    st.title("📊 Comparaison des modèles")
    st.markdown("---")

    comparison_df = artifacts.get("comparison_df", pd.DataFrame())

    if comparison_df.empty:
        st.warning(
            "Rapport de comparaison non disponible. "
            "Lancez l'entraînement : `python train.py`"
        )
        return

    # Tableau comparatif
    st.subheader("Métriques comparatives — 4 modèles")
    st.markdown(
        "_Les métriques prioritaires sont **Recall** et **PR-AUC** — contexte industriel "
        "(faux négatif = panne non détectée = arrêt machine non planifié)._"
    )

    def highlight_best(s):
        """Met en vert la meilleure valeur de chaque colonne numérique."""
        if s.dtype == object:
            return [""] * len(s)
        max_val = s.max()
        return ["background-color: #d4edda; font-weight: bold"
                if v == max_val else "" for v in s]

    numeric_cols = comparison_df.select_dtypes(include="number").columns.tolist()
    styled_df = comparison_df.style.apply(highlight_best, subset=numeric_cols)
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")

    # Graphiques ROC et PR côte à côte
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Courbes ROC")
        if not df.empty:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Aléatoire",
            ))
            model_colors = {
                "Logistic Regression": "#636EFA",
                "Random Forest":       "#EF553B",
                "XGBoost":             "#00CC96",
                "MLP (PyTorch)":       "#AB63FA",
            }
            for _, row in comparison_df.iterrows():
                model_name = row.get("Modèle", "")
                roc_auc = row.get("ROC-AUC", 0)
                color = model_colors.get(model_name, "#000")
                fig_roc.add_trace(go.Scatter(
                    x=[0, 0.5, 1], y=[0, roc_auc, 1],
                    name=f"{model_name} (AUC={roc_auc:.3f})",
                    line=dict(color=color, width=2),
                ))
            fig_roc.update_layout(
                xaxis_title="FPR", yaxis_title="TPR (Recall)",
                template="plotly_white",
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption("Note : courbes indicatives basées sur l'AUC reporté.")

    with col_right:
        st.subheader("PR-AUC par modèle")
        fig_pr = px.bar(
            comparison_df, x="Modèle", y="PR-AUC ↑",
            color="Modèle",
            color_discrete_map={
                "Logistic Regression": "#636EFA",
                "Random Forest":       "#EF553B",
                "XGBoost":             "#00CC96",
                "MLP (PyTorch)":       "#AB63FA",
            },
            labels={"PR-AUC ↑": "PR-AUC (Precision-Recall AUC)"},
        )
        fig_pr.update_layout(showlegend=False, template="plotly_white", yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_pr, use_container_width=True)

    # Radar chart des métriques
    st.markdown("---")
    st.subheader("Vue radar — Performance globale")

    radar_metrics = ["Recall ↑", "F1-Score", "ROC-AUC", "PR-AUC ↑", "Precision"]
    available_metrics = [m for m in radar_metrics if m in comparison_df.columns]

    fig_radar = go.Figure()
    for _, row in comparison_df.iterrows():
        model_name = row.get("Modèle", "")
        values = [row.get(m, 0) for m in available_metrics] + [row.get(available_metrics[0], 0)]
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            name=model_name,
            fill="toself",
            opacity=0.5,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1])),
        template="plotly_white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Justification du modèle final
    st.markdown("---")
    st.subheader("Justification du modèle final")
    best_model_name = artifacts.get("model_name", "Non défini")
    best_threshold = artifacts.get("threshold", 0.5)

    best_row = comparison_df[comparison_df["Modèle"] == best_model_name]
    if not best_row.empty:
        row = best_row.iloc[0]
        st.success(f"""
**Modèle retenu : {best_model_name}**

| Métrique | Valeur |
|----------|--------|
| Recall | {row.get('Recall ↑', 'N/A')} |
| F1-Score | {row.get('F1-Score', 'N/A')} |
| PR-AUC | {row.get('PR-AUC ↑', 'N/A')} |
| Seuil de décision | {best_threshold:.2f} |

**Justification :** Dans un contexte de maintenance industrielle, le Recall est la métrique
prioritaire. Un faux négatif (panne non détectée) entraîne un arrêt machine non planifié,
dont le coût dépasse largement celui d'un faux positif (alerte inutile).
Le modèle retenu offre le meilleur compromis Recall / stabilité / interprétabilité.
        """)


# ---------------------------------------------------------------------------
# Page 4 — Interprétabilité SHAP
# ---------------------------------------------------------------------------

def page_interpretability(artifacts: dict):
    st.title("🔍 Interprétabilité — Pourquoi ce modèle prédit une panne ?")
    st.markdown("---")

    shap_df = artifacts.get("shap_df", pd.DataFrame())
    model_name = artifacts.get("model_name", "Modèle")

    if shap_df.empty:
        st.warning(
            "Analyse SHAP non disponible. Lancez l'entraînement complet : `python train.py`"
        )
        st.markdown("""
**Principe des valeurs SHAP :**
Les valeurs SHAP (SHapley Additive exPlanations) mesurent la contribution de chaque variable
à la prédiction du modèle. Plus la valeur SHAP est élevée en valeur absolue, plus la variable
influence la décision.

Exemple d'interprétation :
- Une vibration élevée → contribution positive à la probabilité de panne
- Une température normale → contribution négative (signe de bon fonctionnement)
        """)
        return

    # Bar chart SHAP (importance globale)
    st.subheader(f"Top {len(shap_df)} variables les plus influentes — {model_name}")
    df_sorted = shap_df.sort_values("mean_abs_shap", ascending=True)

    fig_bar = go.Figure(go.Bar(
        x=df_sorted["mean_abs_shap"],
        y=df_sorted["feature"],
        orientation="h",
        marker=dict(
            color=df_sorted["mean_abs_shap"],
            colorscale="Reds",
        ),
        text=[f"{v:.4f}" for v in df_sorted["mean_abs_shap"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title=f"Importance SHAP moyenne — {model_name}",
        xaxis_title="Valeur SHAP moyenne (|impact sur la prédiction|)",
        yaxis_title="Variable",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Explication en langage naturel
    st.markdown("---")
    st.subheader("Explication en langage naturel")

    if len(shap_df) >= 2:
        top1 = shap_df.iloc[0]["feature"]
        top2 = shap_df.iloc[1]["feature"]
        top3 = shap_df.iloc[2]["feature"] if len(shap_df) >= 3 else "N/A"

        feature_labels = {
            "vibration_rms": "la vibration RMS",
            "temperature_motor": "la température moteur",
            "current_phase_avg": "le courant de phase moyen",
            "pressure_level": "le niveau de pression",
            "rpm": "la vitesse de rotation (RPM)",
            "hours_since_maintenance": "les heures depuis la dernière maintenance",
            "ambient_temp": "la température ambiante",
        }

        label1 = feature_labels.get(top1, top1)
        label2 = feature_labels.get(top2, top2)
        label3 = feature_labels.get(top3, top3)

        st.info(f"""
**Le modèle base principalement ses décisions sur :**

1. **{label1}** — variable la plus déterminante dans la prédiction de panne
2. **{label2}** — deuxième facteur d'influence
3. **{label3}** — troisième facteur d'influence

Une valeur anormalement élevée de ces indicateurs augmente significativement
la probabilité de panne dans les 24 heures suivantes.

**Comment agir ?**
- Surveiller en priorité {label1} et {label2} sur le tableau de bord de supervision
- Déclencher une inspection préventive dès qu'un seuil d'alerte est franchi
- Planifier la maintenance en tenant compte de {label3}
        """)

    # Affichage des figures SHAP statiques si disponibles
    shap_summary_path = PROJECT_ROOT / "models" / "shap_summary.png"
    shap_bar_path = PROJECT_ROOT / "models" / "shap_bar.png"

    col1, col2 = st.columns(2)
    if shap_summary_path.exists():
        with col1:
            st.subheader("SHAP Summary Plot")
            st.image(str(shap_summary_path), use_container_width=True)
            st.caption(
                "Chaque point représente une observation. La couleur indique la valeur "
                "de la feature (rouge = élevée, bleu = faible). La position sur l'axe X "
                "indique l'impact sur la prédiction."
            )

    if shap_bar_path.exists():
        with col2:
            st.subheader("SHAP Bar Plot")
            st.image(str(shap_bar_path), use_container_width=True)
            st.caption(
                "Importance globale : valeur SHAP absolue moyenne sur l'ensemble du test set. "
                "Plus la barre est longue, plus la variable est influente."
            )


# ---------------------------------------------------------------------------
# Navigation principale
# ---------------------------------------------------------------------------

def main():
    artifacts = load_artifacts()
    df = load_dataset()

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/EFREI_Logo.png/320px-EFREI_Logo.png",
                 width=150)
        st.markdown("---")
        st.title("Navigation")

        page = st.radio(
            "Sélectionner une page",
            ["Vue d'ensemble", "Simulation / Prédiction", "Comparaison modèles", "Interprétabilité"],
            index=0,
        )

        st.markdown("---")
        model_name = artifacts.get("model_name", "Non entraîné")
        threshold = artifacts.get("threshold", 0.5)
        model_loaded = artifacts.get("model") is not None

        st.markdown(f"**Modèle actif :** {model_name}")
        st.markdown(f"**Seuil :** {threshold:.2f}")

        if model_loaded:
            st.success("Modèle chargé ✓")
        else:
            st.error("Modèle non disponible")
            st.code("python train.py", language="bash")

        st.markdown("---")
        st.caption("EFREI M1 DE — Bloc 2 RNCP40875")
        st.caption("Maintenance Prédictive Industrielle")

    if page == "Vue d'ensemble":
        page_overview(df, artifacts)
    elif page == "Simulation / Prédiction":
        page_simulation(artifacts)
    elif page == "Comparaison modèles":
        page_models(df, artifacts)
    elif page == "Interprétabilité":
        page_interpretability(artifacts)


if __name__ == "__main__":
    main()
