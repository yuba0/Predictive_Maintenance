"""Dashboard Streamlit — Maintenance Prédictive Industrielle.

Interface de supervision industrielle 4 pages :
  1. Vue d'ensemble  — KPIs, distribution pannes, timeline alertes
  2. Simulation      — Formulaire capteurs → probabilité de panne en temps réel
  3. Modèles         — Comparaison 4 modèles, matrices de confusion réelles, ROC/PR réelles
  4. Interprétabilité — SHAP global et top features
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "predictive_maintenance_v3.csv"

MODEL_COLORS = {
    "Logistic Regression": "#636EFA",
    "Random Forest":       "#EF553B",
    "XGBoost":             "#00CC96",
    "MLP (PyTorch)":       "#AB63FA",
}

RISK_CONFIG = {
    "Faible":   {"color": "#06D6A0", "bg": "#DCFCE7", "text": "#166534", "icon": "✅"},
    "Modéré":   {"color": "#F59E0B", "bg": "#FEF3C7", "text": "#92400E", "icon": "⚠️"},
    "Élevé":    {"color": "#EF4444", "bg": "#FEE2E2", "text": "#991B1B", "icon": "🔴"},
    "Critique": {"color": "#7B0000", "bg": "#7F1D1D", "text": "#FECACA", "icon": "🚨"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config + CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PredictMaint — Supervision Industrielle",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.main .block-container { padding-top: 1.5rem; max-width: 1440px; }

/* ── Sidebar dark theme ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #152847 100%) !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #E2E8F0 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3A5F !important; opacity: 0.6; }
[data-testid="stSidebar"] code { background: rgba(255,255,255,0.08) !important; color: #93C5FD !important; }

/* ── KPI cards ── */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
    border-top: 3px solid #3B82F6;
    height: 100%;
}
.kpi-icon  { font-size: 1.5rem; line-height: 1; margin-bottom: 0.4rem; }
.kpi-label { font-size: 0.7rem; font-weight: 600; color: #64748B;
             text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #0F172A; line-height: 1.15; margin: 0.1rem 0; }
.kpi-delta { font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem; }

/* ── Risk panel ── */
.risk-panel {
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    border: 2px solid;
}
.risk-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.08em; opacity: 0.8; }
.risk-value { font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin: 0.3rem 0; }
.risk-sub   { font-size: 0.82rem; opacity: 0.75; }

/* ── Section sub-header ── */
.sub-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1E3A5F;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 1rem;
    margin-top: 0.2rem;
}

/* ── Info chip ── */
.info-chip {
    display: inline-block;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #1E40AF;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires visuels
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(icon: str, label: str, value: str, delta: str = "", accent: str = "#3B82F6") -> str:
    return (
        f'<div class="kpi-card" style="border-top-color:{accent}">'
        f'  <div class="kpi-icon">{icon}</div>'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-delta">{delta}</div>'
        f'</div>'
    )


def section_header(text: str, icon: str = "") -> None:
    prefix = f"{icon}&nbsp;" if icon else ""
    st.markdown(f'<div class="sub-header">{prefix}{text}</div>', unsafe_allow_html=True)


def _plotly_theme() -> dict:
    return dict(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=20, b=20, l=10, r=10),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des artefacts
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement des artefacts...")
def load_artifacts() -> dict:
    arts: dict = {}

    model_path      = MODELS_DIR / "best_model.pkl"
    model_name_path = MODELS_DIR / "best_model_name.pkl"

    arts["model_name"] = (
        joblib.load(model_name_path) if model_name_path.exists() else "Modèle"
    )
    if model_path.exists():
        if arts["model_name"] == "MLP (PyTorch)":
            from src.models.mlp import load_model as _mlp_load
            arts["model"] = _mlp_load(model_path)
        else:
            arts["model"] = joblib.load(model_path)
    else:
        arts["model"] = None

    def _load(fname, default):
        p = MODELS_DIR / fname
        return joblib.load(p) if p.exists() else default

    arts["threshold"]     = _load("best_threshold.pkl", 0.5)
    arts["feature_names"] = _load("feature_names.pkl", [])
    arts["pipeline"]      = _load("preprocessing_pipeline.pkl", None)
    arts["eval_artifacts"] = _load("eval_artifacts.pkl", {})

    arts["comparison_df"] = (
        pd.read_csv(MODELS_DIR / "comparison_report.csv")
        if (MODELS_DIR / "comparison_report.csv").exists()
        else pd.DataFrame()
    )
    arts["shap_df"] = (
        pd.read_csv(MODELS_DIR / "shap_feature_importance.csv")
        if (MODELS_DIR / "shap_feature_importance.csv").exists()
        else pd.DataFrame()
    )
    return arts


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Prédiction
# ─────────────────────────────────────────────────────────────────────────────

def predict_failure(arts: dict, input_data: dict) -> tuple[float, int, str]:
    model    = arts.get("model")
    pipeline = arts.get("pipeline")
    threshold = arts.get("threshold", 0.5)

    if model is None or pipeline is None:
        return 0.0, 0, "Modèle non disponible"

    df_in = pd.DataFrame([input_data])
    X = pipeline.transform(df_in)

    if type(model).__name__ == "MaintenanceMLP":
        import torch
        model.eval()
        device = next(model.parameters()).device
        X_t = torch.from_numpy(X.astype("float32")).to(device)
        with torch.no_grad():
            proba = float(torch.sigmoid(model(X_t)).cpu().item())
    else:
        proba = float(model.predict_proba(X)[0, 1])

    cls = int(proba >= threshold)
    if proba < 0.3:
        risk = "Faible"
    elif proba < 0.6:
        risk = "Modéré"
    elif proba < 0.8:
        risk = "Élevé"
    else:
        risk = "Critique"
    return proba, cls, risk


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Vue d'ensemble
# ─────────────────────────────────────────────────────────────────────────────

_SENSOR_LABELS = {
    "vibration_rms":          "Vibration RMS",
    "temperature_motor":      "Température moteur",
    "current_phase_avg":      "Courant phase moyen",
    "pressure_level":         "Niveau de pression",
    "rpm":                    "Vitesse (RPM)",
    "hours_since_maintenance":"Heures depuis maintenance",
}


def page_overview(df: pd.DataFrame) -> None:
    st.markdown(
        '<h1 style="font-size:1.75rem;font-weight:700;color:#0F172A;'
        'border-bottom:3px solid #3B82F6;padding-bottom:0.5rem;margin-bottom:1.5rem">'
        '🏭 Vue d\'ensemble — Maintenance Prédictive</h1>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("Dataset non disponible. Vérifiez que `data/predictive_maintenance_v3.csv` est présent.")
        return

    total     = len(df)
    n_fail    = int(df["failure_within_24h"].sum())
    fail_rate = n_fail / total * 100
    n_mach    = df["machine_id"].nunique()
    avg_rul   = df["rul_hours"].mean()
    n_risk    = int(df[df["failure_within_24h"] == 1]["machine_id"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("⚠️", "Taux de panne global", f"{fail_rate:.1f}%",
                             f"{n_fail:,} alertes · {total:,} observations", "#EF4444"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("🏭", "Machines surveillées", str(n_mach),
                             "CNC · Pump · Compressor · Robotic Arm", "#3B82F6"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("⏱️", "RUL moyen estimé", f"{avg_rul:.0f} h",
                             "Durée de vie résiduelle (heures)", "#06D6A0"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("🔴", "Machines à risque", str(n_risk),
                             "Ont eu une panne dans les 24h", "#F59E0B"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        section_header("Taux de panne par type de machine", "📊")
        fb = (
            df.groupby("machine_type")["failure_within_24h"]
            .agg(total="count", failures="sum")
            .reset_index()
        )
        fb["taux"] = fb["failures"] / fb["total"] * 100
        fig = px.bar(
            fb, x="machine_type", y="taux",
            color="machine_type",
            color_discrete_sequence=["#3B82F6", "#EF553B", "#00CC96", "#AB63FA"],
            labels={"machine_type": "Type", "taux": "Taux de panne (%)"},
            text_auto=".1f",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", **_plotly_theme())
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_header("Répartition des types de défaillances", "🔩")
        ft = (
            df[df["failure_within_24h"] == 1]["failure_type"]
            .value_counts()
            .reset_index()
        )
        ft.columns = ["Type", "Nombre"]
        fig = px.pie(
            ft, names="Type", values="Nombre", hole=0.45,
            color_discrete_sequence=["#EF4444", "#F59E0B", "#3B82F6", "#8B5CF6"],
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12),
                          **_plotly_theme())
        st.plotly_chart(fig, use_container_width=True)

    section_header("Timeline des alertes de panne (par jour)", "📅")
    if "timestamp" in df.columns:
        df_t = df.copy()
        df_t["date"] = df_t["timestamp"].dt.date
        tl = df_t.groupby("date")["failure_within_24h"].sum().reset_index()
        tl.columns = ["Date", "Alertes"]
        fig = px.area(
            tl, x="Date", y="Alertes",
            color_discrete_sequence=["#EF4444"],
            labels={"Alertes": "Alertes de panne"},
        )
        fig.update_traces(fillcolor="rgba(239,68,68,0.12)", line_color="#EF4444")
        fig.update_layout(**_plotly_theme())
        st.plotly_chart(fig, use_container_width=True)

    section_header("Distribution des capteurs selon l'état machine", "📡")
    sensor = st.selectbox(
        "Capteur à analyser",
        list(_SENSOR_LABELS.keys()),
        format_func=lambda x: _SENSOR_LABELS.get(x, x),
    )
    fig = px.histogram(
        df.dropna(subset=[sensor]),
        x=sensor, color="failure_within_24h", barmode="overlay",
        color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
        labels={"failure_within_24h": "Panne 24h"},
        opacity=0.75,
    )
    fig.update_layout(legend_title_text="Panne dans 24h", **_plotly_theme())
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Simulation / Prédiction
# ─────────────────────────────────────────────────────────────────────────────

def page_simulation(arts: dict) -> None:
    st.markdown(
        '<h1 style="font-size:1.75rem;font-weight:700;color:#0F172A;'
        'border-bottom:3px solid #3B82F6;padding-bottom:0.5rem;margin-bottom:1.5rem">'
        '🔮 Simulation — Prédiction de panne en temps réel</h1>',
        unsafe_allow_html=True,
    )

    model_ok = arts.get("model") is not None and arts.get("pipeline") is not None

    if not model_ok:
        st.error("Modèle non disponible. Lancez d'abord : `python train.py`")
        return

    # Info banner
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div style="background:#EFF6FF;border-radius:8px;padding:0.7rem 1rem;border-left:3px solid #3B82F6">'
            f'<div style="font-size:0.7rem;color:#64748B;font-weight:600;text-transform:uppercase">Modèle actif</div>'
            f'<div style="font-size:0.9rem;font-weight:700;color:#1E40AF">{arts["model_name"]}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="background:#F0FDF4;border-radius:8px;padding:0.7rem 1rem;border-left:3px solid #06D6A0">'
            f'<div style="font-size:0.7rem;color:#64748B;font-weight:600;text-transform:uppercase">Seuil de décision</div>'
            f'<div style="font-size:0.9rem;font-weight:700;color:#166534">{arts["threshold"]:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div style="background:#FFF7ED;border-radius:8px;padding:0.7rem 1rem;border-left:3px solid #F59E0B">'
            '<div style="font-size:0.7rem;color:#64748B;font-weight:600;text-transform:uppercase">Métrique prioritaire</div>'
            '<div style="font-size:0.9rem;font-weight:700;color:#92400E">Recall (sensibilité)</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        section_header("Paramètres de la machine", "⚙️")

        col_cat, col_num1, col_num2 = st.columns([1, 1.5, 1.5])

        with col_cat:
            machine_type = st.selectbox(
                "Type de machine",
                ["CNC", "Pump", "Compressor", "Robotic Arm"],
            )
            operating_mode = st.selectbox(
                "Mode opératoire",
                ["idle", "normal", "peak"],
                format_func=lambda x: {"idle": "⏸ Repos", "normal": "▶ Normal", "peak": "⚡ Pic"}.get(x, x),
            )
            hours_since_maintenance = st.number_input(
                "Heures depuis maintenance", 0, 600, 150, step=10,
                help="Nombre d'heures écoulées depuis la dernière maintenance réalisée",
            )

        with col_num1:
            vibration_rms = st.slider(
                "Vibration RMS", 0.35, 10.0, 1.5, 0.05,
                help="Amplitude RMS des vibrations mesurées sur le bâti",
            )
            temperature_motor = st.slider(
                "Température moteur (°C)", 28.0, 95.0, 55.0, 0.5,
                help="Température de l'enroulement statorique",
            )
            current_phase_avg = st.slider(
                "Courant phase moyen (A)", 0.0, 20.0, 7.0, 0.1,
                help="Courant efficace moyen des 3 phases d'alimentation",
            )

        with col_num2:
            pressure_level = st.slider(
                "Niveau de pression (bar)", 0.0, 50.0, 25.0, 0.5,
                help="Pression du circuit hydraulique ou pneumatique",
            )
            rpm = st.slider(
                "Vitesse de rotation (RPM)", 0, 3000, 1200, 10,
                help="Régime moteur en tours par minute",
            )
            ambient_temp = st.slider(
                "Température ambiante (°C)", 8.0, 18.0, 13.0, 0.1,
                help="Température mesurée dans l'atelier de production",
            )

        submitted = st.form_submit_button(
            "🔍 Analyser la machine", use_container_width=True, type="primary"
        )

    if submitted:
        input_data = {
            "machine_type":          machine_type,
            "operating_mode":        operating_mode,
            "vibration_rms":         vibration_rms,
            "temperature_motor":     temperature_motor,
            "current_phase_avg":     current_phase_avg,
            "pressure_level":        pressure_level,
            "rpm":                   float(rpm),
            "hours_since_maintenance": float(hours_since_maintenance),
            "ambient_temp":          ambient_temp,
        }

        with st.spinner("Analyse en cours..."):
            proba, cls, risk = predict_failure(arts, input_data)

        cfg = RISK_CONFIG.get(risk, RISK_CONFIG["Modéré"])

        st.markdown("---")
        col_gauge, col_res = st.columns([1.2, 1])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(proba * 100, 1),
                number={"suffix": "%", "font": {"size": 42, "color": cfg["color"]}},
                delta={
                    "reference": arts["threshold"] * 100,
                    "suffix": "% (seuil)",
                    "increasing": {"color": "#EF4444"},
                    "decreasing": {"color": "#06D6A0"},
                },
                title={"text": "Probabilité de panne dans 24h",
                       "font": {"size": 13, "color": "#64748B"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 10}},
                    "bar":  {"color": cfg["color"], "thickness": 0.22},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30],  "color": "#DCFCE7"},
                        {"range": [30, 60], "color": "#FEF3C7"},
                        {"range": [60, 80], "color": "#FEE2E2"},
                        {"range": [80, 100],"color": "#FCA5A5"},
                    ],
                    "threshold": {
                        "line":      {"color": "#1E3A5F", "width": 3},
                        "thickness": 0.8,
                        "value":     arts["threshold"] * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="white",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res:
            st.markdown(
                f'<div class="risk-panel" '
                f'style="border-color:{cfg["color"]};background:{cfg["bg"]};color:{cfg["text"]}">'
                f'  <div class="risk-label">Niveau de risque</div>'
                f'  <div class="risk-value">{cfg["icon"]} {risk}</div>'
                f'  <div class="risk-sub">Probabilité : {proba:.1%} — Seuil : {arts["threshold"]:.0%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            RECOS = {
                "Faible":   ("success", "✅ Aucune panne détectée dans les 24h.",
                             "Surveillance normale. Continuer selon le plan de maintenance préventive."),
                "Modéré":   ("warning", "⚠️ Risque modéré détecté.",
                             "Planifier une inspection dans les 48h. Surveiller les capteurs critiques."),
                "Élevé":    ("error", "🔴 Risque élevé — Intervention requise.",
                             "Alerter le technicien. Prévoir une intervention préventive dans les 12h."),
                "Critique": ("error", "🚨 PANNE IMMINENTE — Arrêt recommandé.",
                             "INTERVENTION IMMÉDIATE. Arrêter la machine et procéder à une maintenance d'urgence."),
            }
            stype, title, detail = RECOS[risk]
            st.markdown("<br>", unsafe_allow_html=True)
            if stype == "success":
                st.success(f"**{title}**\n\n{detail}")
            elif stype == "warning":
                st.warning(f"**{title}**\n\n{detail}")
            else:
                st.error(f"**{title}**\n\n{detail}")

            st.markdown(f"""
| Paramètre | Valeur |
|:----------|-------:|
| Machine | `{machine_type}` |
| Mode | `{operating_mode}` |
| Probabilité de panne | `{proba:.1%}` |
| Classe prédite | `{"Panne (1)" if cls else "Normal (0)"}` |
| Seuil de décision | `{arts["threshold"]:.2f}` |
""")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Comparaison des modèles
# ─────────────────────────────────────────────────────────────────────────────

def page_models(arts: dict) -> None:
    st.markdown(
        '<h1 style="font-size:1.75rem;font-weight:700;color:#0F172A;'
        'border-bottom:3px solid #3B82F6;padding-bottom:0.5rem;margin-bottom:1.5rem">'
        '📊 Comparaison des modèles</h1>',
        unsafe_allow_html=True,
    )

    cdf       = arts.get("comparison_df", pd.DataFrame())
    eval_arts = arts.get("eval_artifacts", {})

    if cdf.empty:
        st.warning("Rapport non disponible. Lancez : `python train.py`")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Métriques",
        "🔲 Matrices de confusion",
        "📈 Courbes ROC & PR",
        "🕸️ Radar & Temps",
    ])

    # ── Tab 1 : Metrics ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "> **Priorité métriques (contexte industriel)** : `Recall` > `PR-AUC` > `F1-Score` > `ROC-AUC`  \n"
            "> Un **faux négatif** = panne non détectée = arrêt machine non planifié (~15 k€–50 k€/h)."
        )
        st.markdown("<br>", unsafe_allow_html=True)

        def _highlight_best(s):
            if s.dtype == object:
                return [""] * len(s)
            max_val = s.max()
            return [
                "background-color:#DCFCE7;font-weight:700;color:#166534" if v == max_val else ""
                for v in s
            ]

        num_cols = cdf.select_dtypes(include="number").columns.tolist()
        styled = cdf.style.apply(_highlight_best, subset=num_cols)
        st.dataframe(styled, use_container_width=True, height=195)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Comparaison visuelle des métriques clés", "📊")

        metrics_to_show = ["Recall ↑", "F1-Score", "PR-AUC ↑", "ROC-AUC"]
        available = [m for m in metrics_to_show if m in cdf.columns]
        df_melt = cdf.melt(
            id_vars=["Modèle"], value_vars=available,
            var_name="Métrique", value_name="Score",
        )
        fig = px.bar(
            df_melt, x="Métrique", y="Score", color="Modèle",
            barmode="group", color_discrete_map=MODEL_COLORS, text_auto=".3f",
        )
        fig.update_layout(
            yaxis=dict(range=[0.5, 1.02]),
            legend=dict(orientation="h", y=1.06),
            **_plotly_theme(),
        )
        fig.update_traces(textfont_size=9, textangle=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2 : Confusion matrices ────────────────────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)

        if not eval_arts:
            st.info(
                "Matrices de confusion non disponibles.  \n"
                "Relancez l'entraînement : `python train.py --skip-imbalance`"
            )
        else:
            from sklearn.metrics import confusion_matrix

            y_test     = eval_arts.get("y_test", np.array([]))
            model_names = [k for k in eval_arts if k != "y_test"]
            cols_cm    = st.columns(min(len(model_names), 2))

            for i, mname in enumerate(model_names):
                y_pred = eval_arts[mname]["y_pred"]
                cm     = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                total = len(y_test)

                color = MODEL_COLORS.get(mname, "#3B82F6")

                with cols_cm[i % 2]:
                    fig_cm = go.Figure(go.Heatmap(
                        z=cm,
                        x=["Prédit : Normal (0)", "Prédit : Panne (1)"],
                        y=["Réel : Normal (0)", "Réel : Panne (1)"],
                        colorscale=[[0, "#F8FAFC"], [0.5, "#BFDBFE"], [1, "#1D4ED8"]],
                        text=[
                            [f"TN = {tn:,}<br>({tn/total:.1%})",
                             f"FP = {fp:,}<br>({fp/total:.1%})"],
                            [f"FN = {fn:,}<br>({fn/total:.1%})",
                             f"TP = {tp:,}<br>({tp/total:.1%})"],
                        ],
                        texttemplate="%{text}",
                        showscale=False,
                    ))
                    fig_cm.update_layout(
                        title=dict(text=mname, font=dict(size=13, color=color)),
                        height=290,
                        margin=dict(t=45, b=10, l=10, r=10),
                        font=dict(family="Inter, sans-serif", size=11),
                        **{k: v for k, v in _plotly_theme().items() if k != "margin"},
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    recall_v    = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision_v = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1_v        = (2 * precision_v * recall_v / (precision_v + recall_v)
                                   if (precision_v + recall_v) > 0 else 0)
                    st.markdown(
                        f'<div style="background:#F8FAFC;border-radius:8px;'
                        f'padding:0.55rem 1rem;font-size:0.8rem;margin-bottom:1.2rem;'
                        f'border-left:3px solid {color}">'
                        f'TP={tp:,} &nbsp;·&nbsp; FP={fp:,} &nbsp;·&nbsp; '
                        f'FN={fn:,} &nbsp;·&nbsp; TN={tn:,}<br>'
                        f'<b>Recall={recall_v:.3f}</b> &nbsp;·&nbsp; '
                        f'Precision={precision_v:.3f} &nbsp;·&nbsp; F1={f1_v:.3f}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Tab 3 : ROC & PR curves ───────────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)

        if not eval_arts:
            st.info(
                "Courbes approximatives affichées (basées sur l'AUC du rapport).  \n"
                "Relancez l'entraînement pour les courbes réelles : `python train.py --skip-imbalance`"
            )
            # Fallback approximation
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="#CBD5E1"), name="Aléatoire",
            ))
            for _, row in cdf.iterrows():
                mn  = row.get("Modèle", "")
                auc = row.get("ROC-AUC", 0)
                fig_roc.add_trace(go.Scatter(
                    x=[0, 0.5, 1], y=[0, auc, 1],
                    name=f"{mn} (AUC={auc:.3f})",
                    line=dict(color=MODEL_COLORS.get(mn, "#000"), width=2),
                ))
            fig_roc.update_layout(
                xaxis_title="FPR", yaxis_title="TPR",
                **_plotly_theme(),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        else:
            from sklearn.metrics import roc_curve, precision_recall_curve, auc as sk_auc

            y_test      = eval_arts.get("y_test", np.array([]))
            model_names = [k for k in eval_arts if k != "y_test"]

            col_roc, col_pr = st.columns(2)

            with col_roc:
                section_header("Courbes ROC (Receiver Operating Characteristic)", "📈")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="#CBD5E1", width=1.5),
                    name="Aléatoire",
                ))
                for mname in model_names:
                    y_proba = eval_arts[mname]["y_proba"]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc_val = sk_auc(fpr, tpr)
                    color = MODEL_COLORS.get(mname, "#3B82F6")
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"{mname}  AUC={roc_auc_val:.4f}",
                        line=dict(color=color, width=2.5),
                    ))
                fig_roc.update_layout(
                    xaxis_title="Taux de faux positifs (FPR)",
                    yaxis_title="Taux de vrais positifs (Recall)",
                    legend=dict(x=0.52, y=0.04, bgcolor="rgba(255,255,255,0.85)",
                                bordercolor="#E2E8F0", borderwidth=1),
                    height=400,
                    **_plotly_theme(),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_pr:
                section_header("Courbes Precision-Recall", "📉")
                fig_pr = go.Figure()
                baseline_pr = float(np.sum(y_test) / len(y_test))
                fig_pr.add_trace(go.Scatter(
                    x=[0, 1], y=[baseline_pr, baseline_pr], mode="lines",
                    line=dict(dash="dash", color="#CBD5E1", width=1.5),
                    name=f"Baseline ({baseline_pr:.2f})",
                ))
                for mname in model_names:
                    y_proba = eval_arts[mname]["y_proba"]
                    prec, rec, _ = precision_recall_curve(y_test, y_proba)
                    pr_auc_val   = sk_auc(rec, prec)
                    color = MODEL_COLORS.get(mname, "#3B82F6")
                    fig_pr.add_trace(go.Scatter(
                        x=rec, y=prec, mode="lines",
                        name=f"{mname}  AUC={pr_auc_val:.4f}",
                        line=dict(color=color, width=2.5),
                    ))
                fig_pr.update_layout(
                    xaxis_title="Recall (Sensibilité)",
                    yaxis_title="Précision",
                    legend=dict(x=0.01, y=0.04, bgcolor="rgba(255,255,255,0.85)",
                                bordercolor="#E2E8F0", borderwidth=1),
                    height=400,
                    **_plotly_theme(),
                )
                st.plotly_chart(fig_pr, use_container_width=True)

            st.caption(
                "Courbes calculées sur le jeu de test (20 % des données, stratifié). "
                "PR-AUC est la métrique de référence pour les données déséquilibrées."
            )

    # ── Tab 4 : Radar & Temps ─────────────────────────────────────────────────
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        col_radar, col_time = st.columns(2)

        with col_radar:
            section_header("Radar des performances globales", "🕸️")
            radar_metrics = ["Recall ↑", "F1-Score", "ROC-AUC", "PR-AUC ↑", "Precision"]
            avail_radar   = [m for m in radar_metrics if m in cdf.columns]
            fig_radar = go.Figure()
            for _, row in cdf.iterrows():
                mn   = row.get("Modèle", "")
                vals = [row.get(m, 0) for m in avail_radar] + [row.get(avail_radar[0], 0)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=avail_radar + [avail_radar[0]],
                    name=mn, fill="toself", opacity=0.4,
                    line=dict(color=MODEL_COLORS.get(mn, "#3B82F6"), width=2),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0.5, 1], tickfont=dict(size=9))),
                legend=dict(orientation="h", y=-0.12),
                height=390,
                **_plotly_theme(),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_time:
            section_header("Temps d'entraînement", "⏱️")
            if "Temps (s)" in cdf.columns:
                fig_time = px.bar(
                    cdf, x="Modèle", y="Temps (s)", color="Modèle",
                    color_discrete_map=MODEL_COLORS, text_auto=".1f",
                )
                fig_time.update_traces(texttemplate="%{y:.1f} s", textposition="outside")
                fig_time.update_layout(
                    showlegend=False, yaxis_title="Secondes", xaxis_title="",
                    height=390,
                    **_plotly_theme(),
                )
                st.plotly_chart(fig_time, use_container_width=True)

        # Justification
        st.markdown("---")
        section_header("Justification du modèle retenu", "🏆")
        best  = arts.get("model_name", "")
        thresh = arts.get("threshold", 0.5)
        best_row = cdf[cdf["Modèle"] == best]

        if not best_row.empty:
            r = best_row.iloc[0]
            col_j1, col_j2 = st.columns([1, 2])
            color = MODEL_COLORS.get(best, "#3B82F6")

            with col_j1:
                st.markdown(
                    f'<div style="background:white;border-radius:12px;padding:1.1rem 1.2rem;'
                    f'box-shadow:0 2px 8px rgba(0,0,0,0.08);border-top:4px solid {color}">'
                    f'<div style="font-size:0.7rem;color:#64748B;font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:0.07em">Modèle retenu</div>'
                    f'<div style="font-size:1.2rem;font-weight:700;color:{color};margin:0.3rem 0">{best}</div>'
                    f'<hr style="border-color:#E2E8F0;margin:0.5rem 0">'
                    f'<table style="width:100%;font-size:0.83rem;border-collapse:collapse">'
                    f'<tr><td style="color:#64748B;padding:0.2rem 0">Recall</td>'
                    f'<td style="text-align:right;font-weight:600">{r.get("Recall ↑","N/A")}</td></tr>'
                    f'<tr><td style="color:#64748B;padding:0.2rem 0">F1-Score</td>'
                    f'<td style="text-align:right;font-weight:600">{r.get("F1-Score","N/A")}</td></tr>'
                    f'<tr><td style="color:#64748B;padding:0.2rem 0">PR-AUC</td>'
                    f'<td style="text-align:right;font-weight:600">{r.get("PR-AUC ↑","N/A")}</td></tr>'
                    f'<tr><td style="color:#64748B;padding:0.2rem 0">Seuil</td>'
                    f'<td style="text-align:right;font-weight:600">{thresh:.2f}</td></tr>'
                    f'</table></div>',
                    unsafe_allow_html=True,
                )

            with col_j2:
                st.info(f"""
**Critère de sélection : 70 % Recall + 30 % F1**

Dans ce contexte de maintenance industrielle, la priorité absolue est le **Recall** (sensibilité).

**Pourquoi ?**
- Un **faux négatif** (panne non détectée) → arrêt non planifié → coût estimé **15 000–50 000 €/h**
- Un **faux positif** (alerte inutile) → inspection préventive → coût limité **~2 000 €**

Le modèle **{best}** offre le meilleur compromis sensibilité / stabilité, avec un seuil de décision
de **{thresh:.2f}** optimisé pour maximiser le Recall sur le jeu de test.
                """)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Interprétabilité SHAP
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_LABELS = {
    "vibration_rms":          "Vibration RMS",
    "temperature_motor":      "Température moteur",
    "current_phase_avg":      "Courant phase moyen",
    "pressure_level":         "Niveau de pression",
    "rpm":                    "Vitesse (RPM)",
    "hours_since_maintenance":"Heures depuis maintenance",
    "ambient_temp":           "Température ambiante",
}


def page_interpretability(arts: dict) -> None:
    st.markdown(
        '<h1 style="font-size:1.75rem;font-weight:700;color:#0F172A;'
        'border-bottom:3px solid #3B82F6;padding-bottom:0.5rem;margin-bottom:1.5rem">'
        '🔍 Interprétabilité SHAP — Pourquoi ce modèle prédit une panne ?</h1>',
        unsafe_allow_html=True,
    )

    shap_df    = arts.get("shap_df", pd.DataFrame())
    model_name = arts.get("model_name", "Modèle")

    if shap_df.empty:
        st.warning("Analyse SHAP non disponible. Lancez : `python train.py`")
        with st.expander("📚 Comprendre SHAP"):
            _shap_methodology()
        return

    section_header(f"Importance globale des variables — {model_name}", "📊")

    df_sorted = shap_df.sort_values("mean_abs_shap", ascending=True).tail(14)
    df_sorted["label"] = df_sorted["feature"].map(
        lambda f: _FEATURE_LABELS.get(f, f)
    )

    fig_bar = go.Figure(go.Bar(
        x=df_sorted["mean_abs_shap"],
        y=df_sorted["label"],
        orientation="h",
        marker=dict(
            color=df_sorted["mean_abs_shap"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="SHAP moyen", thickness=12, len=0.8),
        ),
        text=[f"{v:.4f}" for v in df_sorted["mean_abs_shap"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP moyen : %{x:.5f}<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis_title="Valeur SHAP absolue moyenne (impact sur la prédiction de panne)",
        yaxis_title="",
        height=max(350, len(df_sorted) * 30),
        margin=dict(t=20, b=30, l=10, r=70),
        font=dict(family="Inter, sans-serif"),
        template="plotly_white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Top features cards
    st.markdown("---")
    section_header("Top variables — Explication en langage naturel", "💬")

    top   = shap_df.head(min(5, len(shap_df)))
    icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    cols_feat = st.columns(min(len(top), 5))

    for i, (_, row) in enumerate(top.iterrows()):
        label = _FEATURE_LABELS.get(row["feature"], row["feature"])
        val   = row["mean_abs_shap"]
        with cols_feat[i]:
            st.markdown(
                f'<div style="background:white;border-radius:10px;padding:0.9rem;'
                f'box-shadow:0 1px 6px rgba(0,0,0,0.07);text-align:center;'
                f'border-top:3px solid #3B82F6">'
                f'<div style="font-size:1.4rem">{icons[i]}</div>'
                f'<div style="font-size:0.77rem;font-weight:600;color:#1E3A5F;'
                f'margin:0.3rem 0;line-height:1.3">{label}</div>'
                f'<div style="font-size:0.88rem;font-weight:700;color:#3B82F6">{val:.4f}</div>'
                f'<div style="font-size:0.7rem;color:#94A3B8">impact moyen</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if len(top) >= 2:
        l1 = _FEATURE_LABELS.get(top.iloc[0]["feature"], top.iloc[0]["feature"])
        l2 = _FEATURE_LABELS.get(top.iloc[1]["feature"], top.iloc[1]["feature"])
        l3 = (_FEATURE_LABELS.get(top.iloc[2]["feature"], top.iloc[2]["feature"])
              if len(top) >= 3 else "N/A")
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"""
**Le modèle fonde ses prédictions principalement sur :**

1. **{l1}** — variable la plus discriminante pour détecter une panne imminente
2. **{l2}** — deuxième facteur d'influence sur la prédiction
3. **{l3}** — troisième variable clé

**Comment agir ?** Surveiller ces indicateurs en priorité et déclencher une inspection
dès qu'un seuil d'alerte est franchi sur l'un d'eux.
        """)

    # Static SHAP plots
    st.markdown("---")
    section_header("Visualisations SHAP statiques (générées à l'entraînement)", "🖼️")

    shap_summary_p = PROJECT_ROOT / "models" / "shap_summary.png"
    shap_bar_p     = PROJECT_ROOT / "models" / "shap_bar.png"

    col_s, col_b = st.columns(2)
    if shap_summary_p.exists():
        with col_s:
            st.subheader("Summary Plot")
            st.image(str(shap_summary_p), use_container_width=True)
            st.caption(
                "Chaque point = une observation. Couleur = valeur de la feature "
                "(rouge élevé, bleu faible). Position X = impact sur la prédiction."
            )
    if shap_bar_p.exists():
        with col_b:
            st.subheader("Bar Plot")
            st.image(str(shap_bar_p), use_container_width=True)
            st.caption(
                "Importance globale : valeur SHAP absolue moyenne sur le test set. "
                "Plus la barre est longue, plus la variable est influente."
            )

    # Methodology expander
    with st.expander("📚 Comprendre les valeurs SHAP"):
        _shap_methodology()


def _shap_methodology() -> None:
    st.markdown("""
**SHAP (SHapley Additive exPlanations)** mesure la contribution de chaque variable à une prédiction.

**Interprétation :**
- **SHAP positif** → la variable pousse vers une prédiction "Panne"
- **SHAP négatif** → la variable pousse vers une prédiction "Normal"
- **Magnitude** → plus la valeur absolue est grande, plus la variable est influente

**Exemple :**
- Vibration RMS élevée → SHAP fort positif → augmente la probabilité de panne
- Température dans la normale → SHAP négatif → signe de bon fonctionnement

**Méthodes utilisées :**

| Modèle | Explainer SHAP | Propriété |
|--------|----------------|-----------|
| XGBoost / Random Forest | TreeExplainer | Exact, rapide |
| Logistic Regression | LinearExplainer | Exact |
| MLP PyTorch | KernelExplainer | Approximatif (échantillon) |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar + Navigation
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    arts = load_artifacts()
    df   = load_dataset()

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.5rem 0 1.2rem">
          <div style="font-size:2.2rem">⚙️</div>
          <div style="font-size:1.05rem;font-weight:700;color:#E2E8F0;letter-spacing:0.02em">
            PredictMaint
          </div>
          <div style="font-size:0.72rem;color:#64748B;margin-top:0.2rem">
            Supervision Industrielle
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏭 Vue d'ensemble", "🔮 Simulation", "📊 Comparaison modèles", "🔍 Interprétabilité"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Model status card
        model_name = arts.get("model_name", "Non entraîné")
        threshold  = arts.get("threshold", 0.5)
        model_ok   = arts.get("model") is not None

        st.markdown(
            '<div style="font-size:0.7rem;font-weight:600;color:#64748B;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem">'
            'Statut du modèle</div>',
            unsafe_allow_html=True,
        )

        if model_ok:
            color = MODEL_COLORS.get(model_name, "#3B82F6")
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.05);border-radius:10px;'
                f'padding:0.75rem;border:1px solid rgba(255,255,255,0.1);'
                f'border-left:3px solid {color}">'
                f'  <div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.4rem">'
                f'    <span style="width:8px;height:8px;border-radius:50%;'
                f'background:#06D6A0;display:inline-block;flex-shrink:0"></span>'
                f'    <span style="font-size:0.78rem;font-weight:600;color:#06D6A0">Opérationnel</span>'
                f'  </div>'
                f'  <div style="font-size:0.85rem;font-weight:600;color:#E2E8F0">{model_name}</div>'
                f'  <div style="font-size:0.73rem;color:#94A3B8;margin-top:0.2rem">Seuil : {threshold:.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(239,68,68,0.1);border-radius:10px;'
                'padding:0.75rem;border:1px solid rgba(239,68,68,0.3)">'
                '  <div style="color:#EF4444;font-size:0.83rem;font-weight:600">⚠️ Modèle non chargé</div>'
                '  <div style="color:#94A3B8;font-size:0.73rem;margin-top:0.3rem">'
                '  Lancez :<br><code>python train.py</code></div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Eval artifacts status
        eval_arts = arts.get("eval_artifacts", {})
        n_eval    = len([k for k in eval_arts if k != "y_test"])
        if eval_arts:
            st.markdown(
                f'<div style="margin-top:0.5rem;font-size:0.72rem;color:#64748B">'
                f'✓ Artefacts d\'éval. : {n_eval} modèle(s)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="margin-top:0.5rem;font-size:0.72rem;color:#475569">'
                '○ Artefacts d\'éval. non disponibles<br>'
                '<span style="font-size:0.68rem">(relancez train.py)</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        if not df.empty:
            n_fail = int(df["failure_within_24h"].sum())
            st.markdown(
                f'<div style="font-size:0.72rem;color:#64748B;line-height:1.8">'
                f'📊 Dataset : {len(df):,} observations<br>'
                f'⚠️ Pannes : {n_fail:,} ({n_fail/len(df)*100:.1f}%)<br>'
                f'🏭 Machines : CNC · Pump · Comp · Arm'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.68rem;color:#475569;line-height:1.7">'
            'EFREI M1 Data Engineering<br>'
            'RNCP40875 — Bloc 2<br>'
            'Maintenance Prédictive Industrielle'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Route pages ──────────────────────────────────────────────────────────
    if "Vue d'ensemble" in page:
        page_overview(df)
    elif "Simulation" in page:
        page_simulation(arts)
    elif "Comparaison" in page:
        page_models(arts)
    elif "Interprétabilité" in page:
        page_interpretability(arts)


if __name__ == "__main__":
    main()
