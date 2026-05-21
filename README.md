# Système de Maintenance Prédictive Industrielle

**EFREI M1 Data Engineering — Épreuve certifiante RNCP40875 Bloc 2**

Système complet de classification binaire pour prédire les pannes machine dans les 24 heures sur une ligne d'assemblage automobile.

---

## Architecture du projet

```
predictive_maintenance/
├── data/
│   └── predictive_maintenance_v3.csv   # 24 042 obs., 15 colonnes
├── notebooks/
│   └── 01_EDA.ipynb                    # Analyse exploratoire
├── src/
│   ├── data/
│   │   ├── preprocessing.py            # Pipeline sklearn (imputation + OHE + StandardScaler)
│   │   └── imbalance.py                # Comparaison SMOTE / ROS / RUS / class_weight
│   ├── models/
│   │   ├── baseline.py                 # Logistic Regression
│   │   ├── random_forest.py            # Random Forest + GridSearchCV
│   │   ├── gradient_boosting.py        # XGBoost + RandomizedSearchCV
│   │   └── mlp.py                      # MLP PyTorch (BatchNorm + Dropout + early stopping)
│   └── evaluation/
│       └── metrics.py                  # Métriques, ROC, PR-AUC, SHAP, seuil décision
├── models/                             # Artefacts générés après entraînement
│   ├── best_model.pkl
│   ├── preprocessing_pipeline.pkl
│   └── comparison_report.csv
├── dashboard/
│   └── app.py                          # Interface Streamlit 4 pages
├── api/
│   ├── main.py                         # API REST FastAPI
│   └── schemas.py                      # Validation Pydantic
├── train.py                            # Script principal d'entraînement
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Entraîner les modèles

```bash
# Entraînement complet (tous les modèles)
python train.py

# Options disponibles
python train.py --skip-imbalance          # Ignorer comparaison des méthodes de rééquilibrage
python train.py --model rf                # Entraîner uniquement Random Forest (lr/rf/xgb/mlp)
python train.py --threshold-metric recall # Optimiser le seuil sur le Recall
python train.py --xgb-iter 50            # Plus d'itérations RandomizedSearchCV pour XGBoost
```

### 2. Lancer le dashboard Streamlit

```bash
streamlit run dashboard/app.py
```

Ouvre automatiquement `http://localhost:8501` avec 4 pages :
- **Vue d'ensemble** : KPIs, distribution des pannes, timeline
- **Simulation** : formulaire capteurs → probabilité de panne + jauge
- **Comparaison modèles** : tableau, ROC, PR-AUC, radar
- **Interprétabilité** : SHAP global, top features, explications naturelles

### 3. Lancer l'API FastAPI (optionnel)

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation Swagger : `http://localhost:8000/docs`

```bash
# Exemple d'appel API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_type": "CNC",
    "operating_mode": "peak",
    "vibration_rms": 4.2,
    "temperature_motor": 78.0,
    "current_phase_avg": 14.5,
    "pressure_level": 35.0,
    "rpm": 2000.0,
    "hours_since_maintenance": 380.0,
    "ambient_temp": 16.0
  }'
```

### 4. Explorer le notebook EDA

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## Dataset

| Caractéristique | Valeur |
|----------------|--------|
| Observations | 24 042 |
| Features | 15 colonnes |
| Tâche | Classification binaire (`failure_within_24h`) |
| Déséquilibre | 5.8:1 (20 482 "0" vs 3 560 "1") |
| Valeurs manquantes | vibration_rms (1000), pressure_level (924), temperature_motor (834) |
| Machines | CNC, Pump, Compressor, Robotic Arm |
| Types de pannes | bearing, motor_overheat, hydraulic, electrical |

---

## Modèles implémentés

| Modèle | Technique | Gestion déséquilibre | Optimisation |
|--------|-----------|---------------------|--------------|
| Logistic Regression | Baseline interprétable | `class_weight="balanced"` | Stratified K-Fold CV |
| Random Forest | Ensemble / bagging | `class_weight="balanced"` | GridSearchCV |
| XGBoost | Gradient Boosting | `scale_pos_weight=5.75` | RandomizedSearchCV + early stopping |
| MLP PyTorch | Deep Learning | `pos_weight` dans BCEWithLogitsLoss | Early stopping sur val_loss |

---

## Métriques prioritaires

Dans ce contexte industriel, un **faux négatif** (panne non détectée) entraîne un arrêt machine non planifié, dont le coût dépasse largement celui d'un **faux positif** (alerte inutile).

**Ordre de priorité :** Recall > PR-AUC > F1-Score > ROC-AUC > Accuracy

L'accuracy n'est pas utilisée comme critère de sélection (un classifieur trivial "tout 0" atteindrait 85.2%).

---

## Pipeline de données

```
CSV brut
  ↓  load_data()
DataFrame Pandas
  ↓  train_test_split (stratify=y, 80/20, random_state=42)
X_train / X_test
  ↓  pipeline.fit_transform(X_train) [JAMAIS sur X_test seul]
  ↓  pipeline.transform(X_test)
Features transformées :
  - 7 numériques : imputation médiane + StandardScaler
  - 2 catégorielles OHE : machine_type (4) + operating_mode (3) = 7 dummies
  = 14 features totales
```

**Contrainte anti data leakage** : le `fit()` du pipeline est fait **uniquement** sur `X_train`.

---

## Gestion du déséquilibre

5 méthodes comparées par Stratified K-Fold (k=5) sur le train set :

1. Baseline (aucun rééquilibrage)
2. Random Over-Sampling (duplication aléatoire de la classe minoritaire)
3. SMOTE (génération synthétique d'exemples minoritaires)
4. Random Under-Sampling (suppression aléatoire de la classe majoritaire)
5. `class_weight="balanced"` (pondération dans la loss)

**Important** : SMOTE est appliqué uniquement sur chaque fold d'entraînement, jamais sur le fold de validation.

---

## Interprétabilité SHAP

- **TreeExplainer** pour Random Forest et XGBoost (exact, rapide)
- **LinearExplainer** pour la Logistic Regression
- **KernelExplainer** pour le MLP (model-agnostic, sur échantillon)

Visualisations générées :
- Summary plot (importance + direction de l'effet par feature)
- Bar plot (importance globale en valeur absolue)
- Force plot (explication individuelle d'une prédiction)

---

## Choix techniques justifiés

| Choix | Justification |
|-------|---------------|
| PyTorch pour le MLP | Flexibilité pour `pos_weight`, MPS (Apple Silicon), early stopping natif |
| XGBoost plutôt que LightGBM | `scale_pos_weight` explicite, meilleure interprétation industrielle |
| Seuil de décision ajustable | Le seuil 0.5 par défaut sous-optimise le Recall sur données déséquilibrées |
| Stratified K-Fold partout | Préserve le ratio 5.8:1 dans chaque fold de validation |
| joblib pour la sérialisation | Standard industrie, rapide pour sklearn et PyTorch |
| `random_state=42` partout | Reproductibilité complète des expériences |
