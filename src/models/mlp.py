"""Modèle Deep Learning : MLP (Multi-Layer Perceptron) avec PyTorch.

Architecture justifiée :
- 3 couches cachées avec BatchNorm + ReLU + Dropout pour régularisation
- Dropout (0.4) réduit l'overfitting sur les 19 233 exemples d'entraînement
- BatchNorm accélère la convergence et stabilise le gradient
- pos_weight dans BCEWithLogitsLoss gère le déséquilibre 5.8:1 directement
  dans la fonction de perte, sans modifier la distribution des données

Early stopping sur val_loss évite l'overfitting sans tuning du nombre d'epochs.
Le MLP est comparé aux modèles sklearn pour vérifier si la complexité apporte
un gain réel — dans un contexte tabulaire, ce n'est pas toujours le cas.
"""
from __future__ import annotations

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "mlp.pkl"

# Ratio classes pour pos_weight (5.75 ≈ 20482 / 3560)
POS_WEIGHT = torch.tensor([20482 / 3560], dtype=torch.float32)


class MaintenanceMLP(nn.Module):
    """Réseau de neurones MLP pour la classification binaire de pannes.

    Architecture : Input → [Dense → BN → ReLU → Dropout] × 3 → Output (logit)

    Args:
        input_dim: Nombre de features après transformation (numérique + OHE).
        hidden_dims: Tailles des couches cachées.
        dropout_rate: Taux de Dropout (0.3–0.5 recommandé).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Logit de sortie

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    patience: int = 8,
    val_fraction: float = 0.1,
) -> tuple[MaintenanceMLP, dict]:
    """Entraîne le MLP PyTorch avec early stopping sur val_loss.

    Args:
        X_train: Features d'entraînement transformées (array numpy).
        y_train: Labels d'entraînement.
        epochs: Nombre maximum d'epochs.
        batch_size: Taille des mini-batchs.
        learning_rate: Taux d'apprentissage initial.
        patience: Nombre d'epochs sans amélioration avant arrêt précoce.
        val_fraction: Fraction du train réservée à la validation interne.

    Returns:
        Tuple (modèle_entraîné, historique_loss) où historique contient
        les listes train_loss et val_loss par epoch.
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"  [MLP] Entraînement sur : {device}")

    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.array(X_train, dtype=np.float32)

    # Split validation interne (stratifié manuellement)
    n_val = int(len(X_train) * val_fraction)
    # Trier par label pour stratifier approximativement
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_val_pos = int(n_val * len(pos_idx) / len(y_train))
    n_val_neg = n_val - n_val_pos

    rng = np.random.default_rng(RANDOM_STATE)
    val_pos = rng.choice(pos_idx, n_val_pos, replace=False)
    val_neg = rng.choice(neg_idx, n_val_neg, replace=False)
    val_idx = np.concatenate([val_pos, val_neg])
    train_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # Datasets et DataLoaders PyTorch
    train_ds = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = MaintenanceMLP(input_dim=input_dim).to(device)

    # pos_weight : pénalise 5.75× plus les faux négatifs → optimise le Recall
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=POS_WEIGHT.to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Réduction du LR si la val_loss stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "val_recall": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- Entraînement ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses, val_preds, val_true = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.cpu().numpy().astype(int))

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_recall = recall_score(val_true, val_preds, zero_division=0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_recall"].append(val_recall)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Recall: {val_recall:.3f}")

        # Early stopping sur val_loss
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping déclenché à l'epoch {epoch+1} (patience={patience})")
                break

    # Restaurer le meilleur état
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, history


def evaluate(
    model: MaintenanceMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Évalue le MLP sur le jeu de test avec métriques complètes.

    Args:
        model: MaintenanceMLP fitté (sur CPU).
        X_test: Features de test transformées (array numpy).
        y_test: Labels de test réels.
        threshold: Seuil de décision probabiliste.

    Returns:
        Dictionnaire des métriques : accuracy, precision, recall, f1, roc_auc, pr_auc.
    """
    model.eval()
    # Détecter le device du modèle et y envoyer l'input (évite le device mismatch MPS/CPU)
    device = next(model.parameters()).device
    X_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32)).to(device)
    y_test = np.array(y_test, dtype=int)

    with torch.no_grad():
        logits = model(X_tensor)
        y_proba = torch.sigmoid(logits).cpu().numpy()

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "model": "MLP (PyTorch)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "threshold": threshold,
    }


def save_model(model: MaintenanceMLP, path: Path = MODEL_PATH) -> None:
    """Sérialise le modèle PyTorch avec torch.save (plus fiable que joblib pour PyTorch).

    Sauvegarde le state_dict + la configuration de l'architecture pour permettre
    la reconstruction du modèle au chargement.

    Args:
        model: MaintenanceMLP entraîné (sur CPU).
        path: Chemin de sauvegarde (.pkl conservé pour compatibilité avec train.py).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sauvegarder state_dict + config architecture pour pouvoir reconstruire le modèle
    first_layer = model.network[0]
    input_dim = first_layer.in_features
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "hidden_dims": (256, 128, 64),
        "dropout_rate": 0.4,
    }, path)
    print(f"  Modèle sauvegardé : {path}")


def load_model(path: Path = MODEL_PATH) -> MaintenanceMLP:
    """Charge un modèle MLP sauvegardé avec save_model().

    Args:
        path: Chemin du fichier sauvegardé.

    Returns:
        MaintenanceMLP reconstruit et prêt pour l'inférence (sur CPU, en eval mode).
    """
    checkpoint = torch.load(path, map_location="cpu")
    model = MaintenanceMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout_rate=checkpoint["dropout_rate"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
