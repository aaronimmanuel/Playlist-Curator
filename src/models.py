"""The three classifiers used in this project.

* Logistic Regression  (scikit-learn)
* SVM with RBF kernel  (scikit-learn)
* A small feed-forward neural network (PyTorch)

All three expose the same two methods used by the rest of the code:
    model.predict_proba(X)  -> array of shape (n_songs, n_playlists)
    model.predict(X)        -> array of playlist indices
"""

import copy

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config


# --------------------------------------------------------------------------
# scikit-learn models
# --------------------------------------------------------------------------
def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(**config.LOGREG_PARAMS)


def build_svm() -> SVC:
    return SVC(**config.SVM_PARAMS)


# --------------------------------------------------------------------------
# PyTorch neural network
# --------------------------------------------------------------------------
class PlaylistNet(nn.Module):
    """Input -> [Linear -> ReLU -> Dropout] x N -> Linear (one score per playlist)."""

    def __init__(self, input_dim: int, n_classes: int, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)]
            previous = hidden
        layers.append(nn.Linear(previous, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchMLPClassifier:
    """Small scikit-learn-style wrapper around PlaylistNet (fit / predict / save / load)."""

    def __init__(self, hidden_dims=(64, 32), dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 batch_size=64, max_epochs=200, patience=20, seed=42):
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.net = None
        self.input_dim = None
        self.n_classes = None
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    # ---- training ---------------------------------------------------------
    def fit(self, X_train, y_train, X_val, y_val):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.input_dim = X_train.shape[1]
        self.n_classes = int(max(y_train.max(), y_val.max())) + 1
        self.net = PlaylistNet(self.input_dim, self.n_classes, self.hidden_dims, self.dropout)

        # Give rarer playlists a larger weight so the network does not ignore them.
        weights = compute_class_weight("balanced", classes=np.arange(self.n_classes), y=y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.long)
        X_va = torch.tensor(X_val, dtype=torch.float32)
        y_va = torch.tensor(y_val, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            # one pass over the training data
            self.net.train()
            running_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(xb)
            train_loss = running_loss / len(X_tr)

            # check progress on the validation set
            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(X_va)
                val_loss = criterion(val_logits, y_va).item()
                val_acc = (val_logits.argmax(dim=1) == y_va).float().mean().item()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            if epoch == 1 or epoch % 10 == 0:
                print(f"  epoch {epoch:3d} | train loss {train_loss:.4f} | "
                      f"val loss {val_loss:.4f} | val acc {val_acc:.3f}")

            # early stopping: remember the best weights, stop when no more progress
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"  early stopping at epoch {epoch}")
                    break

        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    # ---- prediction -------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(np.asarray(X), dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ---- saving / loading -------------------------------------------------
    def save(self, path) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "n_classes": self.n_classes,
                    "hidden_dims": self.hidden_dims,
                    "dropout": self.dropout,
                },
                "history": self.history,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "TorchMLPClassifier":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        cfg = checkpoint["config"]
        model = cls(hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"])
        model.input_dim = cfg["input_dim"]
        model.n_classes = cfg["n_classes"]
        model.net = PlaylistNet(cfg["input_dim"], cfg["n_classes"],
                                cfg["hidden_dims"], cfg["dropout"])
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.eval()
        model.history = checkpoint.get("history", model.history)
        return model


# --------------------------------------------------------------------------
# One place to load any saved model by name
# --------------------------------------------------------------------------
def load_model(name: str):
    """Load a trained model: 'logistic_regression', 'svm_rbf' or 'neural_net'."""
    path = config.MODEL_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train` first.")
    if name == "neural_net":
        return TorchMLPClassifier.load(path)
    return joblib.load(path)
