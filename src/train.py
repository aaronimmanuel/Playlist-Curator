"""Step 2: train Logistic Regression, RBF SVM and the PyTorch neural network.

Run from the project root:
    python -m src.train
"""

import joblib
import matplotlib

matplotlib.use("Agg")  # draw figures to files, no window needed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_xy
from src.models import TorchMLPClassifier, build_logistic_regression, build_svm


def plot_nn_history(history: dict) -> None:
    """Save the neural network's loss and validation-accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train loss")
    ax_loss.plot(epochs, history["val_loss"], label="validation loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Neural network: loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["val_accuracy"], color="tab:green")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Neural network: validation accuracy")

    fig.tight_layout()
    path = config.FIGURES_DIR / "nn_training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_xy("train")
    print(f"Training data: {X_train.shape[0]} songs, {X_train.shape[1]} features, "
          f"{len(set(y_train))} playlists\n")

    # ---- 1. Logistic Regression ------------------------------------------
    print("Training Logistic Regression ...")
    log_reg = build_logistic_regression().fit(X_train, y_train)
    joblib.dump(log_reg, config.MODEL_PATHS["logistic_regression"])
    print(f"  training accuracy: {log_reg.score(X_train, y_train):.3f}\n")

    # ---- 2. RBF SVM --------------------------------------------------------
    print("Training RBF SVM ...")
    svm = build_svm().fit(X_train, y_train)
    joblib.dump(svm, config.MODEL_PATHS["svm_rbf"])
    print(f"  training accuracy: {svm.score(X_train, y_train):.3f}\n")

    # ---- 3. PyTorch neural network ----------------------------------------
    # Hold out a small validation set from the TRAINING data for early stopping,
    # so the test set stays untouched until the final evaluation.
    print("Training PyTorch neural network ...")
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=config.VAL_SIZE,
        stratify=y_train,
        random_state=config.RANDOM_STATE,
    )
    net = TorchMLPClassifier(**config.NN_PARAMS).fit(X_fit, y_fit, X_val, y_val)
    net.save(config.MODEL_PATHS["neural_net"])
    nn_train_acc = (net.predict(X_train) == y_train).mean()
    print(f"  training accuracy: {nn_train_acc:.3f}\n")

    plot_nn_history(net.history)

    print("\nAll models saved to:", config.MODELS_DIR)


if __name__ == "__main__":
    main()
