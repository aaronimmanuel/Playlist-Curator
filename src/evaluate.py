"""Step 3: evaluate all trained models on the held-out test set.

Run from the project root:
    python -m src.evaluate

Creates:
    reports/model_comparison.csv
    reports/classification_report_<model>.txt
    reports/figures/confusion_matrix_<model>.png
    reports/figures/model_comparison.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from src import config
from src.data_loader import load_preprocessing_artifacts, load_xy
from src.models import load_model


def plot_confusion_matrix(y_true, y_pred, class_names, title, path) -> None:
    """Save a confusion matrix where each row is normalised to sum to 1 (= recall)."""
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted playlist")
    ax.set_ylabel("True playlist")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame) -> None:
    ax = results.set_index("model")[["accuracy", "macro_f1", "top3_accuracy"]].plot(
        kind="bar", figsize=(8, 5), ylim=(0, 1.05), rot=0
    )
    ax.set_ylabel("score")
    ax.set_title("Test-set performance by model")
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(ax.figure)


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _, label_encoder = load_preprocessing_artifacts()
    class_names = list(label_encoder.classes_)
    X_train, y_train = load_xy("train")
    X_test, y_test = load_xy("test")
    print(f"Evaluating on {len(y_test)} test songs ({len(class_names)} playlists)\n")

    rows = []
    for name in config.MODEL_PATHS:
        display_name = config.MODEL_DISPLAY_NAMES[name]
        model = load_model(name)

        # Predictions are the highest-probability playlist, i.e. exactly what the
        # Streamlit app shows as its top recommendation.
        test_proba = model.predict_proba(X_test)
        y_pred = test_proba.argmax(axis=1)
        train_pred = model.predict_proba(X_train).argmax(axis=1)

        rows.append({
            "model": display_name,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "top3_accuracy": top_k_accuracy_score(
                y_test, test_proba, k=min(3, len(class_names)), labels=range(len(class_names))
            ),
        })

        report = classification_report(
            y_test, y_pred, labels=range(len(class_names)),
            target_names=class_names, zero_division=0,
        )
        (config.REPORTS_DIR / f"classification_report_{name}.txt").write_text(
            f"{display_name}\n\n{report}"
        )
        plot_confusion_matrix(
            y_test, y_pred, class_names, f"{display_name}: confusion matrix (test set)",
            config.FIGURES_DIR / f"confusion_matrix_{name}.png",
        )

        print(f"=== {display_name} ===")
        print(report)

    results = pd.DataFrame(rows)
    results.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    plot_model_comparison(results)

    print("Model comparison")
    print(results.round(3).to_string(index=False))
    print(f"\nReports saved in {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
