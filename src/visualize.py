"""Step 4: PCA visualisation of the song feature space.

Run from the project root:
    python -m src.visualize

Creates:
    reports/figures/pca_playlists.png         (2-D map of songs, coloured by playlist)
    reports/figures/pca_explained_variance.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src import config
from src.data_loader import load_preprocessing_artifacts, load_xy


def scatter_by_playlist(ax, points, labels, class_names, title) -> None:
    colormap = plt.get_cmap("tab10" if len(class_names) <= 10 else "tab20")
    for index, name in enumerate(class_names):
        mask = labels == index
        ax.scatter(points[mask, 0], points[mask, 1], s=14, alpha=0.65,
                   color=colormap(index), label=name)
    ax.set_title(title)
    ax.grid(alpha=0.2)


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _, label_encoder = load_preprocessing_artifacts()
    class_names = list(label_encoder.classes_)
    X_train, y_train = load_xy("train")
    X_test, y_test = load_xy("test")

    # Fit PCA on the training songs only, then project both sets with it.
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE).fit(X_train)
    train_2d = pca.transform(X_train)
    test_2d = pca.transform(X_test)
    var1, var2 = pca.explained_variance_ratio_ * 100
    print(f"PC1 explains {var1:.1f}% of the variance, PC2 explains {var2:.1f}%")

    # ---- Figure 1: songs in 2-D, train and test side by side ----------------
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    scatter_by_playlist(ax_train, train_2d, y_train, class_names,
                        f"Training songs (n={len(y_train)})")
    scatter_by_playlist(ax_test, test_2d, y_test, class_names,
                        f"Test songs (n={len(y_test)})")
    for ax in (ax_train, ax_test):
        ax.set_xlabel(f"PC1 ({var1:.1f}% of variance)")
    ax_train.set_ylabel(f"PC2 ({var2:.1f}% of variance)")
    handles, labels = ax_test.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Playlist", markerscale=1.8)
    fig.suptitle("PCA of scaled audio features + key + genre tags")
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    path = config.FIGURES_DIR / "pca_playlists.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # ---- Figure 2: how much variance do the first components capture? -------
    n_components = min(20, X_train.shape[1])
    full_pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE).fit(X_train)
    ratios = full_pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, n_components + 1), ratios * 100, label="per component")
    ax.plot(range(1, n_components + 1), np.cumsum(ratios) * 100,
            color="tab:red", marker="o", label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance (%)")
    ax.set_title("PCA explained variance")
    ax.set_xticks(range(1, n_components + 1))
    ax.legend()
    fig.tight_layout()
    path = config.FIGURES_DIR / "pca_explained_variance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
