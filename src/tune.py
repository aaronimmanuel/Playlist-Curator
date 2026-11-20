"""Optional step: 5-fold cross-validation and a small hyper-parameter search.

The reference paper evaluated its classical models with 5-fold cross-validation
and tuned the SVM penalty parameter C.  This script does the same on the
TRAINING set only, so the test set stays untouched.

Run from the project root:
    python -m src.tune

Creates:
    reports/cv_results.csv

Then copy the best values it prints into LOGREG_PARAMS / SVM_PARAMS in
src/config.py and re-run `python -m src.train` and `python -m src.evaluate`.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src import config
from src.data_loader import load_xy


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_xy("train")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    searches = {
        "Logistic Regression": GridSearchCV(
            LogisticRegression(max_iter=config.LOGREG_PARAMS["max_iter"],
                               class_weight="balanced"),
            param_grid={"C": [0.01, 0.1, 1, 10, 100]},
            cv=folds, scoring="accuracy", n_jobs=-1,
        ),
        "RBF SVM": GridSearchCV(
            # probability=False makes the search much faster; it does not change
            # which C / gamma is best.
            SVC(kernel="rbf", class_weight="balanced"),
            param_grid={"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.03]},
            cv=folds, scoring="accuracy", n_jobs=-1,
        ),
    }

    all_results = []
    for name, search in searches.items():
        print(f"Running 5-fold cross-validation for {name} ...")
        search.fit(X_train, y_train)

        table = pd.DataFrame(search.cv_results_)
        table.insert(0, "model", name)
        all_results.append(
            table[["model", "params", "mean_test_score", "std_test_score", "rank_test_score"]]
        )

        print(f"  best parameters : {search.best_params_}")
        best_row = table.loc[search.best_index_]
        print(f"  best CV accuracy: {best_row['mean_test_score']:.3f} "
              f"(+/- {best_row['std_test_score']:.3f})\n")

    results = pd.concat(all_results, ignore_index=True)
    path = config.REPORTS_DIR / "cv_results.csv"
    results.to_csv(path, index=False)
    print(f"Full grid saved to {path}")
    print("Copy the best values above into LOGREG_PARAMS / SVM_PARAMS in src/config.py,")
    print("then re-run: python -m src.train  and  python -m src.evaluate")


if __name__ == "__main__":
    main()
