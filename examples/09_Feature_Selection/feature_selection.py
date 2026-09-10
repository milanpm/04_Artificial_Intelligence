"""
File: feature_selection.py
Description: Selects useful features with SelectKBest and determines
             the optimal number of features using cross-validation.
Author: Alex
Date: 2026-09-10
"""

from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "informative_1",
    "informative_2",
    "informative_3",
    "informative_4",
    "informative_5",
    "noise_1",
    "noise_2",
    "noise_3",
]


def create_dataset():
    """Create a classification dataset with useful and noisy features."""
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.74, 0.26],
        class_sep=1.0,
        random_state=42,
        shuffle=False,
    )

    return train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )


def main():
    X_train, X_test, y_train, y_test = create_dataset()

    print("AI Expert Day 9 - Feature Selection")
    print("=" * 45)
    print(f"Feature count: {len(FEATURE_NAMES)}")
    print(f"Feature names: {FEATURE_NAMES}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training positive ratio: {y_train.mean():.2%}")
    print(f"Test positive ratio: {y_test.mean():.2%}")

    selector = SelectKBest(
        score_func=f_classif,
        k=5,
    )

    selector.fit(X_train, y_train)

    print("\nFeature Selection Scores")
    print("-" * 45)

    for feature_name, score, selected in zip(
        FEATURE_NAMES,
        selector.scores_,
        selector.get_support(),
    ):
        status = "Selected" if selected else "Removed"

        print(
            f"{feature_name:<18}"
            f"Score: {score:>8.3f}  "
            f"{status}"
        )

    pipeline = Pipeline([
        (
            "selector",
            SelectKBest(score_func=f_classif),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=3,
                weights="uniform",
                p=2,
            ),
        ),
    ])

    param_grid = {
        "selector__k": list(range(1, len(FEATURE_NAMES) + 1)),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    grid_search.fit(X_train, y_train)

    print("\nCross-Validation Results")
    print("-" * 45)

    mean_scores = grid_search.cv_results_["mean_test_score"]
    std_scores = grid_search.cv_results_["std_test_score"]
    feature_counts = grid_search.cv_results_["param_selector__k"]

    for feature_count, mean_score, std_score in zip(
        feature_counts,
        mean_scores,
        std_scores,
    ):
        print(
            f"k={feature_count}: "
            f"F1={mean_score:.3f} +/- {std_score:.3f}"
        )

    best_feature_count = grid_search.best_params_["selector__k"]

    print("\nBest Feature Selection")
    print("-" * 45)
    print(f"Best number of features: {best_feature_count}")
    print(f"Best mean CV F1: {grid_search.best_score_:.3f}")

    best_selector = grid_search.best_estimator_.named_steps["selector"]
    selected_feature_names = [
        feature_name
        for feature_name, selected in zip(
            FEATURE_NAMES,
            best_selector.get_support(),
        )
        if selected
    ]

    print(f"Selected features: {selected_feature_names}")

    y_pred = grid_search.best_estimator_.predict(X_test)

    print("\nFinal Test Results")
    print("-" * 45)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.3f}")


if __name__ == "__main__":
    main()
