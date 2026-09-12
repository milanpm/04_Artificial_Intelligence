"""
File: feature_importance.py
Description: Compares Random Forest built-in feature importance with
             permutation importance for global model explainability.
Author: Alex
Date: 2026-09-12
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

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
    """Create the same classification dataset used in Day 9."""
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


def print_ranked_importance(title, feature_names, importance_values):
    """Print feature importance values in descending order."""
    ranked_features = sorted(
        zip(feature_names, importance_values),
        key=lambda item: item[1],
        reverse=True,
    )

    print(f"\n{title}")
    print("-" * 55)

    for rank, (feature_name, importance) in enumerate(
        ranked_features,
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{feature_name:<18}"
            f"Importance: {importance:.4f}"
        )

    return ranked_features


def main():
    X_train, X_test, y_train, y_test = create_dataset()

    print("AI Expert Day 10 - Feature Importance and Explainability")
    print("=" * 60)
    print(f"Feature count: {len(FEATURE_NAMES)}")
    print(f"Feature names: {FEATURE_NAMES}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training positive ratio: {y_train.mean():.2%}")
    print(f"Test positive ratio: {y_test.mean():.2%}")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel Performance")
    print("-" * 55)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.3f}")

    built_in_ranking = print_ranked_importance(
        title="Random Forest Built-in Feature Importance",
        feature_names=FEATURE_NAMES,
        importance_values=model.feature_importances_,
    )

    permutation_result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring="f1",
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )

    permutation_ranking = print_ranked_importance(
        title="Permutation Importance",
        feature_names=FEATURE_NAMES,
        importance_values=permutation_result.importances_mean,
    )

    print("\nPermutation Importance Stability")
    print("-" * 55)

    stability_ranking = sorted(
        zip(
            FEATURE_NAMES,
            permutation_result.importances_mean,
            permutation_result.importances_std,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature_name, mean_importance, std_importance in stability_ranking:
        print(
            f"{feature_name:<18}"
            f"{mean_importance:.4f} +/- {std_importance:.4f}"
        )

        output_path = save_importance_chart(
        feature_names=FEATURE_NAMES,
        built_in_values=model.feature_importances_,
        permutation_values=permutation_result.importances_mean,
    )

    print("\nVisualization")
    print("-" * 55)
    print(f"Saved chart: {output_path}")

    print("\nTop Feature Comparison")
    print("-" * 55)
    print(f"Built-in top feature : {built_in_ranking[0][0]}")
    print(f"Permutation top feature: {permutation_ranking[0][0]}")

    print("\nInterpretation")
    print("-" * 55)
    print(
        "Built-in importance measures how much each feature contributed "
        "to tree splits."
    )
    print(
        "Permutation importance measures how much the test F1 score "
        "decreased after shuffling each feature."
    )
    print(
        "A high importance value indicates model dependence, "
        "not a causal relationship."
    )

def save_importance_chart(
    feature_names,
    built_in_values,
    permutation_values,
):
    """Save a chart comparing the two feature-importance methods."""
    sorted_indices = sorted(
        range(len(feature_names)),
        key=lambda index: permutation_values[index],
    )

    sorted_names = [
        feature_names[index]
        for index in sorted_indices
    ]
    sorted_built_in = [
        built_in_values[index]
        for index in sorted_indices
    ]
    sorted_permutation = [
        permutation_values[index]
        for index in sorted_indices
    ]

    y_positions = list(range(len(sorted_names)))
    built_in_positions = [
        position - 0.2
        for position in y_positions
    ]
    permutation_positions = [
        position + 0.2
        for position in y_positions
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    axis.barh(
        built_in_positions,
        sorted_built_in,
        height=0.4,
        label="Random Forest Built-in",
        color="#4C78A8",
    )
    axis.barh(
        permutation_positions,
        sorted_permutation,
        height=0.4,
        label="Permutation Importance",
        color="#F58518",
    )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(sorted_names)
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.set_title(
        "Day 10 - Feature Importance Comparison"
    )
    axis.axvline(
        x=0,
        color="black",
        linewidth=0.8,
    )
    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )
    axis.legend()

    output_directory = (
        Path(__file__).resolve().parents[2] / "outputs"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory / "day10_feature_importance.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path

if __name__ == "__main__":
    main()
