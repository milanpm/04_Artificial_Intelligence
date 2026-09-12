"""
File: imbalanced_data.py
Description: Compares baseline learning, class weighting, and random
             oversampling for an imbalanced binary classification problem.
Author: Alex
Date: 2026-09-12
"""

from collections import Counter

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def create_dataset():
    """Create an imbalanced binary classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.90, 0.10],
        class_sep=1.0,
        flip_y=0.02,
        random_state=RANDOM_STATE,
    )

    return train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def print_class_distribution(title, labels):
    """Print class counts and ratios."""
    class_counts = Counter(labels)
    total_count = len(labels)

    print(f"\n{title}")
    print("-" * 60)

    for class_label in sorted(class_counts):
        count = class_counts[class_label]
        ratio = count / total_count

        print(
            f"Class {class_label}: "
            f"{count:>4} samples "
            f"({ratio:.2%})"
        )


def create_models():
    """Create models for comparing imbalance-handling methods."""
    baseline_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    class_weight_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    oversampling_model = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "sampler",
                RandomOverSampler(
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Baseline": baseline_model,
        "Class Weight": class_weight_model,
        "Random Oversampling": oversampling_model,
    }


def evaluate_cross_validation(models, X_train, y_train):
    """Evaluate each model using stratified cross-validation."""
    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    results = {}

    print("\n5-Fold Stratified Cross-Validation")
    print("=" * 76)
    print(
        f"{'Method':<22}"
        f"{'Accuracy':>13}"
        f"{'Precision':>13}"
        f"{'Recall':>13}"
        f"{'F1':>13}"
    )
    print("-" * 76)

    for model_name, model in models.items():
        scores = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=-1,
        )

        mean_scores = {
            metric: np.mean(scores[f"test_{metric}"])
            for metric in scoring
        }
        results[model_name] = mean_scores

        print(
            f"{model_name:<22}"
            f"{mean_scores['accuracy']:>13.3f}"
            f"{mean_scores['precision']:>13.3f}"
            f"{mean_scores['recall']:>13.3f}"
            f"{mean_scores['f1']:>13.3f}"
        )

    return results


def evaluate_test_set(models, X_train, X_test, y_train, y_test):
    """Train each model and evaluate it on the test set."""
    results = {}

    print("\nIndependent Test Set Evaluation")
    print("=" * 76)
    print(
        f"{'Method':<22}"
        f"{'Accuracy':>13}"
        f"{'Precision':>13}"
        f"{'Recall':>13}"
        f"{'F1':>13}"
    )
    print("-" * 76)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
        }
        results[model_name] = metrics

        print(
            f"{model_name:<22}"
            f"{metrics['accuracy']:>13.3f}"
            f"{metrics['precision']:>13.3f}"
            f"{metrics['recall']:>13.3f}"
            f"{metrics['f1']:>13.3f}"
        )

        matrix = confusion_matrix(y_test, predictions)

        print(f"\n{model_name} Confusion Matrix")
        print(matrix)

    return results


def print_best_methods(cross_validation_results, test_results):
    """Print the best method for selected evaluation metrics."""
    best_cv_f1 = max(
        cross_validation_results,
        key=lambda name: cross_validation_results[name]["f1"],
    )
    best_test_recall = max(
        test_results,
        key=lambda name: test_results[name]["recall"],
    )
    best_test_f1 = max(
        test_results,
        key=lambda name: test_results[name]["f1"],
    )

    print("\nResult Summary")
    print("=" * 60)
    print(f"Best cross-validation F1: {best_cv_f1}")
    print(f"Best test recall        : {best_test_recall}")
    print(f"Best test F1            : {best_test_f1}")

    print("\nInterpretation")
    print("-" * 60)
    print(
        "Accuracy can remain high even when a model misses many "
        "minority-class samples."
    )
    print(
        "Class weighting changes the penalty assigned to mistakes "
        "without creating new samples."
    )
    print(
        "Random oversampling balances only the training data by "
        "duplicating minority-class samples."
    )
    print(
        "The preferred method depends on the cost of false positives "
        "and false negatives."
    )


def main():
    """Run the complete Day 11 experiment."""
    X_train, X_test, y_train, y_test = create_dataset()

    print("AI Expert Day 11 - Handling Imbalanced Data")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples    : {len(X_test)}")
    print(f"Feature count   : {X_train.shape[1]}")

    print_class_distribution(
        "Original Training Distribution",
        y_train,
    )
    print_class_distribution(
        "Original Test Distribution",
        y_test,
    )

    sampler = RandomOverSampler(
        random_state=RANDOM_STATE,
    )
    _, resampled_labels = sampler.fit_resample(
        X_train,
        y_train,
    )

    print_class_distribution(
        "Training Distribution After Random Oversampling",
        resampled_labels,
    )

    models = create_models()

    cross_validation_results = evaluate_cross_validation(
        models,
        X_train,
        y_train,
    )

    test_results = evaluate_test_set(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print_best_methods(
        cross_validation_results,
        test_results,
    )


if __name__ == "__main__":
    main()
