"""
AI Expert Day 7: Cross-Validation and Reliable Model Evaluation

This example demonstrates how to evaluate classification models more
reliably using Stratified K-Fold cross-validation.

Learning objectives:
    1. Understand the limitations of a single train/test split.
    2. Keep the final test set separate from model selection.
    3. Apply 5-fold Stratified K-Fold cross-validation.
    4. Compare accuracy, precision, recall, and F1-score.
    5. Interpret the mean and standard deviation of validation scores.
    6. Select the best model using mean cross-validation F1-score.
    7. Evaluate the selected model once on the final holdout test set.

Models:
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors

Evaluation workflow:
    1. Split the dataset into training and final test sets.
    2. Perform cross-validation using only the training set.
    3. Select the model with the highest mean F1-score.
    4. Retrain the selected model using the complete training set.
    5. Evaluate it once using the untouched test set.

Dataset:
    Synthetic binary classification dataset created with
    sklearn.datasets.make_classification.

Author:
    Alex K.

Repository:
    https://github.com/milanpm/04_Artificial_Intelligence
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def create_models():
    """Create the classification models used for comparison."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(n_neighbors=7),
                ),
            ]
        ),
    }


def main():
    X, y = make_classification(
        n_samples=500,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        weights=[0.75, 0.25],
        class_sep=1.0,
        flip_y=0.03,
        random_state=42,
    )

    # Keep the test set separate until model comparison is complete.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    models = create_models()

    print("Dataset")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Positive ratio in training set: {y_train.mean():.2%}")
    print(f"Positive ratio in test set:     {y_test.mean():.2%}")
    print("\n5-Fold Stratified Cross-Validation")
    print("The final test set is not used during cross-validation.\n")

    header = (
        f"{'Model':<22}"
        f"{'Accuracy':>16}"
        f"{'Precision':>16}"
        f"{'Recall':>16}"
        f"{'F1':>16}"
    )
    print(header)
    print("-" * len(header))

    best_model_name = None
    best_mean_f1 = -1.0

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
        )

        accuracy_mean = scores["test_accuracy"].mean()
        accuracy_std = scores["test_accuracy"].std()
        precision_mean = scores["test_precision"].mean()
        precision_std = scores["test_precision"].std()
        recall_mean = scores["test_recall"].mean()
        recall_std = scores["test_recall"].std()
        f1_mean = scores["test_f1"].mean()
        f1_std = scores["test_f1"].std()

        print(
            f"{name:<22}"
            f"{accuracy_mean:>7.3f} ± {accuracy_std:<6.3f}"
            f"{precision_mean:>7.3f} ± {precision_std:<6.3f}"
            f"{recall_mean:>7.3f} ± {recall_std:<6.3f}"
            f"{f1_mean:>7.3f} ± {f1_std:<6.3f}"
        )

        if f1_mean > best_mean_f1:
            best_mean_f1 = f1_mean
            best_model_name = name

    print(f"\nBest model by mean CV F1: {best_model_name}")
    print(f"Mean CV F1: {best_mean_f1:.3f}")

    # Train the selected model on all training data.
    final_model = models[best_model_name]
    final_model.fit(X_train, y_train)
    final_prediction = final_model.predict(X_test)

    print("\nFinal Holdout Test")
    print(f"Model:     {best_model_name}")
    print(f"Accuracy:  {accuracy_score(y_test, final_prediction):.3f}")
    print(
        "Precision: "
        f"{precision_score(y_test, final_prediction, zero_division=0):.3f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_test, final_prediction, zero_division=0):.3f}"
    )
    print(
        f"F1:        "
        f"{f1_score(y_test, final_prediction, zero_division=0):.3f}"
    )


if __name__ == "__main__":
    main()
