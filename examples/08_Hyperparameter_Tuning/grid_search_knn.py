"""
AI Expert Day 8: Hyperparameter Tuning with GridSearchCV

This example demonstrates how to tune a K-Nearest Neighbors classifier
systematically using GridSearchCV and Stratified K-Fold cross-validation.

Learning objectives:
    1. Understand the difference between parameters and hyperparameters.
    2. Create a preprocessing and classification Pipeline.
    3. Define a hyperparameter search space for KNN.
    4. Use GridSearchCV with Stratified K-Fold cross-validation.
    5. Select hyperparameters using mean cross-validation F1-score.
    6. Inspect and compare the highest-ranking parameter combinations.
    7. Compare the baseline and tuned models on the final test set.
    8. Keep the final test set separate from hyperparameter tuning.

Search space:
    - n_neighbors: 3, 5, 7, 9, 11, 15
    - weights: uniform, distance
    - p: 1, 2

The search evaluates 24 parameter combinations using 5-fold
cross-validation, resulting in 120 model fits.

Dataset:
    Synthetic binary classification dataset created with
    sklearn.datasets.make_classification.

Author:
    Alex

Repository:
    https://github.com/milanpm/04_Artificial_Intelligence
"""

from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_knn_pipeline(n_neighbors=7):
    """Create a scaling and KNN classification pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(n_neighbors=n_neighbors),
            ),
        ]
    )


def evaluate_predictions(y_true, y_prediction):
    """Calculate classification metrics for predicted labels."""
    return {
        "accuracy": accuracy_score(y_true, y_prediction),
        "precision": precision_score(
            y_true,
            y_prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_prediction,
            zero_division=0,
        ),
    }


def print_test_metrics(model_name, metrics):
    """Print final holdout test metrics."""
    print(f"\n{model_name}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")


def print_top_results(grid_search, top_n=5):
    """Print the highest-ranking GridSearchCV parameter combinations."""
    results = grid_search.cv_results_

    ranked_indices = sorted(
        range(len(results["params"])),
        key=lambda index: results["rank_test_score"][index],
    )

    print(f"\nTop {top_n} Hyperparameter Combinations")
    print(
        f"{'Rank':<6}"
        f"{'Neighbors':<12}"
        f"{'Weights':<12}"
        f"{'p':<5}"
        f"{'Mean F1':<12}"
        f"{'Std F1':<12}"
        f"{'Train F1':<12}"
    )
    print("-" * 71)

    for index in ranked_indices[:top_n]:
        parameters = results["params"][index]

        print(
            f"{results['rank_test_score'][index]:<6}"
            f"{parameters['knn__n_neighbors']:<12}"
            f"{parameters['knn__weights']:<12}"
            f"{parameters['knn__p']:<5}"
            f"{results['mean_test_score'][index]:<12.3f}"
            f"{results['std_test_score'][index]:<12.3f}"
            f"{results['mean_train_score'][index]:<12.3f}"
        )


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

    # Keep the test set untouched during hyperparameter tuning.
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

    print("Dataset")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Positive ratio in training set: {y_train.mean():.2%}")
    print(f"Positive ratio in test set:     {y_test.mean():.2%}")

    # Day 7 used seven neighbors, so it remains the baseline.
    baseline_model = create_knn_pipeline(n_neighbors=7)

    baseline_scores = cross_validate(
        baseline_model,
        X_train,
        y_train,
        cv=cross_validator,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
    )

    print("\nBaseline KNN: 5-Fold Cross-Validation")
    print("Parameters: n_neighbors=7, weights=uniform, p=2")
    print(
        "Accuracy:  "
        f"{baseline_scores['test_accuracy'].mean():.3f} "
        f"± {baseline_scores['test_accuracy'].std():.3f}"
    )
    print(
        "Precision: "
        f"{baseline_scores['test_precision'].mean():.3f} "
        f"± {baseline_scores['test_precision'].std():.3f}"
    )
    print(
        "Recall:    "
        f"{baseline_scores['test_recall'].mean():.3f} "
        f"± {baseline_scores['test_recall'].std():.3f}"
    )
    print(
        "F1:        "
        f"{baseline_scores['test_f1'].mean():.3f} "
        f"± {baseline_scores['test_f1'].std():.3f}"
    )

    parameter_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    total_combinations = (
        len(parameter_grid["knn__n_neighbors"])
        * len(parameter_grid["knn__weights"])
        * len(parameter_grid["knn__p"])
    )
    total_fits = total_combinations * cross_validator.get_n_splits()

    print("\nGrid Search")
    print(f"Parameter combinations: {total_combinations}")
    print(f"Cross-validation folds: {cross_validator.get_n_splits()}")
    print(f"Total model fits:       {total_fits}")
    print("Selection metric:       Mean cross-validation F1-score")
    print("The final test set is not used during grid search.")

    grid_search = GridSearchCV(
        estimator=create_knn_pipeline(),
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validator,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Grid Search Result")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best mean CV F1: {grid_search.best_score_:.3f}")

    print_top_results(grid_search)

    # Evaluate both models once using the untouched final test set.
    baseline_model.fit(X_train, y_train)
    baseline_prediction = baseline_model.predict(X_test)
    baseline_test_metrics = evaluate_predictions(
        y_test,
        baseline_prediction,
    )

    tuned_model = grid_search.best_estimator_
    tuned_prediction = tuned_model.predict(X_test)
    tuned_test_metrics = evaluate_predictions(
        y_test,
        tuned_prediction,
    )

    print("\nFinal Holdout Test Comparison")
    print("The test set is evaluated only after tuning is complete.")

    print_test_metrics(
        "Baseline KNN",
        baseline_test_metrics,
    )
    print_test_metrics(
        "Tuned KNN",
        tuned_test_metrics,
    )

    f1_difference = (
        tuned_test_metrics["f1"]
        - baseline_test_metrics["f1"]
    )

    print("\nInterpretation")
    print(
        "CV F1 improvement: "
        f"{grid_search.best_score_ - baseline_scores['test_f1'].mean():+.3f}"
    )
    print(f"Test F1 difference: {f1_difference:+.3f}")

    if f1_difference > 0:
        print(
            "The tuned model also improved F1-score "
            "on the final test set."
        )
    elif f1_difference < 0:
        print(
            "The tuned model improved mean cross-validation F1-score, "
            "but not the final test F1-score."
        )
        print(
            "A single holdout result can differ from the average "
            "cross-validation result."
        )
    else:
        print(
            "The baseline and tuned models produced the same "
            "final test F1-score."
        )


if __name__ == "__main__":
    main()
