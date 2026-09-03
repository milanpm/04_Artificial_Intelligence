"""AI Expert Day 6: Compare classification models fairly."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=4, random_state=42
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
    }

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Positive ratio in test set: {y_test.mean():.2%}\n")

    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, zero_division=0)
        recall = recall_score(y_test, prediction, zero_division=0)
        f1 = f1_score(y_test, prediction, zero_division=0)

        print(f"{name:<22} {accuracy:>9.3f} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f}")
        print(f"  Confusion matrix: {confusion_matrix(y_test, prediction).tolist()}")

    majority_prediction = [0] * len(y_test)
    print("\nAccuracy trap")
    print(f"Majority-class accuracy: {accuracy_score(y_test, majority_prediction):.3f}")
    print(f"Majority-class recall:   {recall_score(y_test, majority_prediction):.3f}")
    print(f"Majority-class F1:       {f1_score(y_test, majority_prediction):.3f}")


if __name__ == "__main__":
    main()
