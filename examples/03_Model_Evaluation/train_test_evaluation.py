"""
File Name: train_test_evaluation.py

Created: 2026.08.30

Author: Alex

Description:
    AI Learning Day 3: Training, Testing, and Model Evaluation.

    Demonstrates how to split a dataset into training and test sets,
    train a decision tree classifier, make predictions on unseen data,
    and evaluate the model using accuracy.
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


# Each sample follows this structure:
# [study_hours, attendance_rate]
X = [
    [1, 50],
    [1, 85],
    [2, 60],
    [2, 90],
    [3, 65],
    [3, 75],
    [3, 95],
    [4, 60],
    [4, 70],
    [5, 75],
    [4, 80],
    [4, 90],
    [5, 80],
    [5, 85],
    [5, 95],
    [6, 80],
    [6, 85],
    [6, 95],
    [7, 90],
    [8, 95],
]

# FAIL and PASS are the labels for the samples above.
y = [
    "FAIL",
    "FAIL",
    "FAIL",
    "FAIL",
    "FAIL",
    "FAIL",
    "FAIL",
    "FAIL",
    "FAIL",
    "FAIL",
    "PASS",
    "PASS",
    "PASS",
    "PASS",
    "PASS",
    "PASS",
    "PASS",
    "PASS",
    "PASS",
    "PASS",
]


# Split the complete dataset into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("=== Dataset Split ===")
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create and train the model using only the training data.
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)

print("\n=== Learned Decision Tree ===")
tree_rules = export_text(
    model,
    feature_names=["study_hours", "attendance_rate"],
)
print(tree_rules)

# Predict the labels of data that the model did not see during training.
y_pred = model.predict(X_test)

print("=== Test Predictions ===")

for features, actual, predicted in zip(X_test, y_test, y_pred):
    study_hours, attendance_rate = features
    result = "CORRECT" if actual == predicted else "WRONG"

    print(
        f"Study Hours: {study_hours}, "
        f"Attendance: {attendance_rate}%, "
        f"Actual: {actual}, "
        f"Predicted: {predicted}, "
        f"Result: {result}"
    )

# Compare the predictions with the correct test labels.
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy:.2f}")
print(f"Accuracy: {accuracy * 100:.1f}%")
