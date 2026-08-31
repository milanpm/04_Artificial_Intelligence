"""
File Name: classification_metrics.py

Created: 2026.08.31

Author: Alex

Description:
    AI Learning Day 4: Classification Model Evaluation Metrics.

    Demonstrates how to evaluate binary classification predictions
    using a confusion matrix, accuracy, precision, recall, F1-score,
    and a classification report.
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# PASS is treated as Positive.
# FAIL is treated as Negative.
y_actual = [
    "PASS",
    "PASS",
    "PASS",
    "PASS",
    "FAIL",
    "FAIL",
    "FAIL",
    "FAIL",
]

y_predicted = [
    "PASS",
    "PASS",
    "PASS",
    "FAIL",
    "PASS",
    "FAIL",
    "FAIL",
    "FAIL",
]


print("=== Classification Results ===")

for index, (actual, predicted) in enumerate(
    zip(y_actual, y_predicted),
    start=1,
):
    result = "CORRECT" if actual == predicted else "WRONG"

    print(
        f"Sample {index}: "
        f"Actual={actual}, "
        f"Predicted={predicted}, "
        f"Result={result}"
    )


# The label order produces this matrix:
#
#                    Predicted
#                    FAIL  PASS
# Actual FAIL          TN    FP
# Actual PASS          FN    TP
matrix = confusion_matrix(
    y_actual,
    y_predicted,
    labels=["FAIL", "PASS"],
)

tn, fp, fn, tp = matrix.ravel()

print("\n=== Confusion Matrix ===")
print("                 Predicted")
print("                 FAIL  PASS")
print(f"Actual FAIL      {tn:4d}  {fp:4d}")
print(f"Actual PASS      {fn:4d}  {tp:4d}")

print("\n=== Confusion Matrix Values ===")
print(f"TP - True Positive:  {tp}")
print(f"TN - True Negative:  {tn}")
print(f"FP - False Positive: {fp}")
print(f"FN - False Negative: {fn}")


# PASS is explicitly selected as the Positive label.
accuracy = accuracy_score(y_actual, y_predicted)
precision = precision_score(
    y_actual,
    y_predicted,
    pos_label="PASS",
)
recall = recall_score(
    y_actual,
    y_predicted,
    pos_label="PASS",
)
f1 = f1_score(
    y_actual,
    y_predicted,
    pos_label="PASS",
)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy:  {accuracy:.2f} ({accuracy * 100:.1f}%)")
print(f"Precision: {precision:.2f} ({precision * 100:.1f}%)")
print(f"Recall:    {recall:.2f} ({recall * 100:.1f}%)")
print(f"F1-score:  {f1:.2f} ({f1 * 100:.1f}%)")

print("\n=== Classification Report ===")
print(
    classification_report(
        y_actual,
        y_predicted,
        labels=["FAIL", "PASS"],
        zero_division=0,
    )
)
