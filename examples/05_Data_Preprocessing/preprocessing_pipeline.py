"""
File: preprocessing_pipeline.py
Project: AI Expert Learning
Day: 5
Topic: Data Preprocessing Pipeline
Author: Alex
Created: 2026-09-01

Description:
    Demonstrates how to combine missing-value imputation, feature
    scaling, and logistic regression into a scikit-learn Pipeline.

    The pipeline learns preprocessing parameters only from the
    training data, which helps prevent data leakage.

Learning Objectives:
    - Combine preprocessing and model training in a Pipeline
    - Replace missing values using median imputation
    - Standardize numerical features
    - Train a logistic regression classifier
    - Evaluate predictions on unseen test data
    - Inspect learned preprocessing parameters

Dependencies:
    - numpy
    - pandas
    - scikit-learn

Usage:
    python examples/05_Data_Preprocessing/preprocessing_pipeline.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = {
        "study_hours": [
            1.0, 1.5, 2.0, 2.5, 3.0,
            np.nan, 3.5, 4.0, 4.5, 5.0,
            5.5, 6.0, np.nan, 7.0, 7.5,
            8.0, 8.5, 9.0, 9.5, 10.0,
        ],
        "attendance": [
            50.0, 55.0, 60.0, 62.0, np.nan,
            68.0, 70.0, 73.0, 75.0, 78.0,
            80.0, np.nan, 85.0, 87.0, 89.0,
            91.0, 93.0, 95.0, 97.0, 99.0,
        ],
        "assignment_score": [
            45.0, 50.0, 55.0, np.nan, 60.0,
            63.0, 65.0, 68.0, 70.0, 73.0,
            np.nan, 78.0, 80.0, 83.0, 85.0,
            88.0, 90.0, 93.0, 96.0, 98.0,
        ],
        "passed": [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ],
    }

    df = pd.DataFrame(data)

    feature_columns = [
        "study_hours",
        "attendance",
        "assignment_score",
    ]

    X = df[feature_columns]
    y = df["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results["actual"] = y_test
    results["predicted"] = predictions
    results["pass_probability"] = probabilities

    print("Original data:")
    print(df)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nPipeline:")
    print(pipeline)

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    print("\nMedians learned from training data:")
    for column, value in zip(
        feature_columns,
        imputer.statistics_,
    ):
        print(f"{column}: {value:.2f}")

    print("\nMeans learned after imputation:")
    for column, value in zip(
        feature_columns,
        scaler.mean_,
    ):
        print(f"{column}: {value:.2f}")

    print("\nTest results:")
    print(results.round(3).sort_index())

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    new_student = pd.DataFrame(
        {
            "study_hours": [np.nan],
            "attendance": [82.0],
            "assignment_score": [78.0],
        }
    )

    new_prediction = pipeline.predict(new_student)[0]
    new_probability = pipeline.predict_proba(
        new_student
    )[0, 1]

    result_label = "Pass" if new_prediction == 1 else "Fail"

    print("\nNew student:")
    print(new_student)

    print(f"\nPrediction: {result_label}")
    print(f"Pass probability: {new_probability:.3f}")


if __name__ == "__main__":
    main()