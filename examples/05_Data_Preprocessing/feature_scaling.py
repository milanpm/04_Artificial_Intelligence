"""
File: feature_scaling.py
Project: AI Expert Learning
Day: 5
Topic: Data Preprocessing and Feature Scaling
Author: Alex
Created: 2026-09-01

Description:
    Demonstrates how to scale numerical features using StandardScaler
    and MinMaxScaler after splitting data into training and test sets.

    The scalers are fitted only on the training data and then applied
    to the test data to prevent data leakage.

Learning Objectives:
    - Split data into training and test sets
    - Understand fit(), transform(), and fit_transform()
    - Standardize features using StandardScaler
    - Normalize features using MinMaxScaler
    - Prevent data leakage during preprocessing
    - Compare the results of different scaling methods

Dependencies:
    - numpy
    - pandas
    - scikit-learn

Usage:
    python examples/05_Data_Preprocessing/feature_scaling.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


data = {
    "study_hours": [
        1.0, 2.0, 2.5, 3.0, 3.5,
        4.0, 5.0, 5.5, 6.0, 7.0,
        8.0, 9.0,
    ],
    "attendance": [
        50.0, 60.0, 65.0, 68.0, 72.0,
        75.0, 80.0, 85.0, 88.0, 90.0,
        95.0, 98.0,
    ],
    "assignment_score": [
        45.0, 55.0, 60.0, 62.0, 68.0,
        70.0, 75.0, 80.0, 85.0, 88.0,
        92.0, 97.0,
    ],
    "passed": [
        0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1,
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
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("Original training data:")
print(X_train)

print("\nOriginal test data:")
print(X_test)

standard_scaler = StandardScaler()

X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)

train_standard_df = pd.DataFrame(
    X_train_standard,
    columns=feature_columns,
    index=X_train.index,
)

test_standard_df = pd.DataFrame(
    X_test_standard,
    columns=feature_columns,
    index=X_test.index,
)

print("\nStandardScaler mean learned from training data:")
for column, value in zip(
    feature_columns,
    standard_scaler.mean_,
):
    print(f"{column}: {value:.2f}")

print("\nTraining data after StandardScaler:")
print(train_standard_df.round(3))

print("\nTest data after StandardScaler:")
print(test_standard_df.round(3))

print("\nScaled training means:")
print(train_standard_df.mean().round(6))

print("\nScaled training standard deviations:")
print(
    np.std(
        X_train_standard,
        axis=0,
        ddof=0,
    ).round(6)
)

minmax_scaler = MinMaxScaler()

X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

train_minmax_df = pd.DataFrame(
    X_train_minmax,
    columns=feature_columns,
    index=X_train.index,
)

test_minmax_df = pd.DataFrame(
    X_test_minmax,
    columns=feature_columns,
    index=X_test.index,
)

print("\nTraining data after MinMaxScaler:")
print(train_minmax_df.round(3))

print("\nTest data after MinMaxScaler:")
print(test_minmax_df.round(3))

print("\nMinMaxScaler training ranges:")
print("Minimum:")
print(train_minmax_df.min().round(3))

print("\nMaximum:")
print(train_minmax_df.max().round(3))

print("\nTraining target:")
print(y_train.sort_index())

print("\nTest target:")
print(y_test.sort_index())
