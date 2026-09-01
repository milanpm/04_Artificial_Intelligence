"""
File: missing_values.py
Project: AI Expert Learning
Day: 5
Topic: Data Preprocessing and Feature Scaling
Author: Alex
Created: 2026-09-01

Description:
    Demonstrates how to detect and handle missing numerical values
    using pandas and scikit-learn's SimpleImputer.

    Missing values in study hours, attendance, and assignment scores
    are replaced with the median value of each feature.

Learning Objectives:
    - Represent missing values using NumPy NaN
    - Detect missing values with pandas
    - Calculate median values for numerical features
    - Replace missing values using SimpleImputer
    - Verify that no missing values remain

Dependencies:
    - numpy
    - pandas
    - scikit-learn

Usage:
    python examples/05_Data_Preprocessing/missing_values.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = {
    "study_hours": [2.0, 5.0, np.nan, 4.0, 6.0, 3.0],
    "attendance": [65.0, 90.0, 80.0, np.nan, 95.0, 70.0],
    "assignment_score": [60.0, 85.0, 75.0, 70.0, np.nan, 68.0],
    "passed": [0, 1, 1, 1, 1, 0],
}

df = pd.DataFrame(data)

print("Original data:")
print(df)

print("\nMissing values by column:")
print(df.isnull().sum())

feature_columns = [
    "study_hours",
    "attendance",
    "assignment_score",
]

X = df[feature_columns]

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

imputed_df = pd.DataFrame(
    X_imputed,
    columns=feature_columns,
)

imputed_df["passed"] = df["passed"]

print("\nMedian values learned by the imputer:")
for column, value in zip(feature_columns, imputer.statistics_):
    print(f"{column}: {value:.2f}")

print("\nData after median imputation:")
print(imputed_df)

print("\nRemaining missing values:")
print(imputed_df.isnull().sum())
