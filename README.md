# Artificial Intelligence Learning

A hands-on learning repository for studying Artificial Intelligence from beginner to expert level.

The goal of this repository is to understand AI concepts through practical Python examples and gradually progress from basic machine learning to deep learning and real-world AI applications.

---

## Learning Goals

- Understand the fundamentals of Artificial Intelligence
- Learn Machine Learning concepts and algorithms
- Understand data, features, labels, training, and prediction
- Learn Deep Learning and Neural Networks
- Study Computer Vision and AI Vision
- Build practical AI projects
- Progress from AI Beginner to AI Expert

---

## Environment

- Python
- scikit-learn

---

## Repository Structure

```text
04_Artificial_Intelligence/
├── examples/
│   └── 01_Basics/
│       ├── rule_based_ai.py
│       └── first_ml.py
├── .gitignore
└── README.md
```

---

# Day 1 - Introduction to Artificial Intelligence

## What I Learned

Day 1 focused on understanding the difference between traditional rule-based programming and Machine Learning.

### 1. Rule-Based Programming

In traditional programming, a programmer explicitly defines the decision rules.

Example:

```python
if study_hours >= 4 and attendance >= 70:
    result = "PASS"
else:
    result = "FAIL"
```

The programmer determines the conditions used to make the decision.

### 2. Machine Learning

In Machine Learning, the decision rule can be learned from training data.

```python
model.fit(X, y)
```

The trained model can then make predictions for new data.

```python
prediction = model.predict(new_student)
```

### 3. Feature and Label

Training data was represented using `X` and `y`.

```text
X = Features
y = Labels
```

In the Day 1 example:

```text
Features
├── study_hours
└── attendance

Label
└── PASS / FAIL
```

### 4. Decision Tree

A `DecisionTreeClassifier` was used as the first Machine Learning model.

Initially, the model learned a rule based on study hours:

```text
study_hours <= 3.50 → FAIL
study_hours > 3.50  → PASS
```

After adding another training example:

```text
[2, 95] → PASS
```

the learned rule changed to:

```text
attendance <= 70.00 → FAIL
attendance > 70.00  → PASS
```

This demonstrates an important Machine Learning concept:

> When training data changes, the model and its learned decision rules can also change.

---

## Day 1 Examples

### Rule-Based Prediction

```bash
python examples/01_Basics/rule_based_ai.py
```

### First Machine Learning Model

```bash
python examples/01_Basics/first_ml.py
```

Example output:

```text
|--- attendance <= 70.00
|   |--- class: 0
|--- attendance >  70.00
|   |--- class: 1

Prediction: PASS
```

---

## Key Concepts

| Concept          | Description                         |
| ---------------- | ----------------------------------- |
| AI               | Artificial Intelligence             |
| Machine Learning | Learning patterns from data         |
| Feature          | Input information used by a model   |
| Label            | Correct output or target value      |
| Training         | Process of learning from data       |
| Model            | Learned decision structure          |
| Prediction       | Model output for new data           |
| Decision Tree    | Tree-based classification algorithm |

---

## Day 1 Summary

The most important lesson from Day 1 is:

> Rule-based programming uses rules written by humans, while Machine Learning learns decision patterns from data.

---

## Day 2 - Data, Features, Labels, and Training Data

Day 2 focused on understanding how training data is structured and how its quality affects the rules learned by a Machine Learning model.

### Dataset Structure

The training dataset contains two features:

- `study_hours`: Number of hours spent studying
- `attendance_rate`: Student attendance percentage

The target label represents the expected result:

- `FAIL`: The student does not satisfy the required conditions
- `PASS`: The student satisfies both conditions

Each training sample follows this structure:

```text
[study_hours, attendance_rate] → label
```

---

## Day 3 - Training, Testing, and Model Evaluation

Day 3 focused on splitting a dataset into training and test sets, training a decision tree with only the training data, and evaluating its predictions on unseen test data.

### Dataset Split

The dataset contains 20 samples and two features:

- `study_hours`: Number of hours spent studying
- `attendance_rate`: Student attendance percentage

The dataset was divided using `train_test_split()`:

```text
Total samples: 20
Training samples: 15
Test samples: 5
```

The model learned only from `X_train` and `y_train`:

```python
model.fit(X_train, y_train)
```

---

## Day 4 - Classification Evaluation Metrics

Day 4 focused on evaluating binary classification results using a confusion matrix and four important evaluation metrics: accuracy, precision, recall, and F1-score.

### Positive and Negative Classes

For this example:

- `PASS` is treated as the Positive class.
- `FAIL` is treated as the Negative class.

### Confusion Matrix

The confusion matrix compares the actual labels with the predicted labels:

```text
                 Predicted
                 FAIL  PASS
Actual FAIL         3     1
Actual PASS         1     3
```

The four classification results were:

```text
TP - True Positive:  3
TN - True Negative:  3
FP - False Positive: 1
FN - False Negative: 1
```

- `TP`: The model correctly predicted PASS.
- `TN`: The model correctly predicted FAIL.
- `FP`: The model incorrectly predicted PASS.
- `FN`: The model incorrectly predicted FAIL.

### Evaluation Metrics

The classification evaluation results were:

```text
Accuracy:  75.0%
Precision: 75.0%
Recall:    75.0%
F1-score:  75.0%
```

- **Accuracy** measures the percentage of all correct predictions.
- **Precision** measures how reliable the Positive predictions are.
- **Recall** measures how many actual Positive samples were found.
- **F1-score** measures the balance between precision and recall.

### Metric Formulas

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 × (Precision × Recall) / (Precision + Recall)
```

### Classification Report

Scikit-learn's `classification_report()` displays precision, recall, F1-score, and support for each class.

The `support` value represents the number of actual samples belonging to each class.

### Source Code

- [`classification_metrics.py`](examples/04_Classification_Metrics/classification_metrics.py)

---

## Day 5 - Data Preprocessing and Feature Scaling

Day 5 focused on preparing raw data for machine learning by handling missing values, scaling numerical features, preventing data leakage, and building a preprocessing pipeline.

### Why Data Preprocessing Matters

Real-world datasets may contain:

- Missing values
- Features with different numerical ranges
- Incorrect or inconsistent values
- Categorical values
- Outliers

Data preprocessing transforms raw data into a form that machine learning models can use effectively.

### Handling Missing Values

Missing numerical values were represented using NumPy's `NaN` value:

```python
np.nan
```

Missing values in each column were identified using:

```python
df.isnull().sum()
```

`SimpleImputer` replaced missing numerical values with the median of each feature:

```python
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
```

The learned median values were:

```text
study_hours:       4.00
attendance:       80.00
assignment_score: 70.00
```

Median imputation is useful because it is less affected by unusually large or small values than mean imputation.

### StandardScaler

`StandardScaler` transforms each feature so that the training data has a mean of approximately `0` and a standard deviation of approximately `1`.

```text
z = (x - mean) / standard deviation
```

The scaler was fitted only on the training data:

```python
standard_scaler = StandardScaler()

X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)
```

The scaled training data had the following properties:

```text
Feature means:               0
Feature standard deviations: 1
```

### MinMaxScaler

`MinMaxScaler` normally transforms training values into the range from `0` to `1`.

```text
scaled value = (x - minimum) / (maximum - minimum)
```

It was applied as follows:

```python
minmax_scaler = MinMaxScaler()

X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)
```

A test value may be greater than `1.0` when it exceeds the maximum learned from the training data.

For example:

```text
Training range for study_hours: 1.0 to 8.0
Test value:                     9.0
Scaled test value:              1.143
```

### Preventing Data Leakage

The dataset must be split before fitting preprocessing tools:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)
```

The correct preprocessing process is:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Calling `fit_transform()` on the test data would calculate new preprocessing parameters using information that should remain unseen.

> Fit preprocessing tools only on the training data and use the learned parameters to transform the test data.

### Preprocessing Pipeline

A scikit-learn `Pipeline` combined missing-value handling, feature scaling, and model training:

```python
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
```

Training the pipeline performs:

```text
Training data
→ Median imputation
→ Standard scaling
→ Logistic regression training
```

Making predictions performs:

```text
Test data
→ Apply learned medians
→ Apply learned scaling parameters
→ Predict the class
```

The pipeline was trained and evaluated using:

```python
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Pipeline Results

The preprocessing steps learned the following values only from the training data:

```text
Median values:
study_hours:       5.25
attendance:       79.00
assignment_score: 73.00

Means after imputation:
study_hours:       5.50
attendance:       77.93
assignment_score: 73.71
```

All six test samples were classified correctly:

```text
Accuracy: 1.00
```

This small artificial dataset demonstrates that the pipeline works correctly. It does not prove that the model will achieve perfect performance on real-world data.

### Predicting a New Student

The pipeline received a new student with a missing `study_hours` value:

```text
study_hours:       NaN
attendance:        82.0
assignment_score:  78.0
```

The missing value was replaced with the training median of `5.25`.

```text
Prediction:       Pass
Pass probability: 0.717
```

### Source Code

- [`missing_values.py`](examples/05_Data_Preprocessing/missing_values.py)
- [`feature_scaling.py`](examples/05_Data_Preprocessing/feature_scaling.py)
- [`preprocessing_pipeline.py`](examples/05_Data_Preprocessing/preprocessing_pipeline.py)

---


## Day 6 — Comparing Machine Learning Models

### Learning Objectives

- Compare Logistic Regression, Decision Tree, and K-Nearest Neighbors
- Train and evaluate every model on the same data split
- Compare Accuracy, Precision, Recall, F1-score, and Confusion Matrix
- Understand why accuracy alone may be misleading
- Distinguish models that require feature scaling from models that generally do not
- Prevent data leakage by applying preprocessing inside a Pipeline

### Example

```bash
python examples/06_Model_Comparison/model_comparison.py
```

### Model and Preprocessing Comparison

| Model | Feature Scaling | Main Characteristic |
| --- | --- | --- |
| Logistic Regression | Recommended | Linear, probabilistic, interpretable baseline |
| Decision Tree | Usually unnecessary | Nonlinear rule-based splits |
| K-Nearest Neighbors | Usually required | Distance-based prediction |

### Result

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.760 | 0.579 | 0.333 | 0.423 |
| Decision Tree | 0.832 | 0.833 | 0.455 | 0.588 |
| K-Nearest Neighbors | 0.888 | 0.952 | 0.606 | 0.741 |

A majority-class baseline achieved 0.736 accuracy but 0.000 recall and 0.000 F1-score. This demonstrates why accuracy alone is not sufficient for imbalanced classification.

### Key Lesson

The best model is not necessarily the model with the highest accuracy. Model selection must consider the problem objective, error costs, preprocessing requirements, and multiple evaluation metrics.

### Next Step

**Day 7 — Cross-Validation and Reliable Model Evaluation**


## Progress

- [x] Day 1 - AI Fundamentals and First Machine Learning Model
- [x] Day 2 - Data, Features, Labels, and Training Data
- [x] Day 3 - Training, Testing, and Model Evaluation
- [x] Day 4 - Classification Evaluation Metrics
- [x] Day 5 - Data Preprocessing and Feature Scaling
- [x] Day 6 - Comparing Machine Learning Models
- [ ] Day 7 - Cross-Validation and Reliable Model Evaluation

---

## Author

Alex
