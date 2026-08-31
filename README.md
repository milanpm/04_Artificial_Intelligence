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

## Progress

- [x] Day 1 - AI Fundamentals and First Machine Learning Model
- [x] Day 2 - Data, Features, Labels, and Training Data
- [x] Day 3 - Training, Testing, and Model Evaluation
- [x] Day 4 - Classification Evaluation Metrics

---

## Author

Alex
