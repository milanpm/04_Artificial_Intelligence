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

## Progress

- [x] Day 1 - AI Fundamentals and First Machine Learning Model
- [ ] Day 2 - Data, Features, Labels, and Training Data

---

## Author

Alex