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

## AI Learning Blog

The learning notes for each completed day are also published on the
[Zero-Score Club](https://zsclub.blogspot.com/) blog.

| Day | Topic | Blog Post |
| --- | --- | --- |
| 1 | From Rule-Based Programming to Machine Learning | [Read Day 1](https://zsclub.blogspot.com/2026/08/ai-learning-day-1-from-rule-based.html) |
| 2 | Data, Features, Labels, and Training Data | [Read Day 2](https://zsclub.blogspot.com/2026/08/ai-learning-day-2-understanding-data.html) |
| 3 | Training, Testing, and Model Evaluation | [Read Day 3](https://zsclub.blogspot.com/2026/08/ai-learning-day-3-training-testing-model-evaluation.html) |
| 4 | Classification Evaluation Metrics | [Read Day 4](https://zsclub.blogspot.com/2026/08/ai-learning-day-4-understanding-classification-metrics.html) |
| 5 | Data Preprocessing and Feature Scaling | [Read Day 5](https://zsclub.blogspot.com/2026/09/ai-learning-day-5-data-preprocessing-feature-scaling.html) |
| 6 | Comparing Machine Learning Models | [Read Day 6](https://zsclub.blogspot.com/2026/09/ai-learning-day-6-comparing-machine-learning-models.html) |

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

---

## Day 7 — Cross-Validation and Reliable Model Evaluation

### Learning Objectives

- Understand the limitations of a single train/test split
- Keep the final test set separate from model selection
- Apply 5-fold Stratified K-Fold cross-validation
- Preserve class proportions across validation folds
- Evaluate models using Accuracy, Precision, Recall, and F1-score
- Interpret the mean and standard deviation of validation scores
- Select a model using mean cross-validation F1-score
- Evaluate the selected model once on the final holdout test set

### Example

```bash
python examples/07_Cross_Validation/cross_validation.py
```

### Evaluation Workflow

1. Split the dataset into training and final test sets.
2. Keep the final test set untouched during model comparison.
3. Perform 5-fold Stratified Cross-Validation using only the training set.
4. Compare the mean and standard deviation of multiple evaluation metrics.
5. Select the model with the highest mean cross-validation F1-score.
6. Retrain the selected model using all training data.
7. Evaluate it once on the final holdout test set.

### Why Stratified K-Fold?

The dataset contains fewer positive samples than negative samples. Stratified K-Fold preserves approximately the same class distribution in every fold, making the evaluation more reliable for imbalanced classification.

```text
Training samples: 375
Test samples: 125
Positive ratio in training set: 26.40%
Positive ratio in test set:     26.40%
```

### Cross-Validation Result

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.792 ± 0.032 | 0.705 ± 0.104 | 0.394 ± 0.067 | 0.499 ± 0.065 |
| Decision Tree | 0.811 ± 0.031 | 0.750 ± 0.147 | 0.465 ± 0.086 | 0.562 ± 0.066 |
| K-Nearest Neighbors | **0.843 ± 0.030** | **0.820 ± 0.072** | **0.514 ± 0.091** | **0.629 ± 0.087** |

K-Nearest Neighbors achieved the highest mean F1-score and was selected as the final model.

### Final Holdout Test Result

| Metric | Score |
| --- | ---: |
| Accuracy | 0.888 |
| Precision | 0.952 |
| Recall | 0.606 |
| F1-score | 0.741 |

The holdout result is higher than the cross-validation average. This demonstrates why a single test result may provide an optimistic estimate of model performance.

### Mean and Standard Deviation

A cross-validation score such as:

```text
F1-score: 0.629 ± 0.087
```

means that the average F1-score across the five validation folds was `0.629`, with a standard deviation of `0.087`.

A higher mean indicates better average performance, while a smaller standard deviation indicates more stable performance across different data subsets.

### Key Lesson

A single train/test split shows performance on only one particular data division. Cross-validation provides a more reliable estimate by evaluating the model repeatedly on different validation subsets.

The final test set must remain untouched during model comparison. Repeatedly checking the test set while selecting or tuning a model can indirectly overfit the model-selection process to that test set.

### Source Code

- [`cross_validation.py`](examples/07_Cross_Validation/cross_validation.py)

### Next Step

**Day 8 — Hyperparameter Tuning with GridSearchCV**

---

## Day 8 — Hyperparameter Tuning with GridSearchCV

### Learning Objectives

- Understand the difference between model parameters and hyperparameters
- Build a preprocessing and KNN classification Pipeline
- Define a hyperparameter search space
- Use GridSearchCV with Stratified K-Fold cross-validation
- Select hyperparameters using mean cross-validation F1-score
- Compare the highest-ranking parameter combinations
- Identify possible overfitting from training and validation scores
- Evaluate the tuned model once on the untouched final test set

### Example

```bash
python examples/08_Hyperparameter_Tuning/grid_search_knn.py
```

### Baseline Model

Day 7 used the following K-Nearest Neighbors configuration:

```text
n_neighbors=7
weights=uniform
p=2
```

The same model was retained as the baseline so that the effect of hyperparameter tuning could be measured fairly.

### Baseline Cross-Validation Result

| Metric | Mean Score |
| --- | ---: |
| Accuracy | 0.843 ± 0.030 |
| Precision | 0.820 ± 0.072 |
| Recall | 0.514 ± 0.091 |
| F1-score | 0.629 ± 0.087 |

### Hyperparameter Search Space

| Hyperparameter | Candidate Values | Meaning |
| --- | --- | --- |
| `n_neighbors` | 3, 5, 7, 9, 11, 15 | Number of neighboring samples used for prediction |
| `weights` | `uniform`, `distance` | Whether all neighbors vote equally or closer neighbors receive more weight |
| `p` | 1, 2 | Manhattan distance (`p=1`) or Euclidean distance (`p=2`) |

The grid contained 24 parameter combinations:

```text
6 neighbor values × 2 weight methods × 2 distance metrics
= 24 combinations
```

Each combination was evaluated with 5-fold cross-validation:

```text
24 combinations × 5 folds = 120 model fits
```

The final test set was not used during the grid search.

### Best Grid Search Result

```text
n_neighbors=3
weights=uniform
p=2
Best mean CV F1-score: 0.690
```

The tuned model improved the mean cross-validation F1-score by approximately `0.062` compared with the baseline model.

### Top Hyperparameter Combinations

| Rank | Neighbors | Weights | p | Mean CV F1 | Standard Deviation | Mean Train F1 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 3 | uniform | 2 | 0.690 | 0.049 | 0.808 |
| 2 | 3 | distance | 2 | 0.687 | 0.056 | 1.000 |
| 3 | 3 | uniform | 1 | 0.680 | 0.049 | 0.855 |
| 3 | 3 | distance | 1 | 0.680 | 0.049 | 1.000 |
| 5 | 15 | distance | 1 | 0.678 | 0.029 | 1.000 |

The distance-weighted configurations achieved a training F1-score of `1.000`. This large difference between training and validation performance indicates possible overfitting.

### Final Holdout Test Comparison

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Baseline KNN | **0.888** | **0.952** | 0.606 | **0.741** |
| Tuned KNN | 0.848 | 0.769 | 0.606 | 0.678 |

The tuned model improved the mean cross-validation F1-score but achieved a lower F1-score on the final holdout test set.

Both models produced the same recall of `0.606`, but the tuned model had lower precision. This means that it found the same proportion of positive samples while producing more false-positive predictions.

### Why Not Select the Baseline Again?

The final test set must not be used for model selection. Choosing the baseline model again only because it performed better on this test set would indirectly tune the decision to the test data.

The correct conclusion is that GridSearchCV selected the tuned model using the training data, but its performance on one independent holdout set was lower than expected.

Cross-validation estimates average performance across several validation subsets, while the holdout test measures performance on one specific subset. The two results can therefore differ, especially when the dataset is relatively small.

### Key Lesson

GridSearchCV provides a systematic method for testing multiple hyperparameter combinations. Each combination is evaluated under the same cross-validation procedure, making the comparison more reliable than manually testing settings on the final test set.

However, hyperparameter tuning does not guarantee better performance on every independent test set. Training scores, validation scores, and final test scores must be interpreted together.

### Source Code

- [`grid_search_knn.py`](examples/08_Hyperparameter_Tuning/grid_search_knn.py)

### Next Step

**Day 9 — Feature Selection and Model Interpretation**

## Day 9 — Feature Selection and Model Interpretation

Day 9 explores how to identify useful input features and remove features that provide little predictive information.

The example uses `SelectKBest` with the ANOVA F-test to score each feature individually. A pipeline and `GridSearchCV` are then used to determine the best number of features without leaking validation information into the training process.

### Learning Objectives

- Understand why unnecessary features can reduce model performance
- Generate informative and noise features for comparison
- Score classification features with `SelectKBest` and `f_classif`
- Perform feature selection inside a machine learning pipeline
- Determine the optimal feature count using cross-validation
- Interpret feature scores and selected features
- Evaluate the final model on an independent test set

### Run the Example

```bash
python examples/09_Feature_Selection/feature_selection.py
```

### Dataset

The synthetic classification dataset contains 500 samples and eight input features:

```text
5 informative features
3 noise features
```

The data is divided using a stratified train-test split:

| Dataset | Samples | Positive Ratio |
| --- | ---: | ---: |
| Training set | 375 | 26.13% |
| Test set | 125 | 26.40% |

Setting `shuffle=False` in `make_classification` keeps the five informative features before the three noise features. This makes the feature-selection results easier to interpret.

### Initial Feature Scores

`SelectKBest` with `f_classif` evaluates each feature independently using the training data.

| Feature | ANOVA F-score | Initial Selection |
| --- | ---: | --- |
| `informative_1` | 46.896 | Selected |
| `informative_2` | 45.410 | Selected |
| `informative_3` | 2.178 | Removed |
| `informative_4` | 61.024 | Selected |
| `informative_5` | 28.974 | Selected |
| `noise_1` | 3.596 | Selected |
| `noise_2` | 0.004 | Removed |
| `noise_3` | 0.024 | Removed |

The initial selector uses `k=5` only to inspect feature scores.

Although `informative_3` is a genuinely informative feature, its individual F-score is lower than the score of `noise_1`. This demonstrates that a univariate feature-selection method does not measure interactions between multiple features.

### Feature-Selection Pipeline

The final experiment uses the following pipeline:

```text
SelectKBest
    ↓
StandardScaler
    ↓
K-Nearest Neighbors
```

Feature selection is placed inside the pipeline so that each cross-validation fold learns its selection rules from only its own training subset.

This prevents information from the validation fold from leaking into feature selection.

The KNN model uses the best hyperparameters found during Day 8:

```text
n_neighbors=3
weights=uniform
p=2
```

### Feature Count Search

The number of selected features was evaluated from 1 through 8 using stratified 5-fold cross-validation.

```text
8 feature-count candidates × 5 folds
= 40 model fits
```

The final test set was not used to select the feature count.

### Cross-Validation Results

| Selected Features | Mean CV F1 | Standard Deviation |
| ---: | ---: | ---: |
| 1 | 0.516 | 0.099 |
| 2 | 0.615 | 0.069 |
| 3 | 0.796 | 0.056 |
| 4 | 0.837 | 0.030 |
| 5 | 0.781 | 0.047 |
| 6 | **0.871** | 0.038 |
| 7 | 0.841 | 0.012 |
| 8 | 0.806 | 0.048 |

The best result was obtained with six selected features:

```text
Best number of features: 6
Best mean CV F1-score: 0.871 ± 0.038
```

Compared with using all eight features, selecting six features improved the mean cross-validation F1-score from `0.806` to `0.871`.

### Selected Features

The best pipeline selected the following features:

```text
informative_1
informative_2
informative_3
informative_4
informative_5
noise_1
```

All five informative features were included. The selector also included `noise_1` because it showed a stronger individual statistical relationship with the target than the other noise features.

This does not mean that `noise_1` is genuinely important. Random features can sometimes show an accidental relationship with the target, especially when the dataset is limited.

### Final Holdout Test Result

| Metric | Score |
| --- | ---: |
| Accuracy | 0.912 |
| Precision | 0.824 |
| Recall | 0.848 |
| F1-score | 0.836 |

The final test F1-score was `0.035` lower than the best mean cross-validation score:

```text
Best mean CV F1: 0.871
Final test F1:    0.836
```

This relatively small difference indicates that the selected pipeline generalized reasonably well to the independent test set.

### Key Lesson

Feature selection is part of model training and must be performed inside the cross-validation pipeline.

A high individual feature score does not guarantee that a feature is genuinely important, and a low individual score does not guarantee that the feature is useless when combined with other features.

The number of selected features should therefore be determined using cross-validation rather than chosen only from individual feature scores or final test performance.

### Source Code

- [`feature_selection.py`](examples/09_Feature_Selection/feature_selection.py)

### Next Step

**Day 10 — Feature Importance and Model Explainability**

## Progress

- [x] Day 1 - AI Fundamentals and First Machine Learning Model
- [x] Day 2 - Data, Features, Labels, and Training Data
- [x] Day 3 - Training, Testing, and Model Evaluation
- [x] Day 4 - Classification Evaluation Metrics
- [x] Day 5 - Data Preprocessing and Feature Scaling
- [x] Day 6 - Comparing Machine Learning Models
- [x] Day 7 - Cross-Validation and Reliable Model Evaluation
- [x] Day 8 - Hyperparameter Tuning with GridSearchCV
- [ ] Day 9 - Feature Selection and Model Interpretation

---

## Author

Alex
