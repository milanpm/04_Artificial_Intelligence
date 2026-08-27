"""
File Name: first_ml.py
Created: 2026.08.27
Author: Alex
Description: Train a simple Decision Tree classifier and predict PASS/FAIL.
"""

from sklearn.tree import DecisionTreeClassifier, export_text

# Training data
# X = [study_hours, attendance]
X = [
    [1, 50],
    [2, 60],
    [3, 65],
    [4, 75],
    [5, 80],
    [6, 90],
    [8, 95],
    [2, 95],
]

# y = result
# 0 = FAIL
# 1 = PASS
y = [
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
]


# Create the model
model = DecisionTreeClassifier(random_state=42)

# 1. Train
model.fit(X, y)

# 2. Display learned rules
tree_rules = export_text(
    model,
    feature_names=["study_hours", "attendance"]
)

print(tree_rules)

# 3. Predict
new_student = [[2, 95]]
prediction = model.predict(new_student)


# Print result
if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")