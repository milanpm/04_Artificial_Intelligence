from sklearn.tree import DecisionTreeClassifier, export_text


# Training features
# Each row: [study_hours, attendance_rate]
X_train = [
    [1, 50],
    [2, 90],
    [3, 70],
    [4, 60],
    [4, 80],
    [5, 70],
    [5, 90],
    [6, 75],
    [6, 95],
]

# PASS requires sufficient study time and attendance.
y_train = [
    "FAIL",  # Low study time and low attendance
    "FAIL",  # High attendance, but insufficient study time
    "FAIL",  # Insufficient study time
    "FAIL",  # Sufficient study time, but low attendance
    "PASS",  # Sufficient study time and attendance
    "FAIL",  # Sufficient study time, but low attendance
    "PASS",  # Sufficient study time and attendance
    "FAIL",  # Attendance is below the required level
    "PASS",  # Sufficient study time and attendance
]

print("=== Training Data ===")

for features, label in zip(X_train, y_train):
    study_hours = features[0]
    attendance_rate = features[1]

    print(
        f"Study Hours: {study_hours}, "
        f"Attendance: {attendance_rate}%, "
        f"Label: {label}"
    )

print()
print(f"Number of samples: {len(X_train)}")
print(f"Number of features: {len(X_train[0])}")
print(f"Number of labels: {len(y_train)}")

# Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

tree_rules = export_text(
    model,
    feature_names=["study_hours", "attendance_rate"],
)

print()
print("=== Learned Rules ===")
print(tree_rules)

# New student data
# [study_hours, attendance_rate]
new_students = [
    [2, 85],
    [5, 75],
    [6, 90],
]

# Predict labels
predictions = model.predict(new_students)

print()
print("=== Predictions ===")

for features, prediction in zip(new_students, predictions):
    study_hours = features[0]
    attendance_rate = features[1]

    print(
        f"Study Hours: {study_hours}, "
        f"Attendance: {attendance_rate}%, "
        f"Prediction: {prediction}"
    )
