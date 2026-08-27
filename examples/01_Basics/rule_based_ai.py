"""
File Name: rule_based_ai.py
Created: 2026.08.27
Author: Alex
Description: Demonstrate rule-based PASS/FAIL prediction using logical conditions.
"""

study_hours = 5
attendance = 80

if study_hours >= 4 and attendance >= 70:
    result = "PASS"
else:
    result = "FAIL"

print("Study Hours:", study_hours)
print("Attendance:", attendance)
print("Prediction:", result)