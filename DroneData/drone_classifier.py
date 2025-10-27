"""
drone_classifier.py
-------------------
This program reads UAV log data from uav_logs.csv and uses
KNN and Decision Tree classifiers to categorize the flights
based on the 'authorized_use_select_all' field.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load dataset
df = pd.read_csv("uav_logs.csv")

# Step 2: Drop columns that are mostly timestamps, text summaries, or duplicates
# (These are not useful for classification directly)
drop_cols = [
    "timestamp", "start_date", "end_date", "start_time", "end_time",
    "summary_description_of_flight", "privacy_risks_summary_of", "privacy_risk_mitigation_action"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Step 3: Handle missing values
df = df.dropna()

# Step 4: Encode all categorical columns
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Define features and target
target_col = "authorized_use_select_all"
X = df.drop(columns=[target_col])
y = df[target_col]

# Step 6: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Step 8: Train Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Step 9: Evaluate both models
print("\n=== KNN Results ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_knn), 3))
print(classification_report(y_test, y_pred_knn))

print("\n=== Decision Tree Results ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt), 3))
print(classification_report(y_test, y_pred_dt))

# Step 10: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, ax=axes[0], cmap="Blues")
axes[0].set_title("KNN Confusion Matrix")
ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test, ax=axes[1], cmap="Greens")
axes[1].set_title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()

# Step 11: (Optional) Visualize Decision Tree
plt.figure(figsize=(16, 8))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=[str(c) for c in dt.classes_],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Visualization")
plt.show()
