"""
drone_regression.py
-------------------
Performs a multidimensional linear regression on UAV log data
to predict the surveyed area based on flight and registration features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("uav_logs.csv")

# Step 2: Drop irrelevant or purely textual columns
drop_cols = [
    "timestamp", "start_date", "end_date", "start_time", "end_time",
    "summary_description_of_flight", "privacy_risks_summary_of", "privacy_risk_mitigation_action"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Step 3: Drop rows with missing values
df = df.dropna()

# Step 4: Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Define features (X) and target (y)
# We'll predict 'location_area_surveyed' based on other features
target_col = "location_area_surveyed"
X = df.drop(columns=[target_col])
y = df[target_col]

# Step 6: Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Regression Results ===")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R² Score: {r2:.3f}")

# Step 10: Visualizations

# (a) Scatter: Predicted vs Actual
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color="blue", edgecolor="k")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()

# (b) 2D slice: example vs one feature (for visualization simplicity)
example_feature = "type_of_data_collected"
if example_feature in X.columns:
    plt.figure(figsize=(7,5))
    plt.scatter(X_test[example_feature], y_test, color="green", label="Actual", alpha=0.6)
    plt.scatter(X_test[example_feature], y_pred, color="orange", label="Predicted", alpha=0.6)
    plt.xlabel(example_feature)
    plt.ylabel("Location Area Surveyed")
    plt.title(f"2D Slice: {example_feature} vs Predicted Area")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Step 11: Display coefficients
coef_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\n=== Feature Importance (Coefficients) ===")
print(coef_table)
