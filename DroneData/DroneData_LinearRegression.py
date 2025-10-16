# DroneDataa_LinearRegression.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# --- 1. Load CSV ---
df = pd.read_csv("uav_logs.csv")

# --- 2. Encode drone type numerically ---
drone_map = {
    'DJI Spark': 1,
    'DJI Mavic 2': 2,
    'DJI Mavic 3': 3,
    'DJI Inspire': 4
}
df['drone_numeric'] = df['drone_make_model'].map(drone_map)

# Drop rows without a mapped drone type
df = df.dropna(subset=['drone_numeric'])

# --- 3. Compute flight duration in minutes using start_time and end_time ---
df['start_time_dt'] = pd.to_datetime(df['start_time'], errors='coerce')
df['end_time_dt'] = pd.to_datetime(df['end_time'], errors='coerce')

# Calculate duration in minutes
df['flight_duration_min'] = (df['end_time_dt'] - df['start_time_dt']).dt.total_seconds() / 60

# Keep only positive durations
df = df[df['flight_duration_min'] > 0]

# Drop rows with invalid times
df = df.dropna(subset=['flight_duration_min'])

# --- 4. Scatter plot: Flight Duration vs Drone Type ---
plt.figure(figsize=(6,4))
sns.scatterplot(x='drone_numeric', y='flight_duration_min', data=df)
plt.xlabel("Drone Type (numeric)")
plt.ylabel("Flight Duration (minutes)")
plt.title("Flight Duration vs Drone Type")
plt.xticks([1,2,3,4], ['Spark','Mavic 2','Mavic 3','Inspire'])
plt.show()

# --- 5. Correlation ---
corr = df[['flight_duration_min', 'drone_numeric']].corr().iloc[0,1]
print(f"Correlation between flight duration and drone type: {corr:.2f}")

# --- 6. Simple Linear Regression ---
X = df[['drone_numeric']]
y = df['flight_duration_min']
model = LinearRegression()
model.fit(X, y)
df['predicted_duration'] = model.predict(X)

# Plot regression line
plt.figure(figsize=(6,4))
sns.scatterplot(x='drone_numeric', y='flight_duration_min', data=df, label='Actual')
sns.lineplot(x='drone_numeric', y='predicted_duration', data=df, color='red', label='Regression')
plt.xlabel("Drone Type (numeric)")
plt.ylabel("Flight Duration (minutes)")
plt.title("Linear Regression: Flight Duration vs Drone Type")
plt.xticks([1,2,3,4], ['Spark','Mavic 2','Mavic 3','Inspire'])
plt.show()

print("Linear regression coefficients:")
print(f"Intercept: {model.intercept_:.2f}, Slope: {model.coef_[0]:.2f}")
