# weather_pca.py
# Perform PCA on real weather data using Meteostat
# --------------------------------------------------

import pandas as pd
from meteostat import Point, Daily
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Define the location and time range
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)

# Example location: Denver, Colorado
denver = Point(39.7392, -104.9903)

# Step 2: Fetch daily weather data from Meteostat
data = Daily(denver, start, end)
df = data.fetch()

# Step 3: Select relevant numeric features for PCA
df = df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres']].dropna()

# Step 4: Standardize (scale) the data
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# Step 5: Apply PCA with 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled)

# Step 6: Create a DataFrame with principal components
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Step 7: Combine with original data (optional)
df_final = pd.concat([df.reset_index(drop=True), df_pca], axis=1)

# Step 8: Display explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Step 9: Plot the PCA results
plt.figure(figsize=(8,6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c='skyblue', edgecolor='black', alpha=0.7)
plt.title('PCA of Denver Weather Data (2024)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Step 10: Save results to CSV
df_final.to_csv('denver_weather_pca.csv', index=False)
print("\nData with PCA components saved to 'denver_weather_pca.csv'")
