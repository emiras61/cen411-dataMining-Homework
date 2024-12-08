# Importing necessary libraries
import kagglehub
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 1: Download the dataset
# Downloading the World Happiness Report (2019) dataset
path = kagglehub.dataset_download("unsdsn/world-happiness")
data_path = f"{path}/2019.csv"
print("Dataset downloaded:", data_path)

# Step 2: Load the dataset
# Load the dataset using pandas
data = pd.read_csv(data_path)
print("First 5 rows of the dataset:\n", data.head())

# Step 3: Feature selection
# Select only numeric columns for clustering
numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
X = data[numeric_columns]

# Step 4: Scale the features
# Normalize the data to ensure all features contribute equally to clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Determine the optimal number of clusters (Elbow method)
# Using the Elbow method to find the optimal number of clusters
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow method results
plt.plot(k_range, inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.show()

# Step 6: Train the KMeans model
# Based on the Elbow method, select the optimal number of clusters (e.g., k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Step 7: Evaluate clustering performance using Silhouette Score
# Calculate the silhouette score to evaluate clustering quality
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score (k={optimal_k}): {silhouette_avg}")

# Step 8: Visualize clusters using PCA
# Use PCA to reduce dimensions to two for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter plot of clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
plt.title(f"KMeans Clustering (k={optimal_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Step 9: Analyze cluster characteristics
# Add cluster labels to the original dataset for analysis
data["Cluster"] = clusters

# Group by clusters and calculate mean for numeric columns
cluster_summary = data.groupby("Cluster")[numeric_columns].mean()
print("\nCluster Summary:")
print(cluster_summary)