#Create a K-means clustering algorithm to group customers of a retail store based on their purchase history


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Select the features to use for clustering
features = ['Annual Income (k$)','Spending Score (1-100)']
a = df[features]

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
x_scaled = scaler.fit_transform(a)

# Choose the number of clusters
n_clusters = 5

# Create the KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Fit the model to the scaled data
kmeans.fit(x_scaled)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Add the cluster labels to the dataframe
df['Cluster'] = labels

# Plot the clusters
plt.figure(figsize=(10,7))
plt.scatter(a['Annual Income (k$)'], a['Spending Score (1-100)'], c=labels, label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering')
plt.show()


'''#plot bar graph
plt.bar(a['Annual Income (k$)'], a['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Bar Graph')
plt.show()'''