# kmeans_job.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    return df

def compute_elbow(df, max_k=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Save elbow plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_k+1), inertias, marker='o')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig(os.path.join(OUTPUT_DIR, 'elbow.png'))
    plt.close()
    
    # Save inertias
    with open(os.path.join(OUTPUT_DIR, 'inertia.json'), 'w') as f:
        json.dump(inertias, f)
    
    return inertias

def run_kmeans(df, k=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Save results
    df.to_csv(os.path.join(OUTPUT_DIR, 'kmeans_output.csv'), index=False)
    
    # Save cluster centers
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns[:-1])
    centers.to_csv(os.path.join(OUTPUT_DIR, 'cluster_centers.csv'), index=False)
    
    # Optional: PCA 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'], cmap='viridis', alpha=0.6)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'PCA Plot of Clusters (k={k})')
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_clusters.png'))
    plt.close()
    
    return df, centers
