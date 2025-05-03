"""
🎯 Goal
Use DBSCAN to cluster users based on behavior, especially to:

Identify dense reviewer groups (e.g., active foodies)

Flag noise/outliers (e.g., erratic or bot-like users)

Capture arbitrary-shaped clusters, unlike K-Means
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from utils.utils import user_feature


def dbscan_clustering(df, eps=1.5, min_samples=5, show_kdist=True):
    print("📐 [INFO] Preparing feature matrix...")
    features = df.drop(columns=['user_id'], errors='ignore').select_dtypes(include=np.number)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Optional: visualize k-distance for eps tuning
    if show_kdist:
        print("🔍 [INFO] Computing k-distance for eps tuning...")
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])
        plt.figure(figsize=(8, 4))
        plt.plot(k_distances)
        plt.axhline(y=eps, color='red', linestyle='--', label=f"eps = {eps}")
        plt.title("k-distance Plot for DBSCAN")
        plt.xlabel("Sorted Points")
        plt.ylabel("k-distance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"🔗 [INFO] Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    # Append cluster labels
    result_df = df.copy()
    result_df['dbscan_cluster'] = labels

    # Cluster summary
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    print("📊 [INFO] Cluster size distribution:")
    print(cluster_counts)

    return result_df, cluster_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="dbscan", description="Depth-based Scan"
    )
    parser.add_argument(
        "--top_k",
        "-t",
        type=int,
        required=True,
        help="Top-k for one-hot encoding",
    )
    parser.add_argument(
        "--min",
        "-u",
        type=int,
        required=True,
        help="Minimum number of useful needed to retain a pattern",
    )

    args = parser.parse_args()
    top_k = args.top_k
    min_useful = args.min

    df = user_feature(
        review_fp="data/review.json",
        business_fp="data/business.json",
        user_fp="data/user.json",
        checkin_fp="data/checkin.json",
        tip_fp="data/tip.json",
        top_k=top_k,
        min_useful=min_useful,
        useful_thres=1.2,
    )

    dbscan_clustering(df=df)
