import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from utils.utils import load_merge, build_user_feature_matrix


def hierarchical_clustering(df, sample_size=500, n_clusters=5,
                                distance_metric='euclidean',
                                linkage_method='ward'):
    print("📐 [INFO] Preparing feature matrix...")
    features = df.drop(columns=['user_id'], errors='ignore').select_dtypes(
        include=np.number)

    # Subsample for visualization if needed
    if len(features) > sample_size:
        features = features.sample(n=sample_size, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    print(
        f"🔗 [INFO] Computing linkage matrix with method={linkage_method}, metric={distance_metric}...")
    linkage_matrix = linkage(X_scaled, method=linkage_method,
                             metric=distance_metric)

    # Dendrogram plot
    print("🌲 [INFO] Plotting dendrogram...")
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90.,
               leaf_font_size=10., show_contracted=True)
    plt.title("User Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster Size")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    # Optional: cut the dendrogram to get flat clusters
    print(f"✂️ [INFO] Cutting dendrogram at n_clusters={n_clusters}...")
    model = AgglomerativeClustering(n_clusters=n_clusters,
                                    metric=distance_metric,
                                    linkage=linkage_method)
    cluster_labels = model.fit_predict(X_scaled)

    # Analyze cluster centroids
    cluster_profiles = pd.DataFrame(X_scaled, columns=features.columns)
    cluster_profiles['cluster'] = cluster_labels
    summary = cluster_profiles.groupby('cluster').mean()

    print("📊 [INFO] Cluster profile summary:")
    print(summary)

    return cluster_labels, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="hierarch-clustering", description="Hierarchical Clustering"
    )
    parser.add_argument(
        "--top_k",
        "-t",
        type=int,
        required=True,
        help="Top-k for one-hot encoding",
    )

    args = parser.parse_args()
    top_k = args.top_k

    df = load_merge(
        review_fp="data/review.json", business_fp="data/business.json"
    )
    df = build_user_feature_matrix(
        df=df,
        checkin_fp="data/checkin.json",
        tip_fp="data/tip.json",
        top_k=top_k,
    )

    hierarchical_clustering(df=df)
