from pandas import DataFrame
from typing import List

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from utils.utils import user_feature


def kmeans_tsne(
        df: DataFrame,
        k_range: List[int],
        random_state: int,
        method: str,
        perplexity: int,
):
    X = df.drop(columns=['user_id', 'label'], errors='ignore')

    best_k = None
    best_score = -1
    best_labels = None

    print(f"🧪 [INFO] Testing KMeans for k in {k_range}...")
    for k in range(k_range[0], k_range[1] + 1):
        print(f"▶️ Trying k = {k}")
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)
        print(f"✅ Silhouette Score (k={k}): {score:.4f}")

        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels
        # if k == 6:
        #     best_k = k
        #     best_score = score
        #     best_labels = labels
        #     break

    print(f"🏆 [INFO] Best k = {best_k} with Silhouette Score = {best_score:.4f}")

    # Project to 2D with t-SNE or PCA
    print(f"📐 [INFO] Projecting with {method.upper()} for 2D visualization...")
    if method == "t-SNE":
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=200,
            random_state=random_state
        )
    elif method == "PCA":
        model = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'.")

    best_proj = model.fit_transform(X)

    print("📊 [INFO] Plotting clusters...")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=best_proj[:, 0],
        y=best_proj[:, 1],
        hue=best_labels,
        palette="tab10",
        s=60
    )
    plt.title(f"{method.upper()} Visualization of KMeans Clusters (k={best_k})")
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    df_clustered = df.copy()
    df_clustered['cluster'] = best_labels

    return df_clustered, best_k, best_score


# def visualize_pca(X, labels):
#     print("🧠 [INFO] Visualizing with PCA...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
#
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', s=50)
#     plt.title("K-Means User Clusters (PCA)")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.legend(title="Cluster")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


def cluster_profile(df: DataFrame):
    df = df.drop(
        columns=['user_id'], errors='ignore'
    ).select_dtypes(include=np.number)
    print("📊 [INFO] Cluster Profile Summary:")
    return df.groupby('cluster').mean().round(4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="clustering", description="K-Means User Clustering"
    )
    parser.add_argument(
        "--min",
        "-u",
        type=int,
        required=True,
        help="Minimum number of useful needed to retain a pattern",
    )
    parser.add_argument(
        "--top_k",
        "-t",
        type=int,
        required=True,
        help="Top-k for one-hot encoding",
    )

    args = parser.parse_args()
    min_useful = args.min
    top_k = args.top_k

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

    df, _, _ = kmeans_tsne(
        df=df, k_range=[2, 8], random_state=42, method="t-SNE", perplexity=35
    )

    profile = cluster_profile(df)
    print(profile)
