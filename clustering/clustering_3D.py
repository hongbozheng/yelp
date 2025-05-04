from pandas import DataFrame
from typing import List
import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from utils.utils import user_feature

import plotly.express as px


def kmeans_3d_plotly(
    df: DataFrame,
    k_range: List[int],
    random_state: int,
    method: str,
    perplexity: int,
):
    # 构造聚类输入 X，丢弃 user_id 和 label
    X = df.drop(columns=['user_id', 'label'], errors='ignore')

    best_k = None
    best_score = -1
    best_labels = None

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    # 三维降维
    if method.lower() in ("t-sne", "tsne"):
        reducer = TSNE(n_components=3, perplexity=perplexity,
                       learning_rate=200, random_state=random_state)
    elif method.lower() == "pca":
        reducer = PCA(n_components=3, random_state=random_state)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'.")

    proj_3d = reducer.fit_transform(X)

    # 准备可视化 DataFrame，保留原始 label
    df_vis = df.reset_index(drop=True).copy()
    df_vis['cluster'] = best_labels
    df_vis[['x','y','z']] = proj_3d

    # 1) 按 cluster 着色
    fig1 = px.scatter_3d(
        df_vis, x='x', y='y', z='z',
        color='cluster',
        title=f"{method.upper()} 3D KMeans (k={best_k})",
        labels={'x':f'{method.upper()}1','y':f'{method.upper()}2','z':f'{method.upper()}3'},
        width=800, height=600
    )
    fig1.update_traces(marker=dict(size=4))

    # 2) 按 label 着色
    fig2 = px.scatter_3d(
        df_vis, x='x', y='y', z='z',
        color='label',
        title=f"{method.upper()} 3D by Label",
        labels={'x':f'{method.upper()}1','y':f'{method.upper()}2','z':f'{method.upper()}3'},
        width=800, height=600
    )
    fig2.update_traces(marker=dict(size=4))

    fig1.show()
    fig2.show()

    return df_vis, best_k, best_score


def feature_f_scores(df_vis: DataFrame) -> pd.Series:
    """
    基于 ANOVA F-score 评估特征对聚类的区分度。
    返回：按分数降序排列的 Series。
    """
    numeric_feats = df_vis.select_dtypes(include=np.number).columns.drop(['cluster','x','y','z'])
    clusters = df_vis['cluster'].unique()
    sizes = df_vis.groupby('cluster').size()
    grand_means = df_vis[numeric_feats].mean()

    f_scores = {}
    for f in numeric_feats:
        # 组间方差
        means = df_vis.groupby('cluster')[f].mean()
        ss_between = np.sum(sizes * (means - grand_means[f])**2) / (len(clusters)-1)
        # 组内方差（取平均）
        ss_within = df_vis.groupby('cluster')[f].var().mean()
        f_scores[f] = ss_between / ss_within if ss_within > 0 else np.nan

    return pd.Series(f_scores).sort_values(ascending=False)


def feature_importances_rf(df_vis: DataFrame) -> pd.Series:
    """
    基于 RandomForest 预测 cluster 标签，获取特征重要性。
    返回：按重要性降序排列的 Series。
    """
    numeric_feats = df_vis.select_dtypes(include=np.number).columns.drop(['cluster','x','y','z'])
    X = df_vis[numeric_feats]
    y = df_vis['cluster']
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=numeric_feats)
    return importances.sort_values(ascending=False)


def cluster_profile(df_vis: DataFrame) -> pd.DataFrame:
    df_num = df_vis.select_dtypes(include=np.number).drop(columns=['x','y','z'], errors='ignore')
    return df_num.groupby('cluster').mean().round(4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", "-u", type=int, default=2)
    parser.add_argument("--top_k", "-t", type=int, default=50)
    parser.add_argument("--method", "-m", choices=["tsne","pca"], default="tsne")
    parser.add_argument("--perplexity", type=int, default=35)
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # 1. 生成特征
    df = user_feature(
        review_fp="../data/review.json",
        business_fp="../data/business.json",
        user_fp="../data/user.json",
        checkin_fp="../data/checkin.json",
        tip_fp="../data/tip.json",
        top_k=args.top_k,
        min_useful=args.min,
        useful_thres=1.2,
    )

    # 2. 聚类与可视化
    df_vis, best_k, best_score = kmeans_3d_plotly(
        df, [args.kmin, args.kmax],
        random_state=args.random_state,
        method=args.method,
        perplexity=args.perplexity,
    )

    # 3. 打印聚类档案
    print("=== Cluster Profile ===")
    print(cluster_profile(df_vis))

    # 4. 特征贡献度排序
    top_n = 10
    print(f"\n=== Top {top_n} Features by ANOVA F-score ===")
    print(feature_f_scores(df_vis).head(top_n))

    print(f"\n=== Top {top_n} Features by RandomForest Importance ===")
    print(feature_importances_rf(df_vis).head(top_n))
