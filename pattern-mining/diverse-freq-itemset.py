"""
From a large pool of frequent itemsets:

Select k itemsets that are not only frequent, but also diverse and non-redundant

Avoid dozens of repetitive patterns like:

{Bars}, {Bars, Nightlife}, {Bars, Nightlife, Lounges}, etc.

This shows awareness of pattern explosion and how to deal with it using diversity-aware summarization.
"""

from pandas import DataFrame

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from mlxtend.frequent_patterns import apriori
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from utils.utils import load_merge, one_hot_encode


def apriori_all(df: DataFrame, min_sup: float):
    freq_itemsets = apriori(df=df, min_support=min_sup, use_colnames=True)

    return freq_itemsets


def top_k_diverse_itemsets(freq_itemsets, k: int = 10):
    print(f"🧠 [INFO] Selecting top-{k} diverse itemsets...")

    # Step 1: Build binary matrix of itemsets
    items = sorted({item for s in freq_itemsets['itemsets'] for item in s})
    itemsets_bin = freq_itemsets['itemsets'].apply(
        lambda s: [int(i in s) for i in items]
    ).tolist()
    bin_matrix = pd.DataFrame(itemsets_bin, columns=items)

    # ✅ FIX: Convert to NumPy boolean array for Jaccard
    distances = pairwise_distances(
        bin_matrix.values.astype(bool), metric='jaccard'
    )

    # Step 3: Agglomerative clustering into k groups
    clustering = AgglomerativeClustering(
        n_clusters=k, metric='precomputed', linkage='average'
    )
    labels = clustering.fit_predict(distances)

    # Step 4: For each cluster, pick the most frequent itemset
    freq_itemsets['cluster'] = labels
    top_diverse = (
        freq_itemsets.sort_values(by='support', ascending=False)
        .groupby('cluster')
        .first()
        .reset_index()
    )

    return top_diverse[['itemsets', 'support', 'cluster']]


def top_k_diverse_patterns(
        top_k: int = 50,
        min_sup: float = 0.01, min_len: int = 2,
        k: int = 10,
):
    df = load_merge(
        review_fp="../data/review.json", business_fp="../data/business.json"
    )

    print("🔍 [INFO] Running Apriori mining...")
    ohe_df = one_hot_encode(df=df, top_k=top_k)
    freq_itemsets = apriori_all(df=ohe_df, min_sup=min_sup)

    print("🌈 [INFO] Extracting top-k diverse itemsets...")
    top_diverse = top_k_diverse_itemsets(freq_itemsets=freq_itemsets, k=k)

    print(f"📌 [INFO] Top-{k} Diverse Itemsets:")
    print(top_diverse)

    return top_diverse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="region-freq-pattern", description="Mine city-specific patterns"
    )
    parser.add_argument(
        "--top_k",
        "-t",
        type=int,
        required=True,
        help="Top-k for one-hot encoding",
    )
    parser.add_argument(
        "--min_sup",
        "-s",
        type=float,
        required=True,
        help="Minimum support",
    )
    parser.add_argument(
        "--min_len",
        "-l",
        type=float,
        required=True,
        help="Minimum length of itemset",
    )
    parser.add_argument(
        "--k_itemset",
        "-k",
        type=int,
        required=True,
        help="k-itemset",
    )

    args = parser.parse_args()
    top_k = args.top_k
    min_sup = args.min_sup
    min_len = args.min_len
    k = args.k_itemset

    top_k_diverse_patterns(
        top_k=top_k,
        min_sup=min_sup,
        min_len=min_len,
        k=k,
    )
