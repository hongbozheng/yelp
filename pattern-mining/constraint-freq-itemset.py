"""
🎯 Goal
Discover which business category combinations are most common in each city. This helps answer:

What kinds of businesses are popular together in each city?

How does user behavior differ across regions?

This also shows that you understand subgroup analysis, an advanced application of pattern mining.
"""

from pandas import DataFrame
from typing import List

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from mlxtend.frequent_patterns import apriori
from utils.utils import load_merge


def one_hot_encode(df: DataFrame, top_k: int, cats: List[str]):
    print("📐 One-hot encoding (with constraints applied)...")
    all_cats = df['categories'].explode().dropna()
    top_cats = all_cats.value_counts().head(top_k).index.tolist()

    df['filtered_cats'] = df['categories'].apply(
        lambda x: list(set(x).intersection(top_cats))
    )

    # Only include rows that contain required categories (anti-monotone constraint)
    df = df[df['filtered_cats'].apply(lambda x: any(c in x for c in cats))]

    ohe_df = pd.DataFrame(data=False, index=df.index, columns=top_cats)
    for i, cats in df['filtered_cats'].items():
        for cat in cats:
            ohe_df.at[i, cat] = True

    return ohe_df


def constrained_apriori(
        df: DataFrame,
        min_sup: float,
):
    print("🔍 Running Apriori with constraints...")
    freq_itemsets = apriori(df=df, min_support=min_sup, use_colnames=True)

    return freq_itemsets.sort_values(by='support', ascending=False)


def constraint_freq_itemset(
        top_k: int = 50,
        cats: List[str] = ["Bars"],
        min_sup: float = 0.01,
        min_len: int = 2,
        k: int = 10,
):
    df = load_merge(
        review_fp="data/review.json", business_fp="data/business.json"
    )
    ohe_df = one_hot_encode(df=df, top_k=top_k, cats=cats)
    freq_itemsets = constrained_apriori(df=ohe_df, min_sup=min_sup)

    # Constraint: only keep itemsets that contain required categories
    freq_itemsets = freq_itemsets[
        freq_itemsets['itemsets'].apply(lambda x: any(c in x for c in cats))
    ]
    # Constraint: itemset length ≥ 2
    freq_itemsets = freq_itemsets[
        freq_itemsets['itemsets'].apply(lambda x: len(x) >= min_len)
    ]

    print(f"📌 [INFO] Top {k} Constrained freq_itemsets (must contain {cats}):")
    print(freq_itemsets[['itemsets', 'support']].head(k))

    return


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

    constraint_freq_itemset(
        top_k=top_k,
        cats=["Bars"],
        min_sup=min_sup,
        min_len=min_len,
        k=k,
    )
