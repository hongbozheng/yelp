"""
🎯 Goal
Discover which business category combinations are most common in each city. This helps answer:

What kinds of businesses are popular together in each city?

How does user behavior differ across regions?

This also shows that you understand subgroup analysis, an advanced application of pattern mining.
"""

from pandas import DataFrame

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from mlxtend.frequent_patterns import apriori
from tqdm import tqdm
from utils.utils import load_merge, one_hot_encode


def apriori_per_city(df: DataFrame, top_k: int, min_sup: float) -> DataFrame:
    cities = df['city'].value_counts().index.tolist()
    city_rules = []

    for city in tqdm(iterable=cities, desc="🏙️ [INFO] Running Apriori per city"):
        city_df = df[df['city'] == city].copy()

        # if len(city_df) < 200:  # Skip very small samples
        #     continue

        ohe_df = one_hot_encode(df=city_df, top_k=top_k)

        freq_itemsets = apriori(
            df=ohe_df, min_support=min_sup, use_colnames=True
        )
        freq_itemsets['city'] = city
        city_rules.append(freq_itemsets)

    return pd.concat(city_rules, ignore_index=True)


def city_specific_itemset(top_k: int = 50, min_sup: float = 0.02, k: int = 5):
    df = load_merge(
        review_fp="data/review.json", business_fp="data/business.json"
    )
    all_city_freqs = apriori_per_city(df=df, top_k=top_k, min_sup=min_sup)

    print(f"📌 [INFO] Top {k} itemsets per city:")
    top_per_city = (
        all_city_freqs.sort_values(by=['city', 'support'], ascending=[True, False])
        .groupby('city')
        .head(k)
    )
    print(top_per_city[['city', 'itemsets', 'support']])

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
        "--k_itemset",
        "-k",
        type=int,
        required=True,
        help="k-itemset",
    )

    args = parser.parse_args()
    top_k = args.top_k
    min_sup = args.min_sup
    k = args.k_itemset

    city_specific_itemset(
        top_k=top_k,
        min_sup=min_sup,
        k=k,
    )
