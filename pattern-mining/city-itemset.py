"""
🎯 Goal
Discover which business category combinations are most common in each city. This helps answer:

What kinds of businesses are popular together in each city?

How does user behavior differ across regions?

This also shows that you understand subgroup analysis, an advanced application of pattern mining.

python3 pattern-mining/city-itemset.py -t 100 -u 2 -s 0.05 -k 4

"""

from pandas import DataFrame

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from utils.utils import review_feature


def mine_pattern(df: DataFrame, min_sup: float, k: int) -> DataFrame:
    results = []

    # Get one-hot encoded category columns
    top_cats = df.drop(columns=['city', 'label']).columns.tolist()

    for city, group in df.groupby('city'):
        ohe = group[top_cats].astype(bool)
        itemsets = fpgrowth(df=ohe, min_support=min_sup, use_colnames=True)
        itemsets = itemsets[itemsets['itemsets'].apply(len) >= k]
        itemsets['city'] = city
        results.append(itemsets)

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="region-freq-pattern", description="Mine city-specific patterns"
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
    min_useful = args.min
    min_sup = args.min_sup
    k = args.k_itemset

    df, top_cats = review_feature(
        review_fp="data/review.json",
        business_fp="data/business.json",
        user_fp="data/user.json",
        checkin_fp="data/checkin.json",
        tip_fp="data/tip.json",
        top_k=top_k,
        min_useful=min_useful,
    )

    top_cats.extend(['city', 'label'])
    df = df[top_cats]

    helpful_df = mine_pattern(
        df=df[df['label'] == 1],
        min_sup=min_sup,
        k=k,
    )
    print(f"💾 [INFO] Saving to `city-itemset-helpful.csv`...")
    os.makedirs(name='results', exist_ok=True)
    helpful_df.to_csv(f"results/city-itemset-helpful.csv", index=False)

    unhelpful_df = mine_pattern(
        df=df[df['label'] == 0],
        min_sup=min_sup,
        k=k,
    )
    print(f"💾 [INFO] Saving to `city-itemset-unhelpful.csv`...")
    os.makedirs(name='results', exist_ok=True)
    unhelpful_df.to_csv(f"results/city-itemset-unhelpful.csv", index=False)
