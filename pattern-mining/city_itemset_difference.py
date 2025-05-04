#!/usr/bin/env python3
"""
🎯 Goal
Discover which business category combinations are most different between helpful and unhelpful reviews in each city.
Usage:
    python3 pattern-mining/city-itemset.py -t 100 -u 2 -s 0.05 -k 4
"""

from pandas import DataFrame
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from utils.utils import review_feature


def mine_pattern(df: DataFrame, min_sup: float, k: int) -> DataFrame:
    """
    For each city, find all frequent itemsets of length >= k with support >= min_sup.
    """
    results = []
    # one-hot columns are everything except 'city' and 'label'
    cats = [c for c in df.columns if c not in ('city', 'label')]
    for city, grp in df.groupby('city'):
        ohe = grp[cats].astype(bool)
        itemsets = fpgrowth(df=ohe, min_support=min_sup, use_colnames=True)
        itemsets = itemsets[itemsets['itemsets'].apply(len) >= k].copy()
        itemsets['city'] = city
        results.append(itemsets)
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['support', 'itemsets', 'city'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="city-itemset",
        description="Mine and compare city-specific category itemsets"
    )
    parser.add_argument(
        "--top_k", "-t",
        type=int, default=100,
        help="Top-k categories for one-hot encoding (default: 100)"
    )
    parser.add_argument(
        "--min", "-u",
        dest="min_useful",
        type=int, default=2,
        help="Minimum number of useful to retain a review (default: 2)"
    )
    parser.add_argument(
        "--min_sup", "-s",
        type=float, default=0.05,
        help="Minimum support threshold (default: 0.05)"
    )
    parser.add_argument(
        "--k_itemset", "-k",
        dest="k",
        type=int, default=4,
        help="Minimum itemset size k (default: 4)"
    )
    args = parser.parse_args()

    # load and filter features
    df, top_cats = review_feature(
        review_fp="../data/review.json",
        business_fp="../data/business.json",
        user_fp="../data/user.json",
        checkin_fp="../data/checkin.json",
        tip_fp="../data/tip.json",
        top_k=args.top_k,
        min_useful=args.min_useful,
    )
    # keep only the one-hot cats plus city and label
    cols = top_cats + ['city', 'label']
    df = df[cols]

    # mine for helpful and unhelpful separately
    helpful = mine_pattern(df[df['label'] == 1], min_sup=args.min_sup, k=args.k)
    unhelpful = mine_pattern(df[df['label'] == 0], min_sup=args.min_sup, k=args.k)

    os.makedirs('results', exist_ok=True)
    helpful.to_csv("results/city-itemset-helpful.csv", index=False)
    unhelpful.to_csv("results/city-itemset-unhelpful.csv", index=False)
    print("💾 [INFO] Saved helpful and unhelpful itemsets.")

    # compute difference in support per city & itemset
    diff = pd.merge(
        helpful[['city','itemsets','support']],
        unhelpful[['city','itemsets','support']],
        on=['city','itemsets'], how='outer', suffixes=('_helpful','_unhelpful')
    ).fillna(0)
    diff['support_diff'] = diff['support_helpful'] - diff['support_unhelpful']
    # sort by absolute difference descending
    diff = diff.reindex(diff.support_diff.abs().sort_values(ascending=False).index)

    diff.to_csv("results/city-itemset-diff.csv", index=False)
    print("💾 [INFO] Saved difference-ranked itemsets to results/city-itemset-diff.csv")
