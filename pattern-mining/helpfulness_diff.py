#!/usr/bin/env python3
"""
🎯 Goal
Find the patterns whose support difference between helpful and non-helpful reviews
is the largest (in both directions).

Usage:
  python3 pattern-mining/helpfulness_diff.py -t 100 -u 2 -s 0.05 -l 3 --task category
  python3 pattern-mining/helpfulness_diff.py -t 50  -u 2 -s 0.10 -l 3 --task user
"""

import os, sys
from pandas import DataFrame
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from utils.utils import review_feature, user_feature, binarize


def mine_patterns(df_bool: DataFrame, min_sup: float):
    """
    Runs fpgrowth on df_bool, returns DataFrame with columns ['itemsets','support'].
    """
    return fpgrowth(df=df_bool.astype(bool),
                    min_support=min_sup,
                    use_colnames=True,
                    max_len=6)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="helpfulness_diff",
        description="Find patterns most exclusive to helpful vs non-helpful reviews"
    )
    parser.add_argument("-t", "--top_k",   type=int,   default=50,
                        help="Top-k one-hot features")
    parser.add_argument("-u", "--min",     dest="min_useful", type=int,
                        default=2, help="Threshold for helpful (useful ≥ u)")
    parser.add_argument("-s", "--min_sup", type=float, default=0.2,
                        help="Min support for fpgrowth")
    parser.add_argument("-l", "--k_itemset", dest="k",      type=int,
                        default=3, help="Min itemset size k")
    parser.add_argument("--task", choices=["user", "category"],
                        default="user",
                        help="Which feature set to mine")

    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    # Load and binarize features
    if args.task == "user":
        df_all = user_feature(
            review_fp="../data/review.json",
            business_fp="../data/business.json",
            user_fp="../data/user.json",
            checkin_fp="../data/checkin.json",
            tip_fp="../data/tip.json",
            top_k=args.top_k,
            min_useful=args.min_useful,
            useful_thres=1.2
        )
        df_all = binarize(df=df_all, method=0.80)
        to_drop = [col for col in df_all.columns if (col.startswith("compliment_"))]

        # drop them (along with user_id if you still need that)
        df_all = df_all.drop(columns=to_drop + ['user_id', 'useful', 'useful_user'])
    else:
        df_all, top_cats = review_feature(
            review_fp="../data/review.json",
            business_fp="../data/business.json",
            user_fp="../data/user.json",
            checkin_fp="../data/checkin.json",
            tip_fp="../data/tip.json",
            top_k=args.top_k,
            min_useful=args.min_useful
        )
        df_all = df_all[top_cats + ['label']]

    # Split helpful vs non-helpful
    df_help = df_all[df_all['label'] == 1].reset_index(drop=True)
    df_unh  = df_all[df_all['label'] == 0].reset_index(drop=True)

    # Mine frequent patterns
    help_patterns = mine_patterns(df_help.drop(columns=['label']), min_sup=args.min_sup)
    unh_patterns  = mine_patterns(df_unh .drop(columns=['label']), min_sup=args.min_sup)

    # Extract supports and itemsets
    help_df = help_patterns[['itemsets','support']].rename(columns={'support':'support_help'})
    unh_df  = unh_patterns [['itemsets','support']].rename(columns={'support':'support_unhelp'})

    # Merge on itemsets
    merged = pd.merge(help_df, unh_df, on='itemsets', how='outer').fillna(0)

    # Compute support difference
    merged['support_diff'] = merged['support_help'] - merged['support_unhelp']
    # Compute itemset length
    merged['itemset_len'] = merged['itemsets'].apply(len)
    # Filter by length
    merged = merged[merged['itemset_len'] >= args.k]

    # Top patterns exclusive to helpful (largest positive diff)
    help_excl = merged.sort_values('support_diff', ascending=False) \
        .head(50)
    help_excl.to_csv(f"results/{args.task}-exclusive-helpful.csv", index=False)

    # Top patterns exclusive to unhelpful (most negative diff)
    unh_excl = merged.sort_values('support_diff', ascending=True) \
        .head(50)
    unh_excl.to_csv(f"results/{args.task}-exclusive-unhelpful.csv", index=False)

    print("💾 [INFO] Saved:")
    print(f"  • Exclusive-helpful patterns:   results/{args.task}-exclusive-helpful.csv")
    print(f"  • Exclusive-unhelpful patterns: results/{args.task}-exclusive-unhelpful.csv")