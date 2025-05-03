"""
🎯 Goal
Understand which business category combinations are more common in:

✅ Helpful reviews (useful ≥ 2)

❌ Non-helpful reviews (useful == 0)

This gives insight into behavioral and contextual differences behind useful vs.
non-useful feedback.

user cli:
python3 pattern-mining/helpfulness.py -t 50 -u 2 -s 0.10 -l 3

category cli:
python3 pattern-mining/helpfulness.py -t 100 -u 2 -s 0.05 -l 3

"""

from pandas import DataFrame

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from utils.utils import review_feature, user_feature, binarize


def mine_pattern(
        df: DataFrame,
        min_sup: float = 0.10,
        k: int = 3,
):
    df = df.drop(columns=['label'])
    df = df.astype(bool)
    df = fpgrowth(
        df=df, min_support=min_sup, use_colnames=True, max_len=6
    )

    rules = association_rules(df, metric='lift', min_threshold=1.2)
    rules = rules[rules['antecedents'].apply(lambda x: len(x) >= k)]
    print("📌 Top rules overrepresented in reviews:")
    print(rules.head(10))

    print("🔍 [INFO] Filtering by itemset length...")
    df = df[df['itemsets'].apply(lambda x: len(x) >= k)]

    print("📏 [INFO] Sorting by itemset length (descending)...")
    df['itemset_len'] = df['itemsets'].apply(len)
    df = df.sort_values(by='itemset_len', ascending=False)

    print("📌 Top patterns overrepresented in reviews:")
    print(df.head(10))

    print(f"✅ [INFO] Final pattern table shape: {df.shape}")

    return df, rules


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pattern_mining", description="Mine frequent pattern"
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
    task = "cat"

    if task == "user":
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
        df = binarize(
            df=df,
            method=0.80,
        )
        features = [
            'user_id', 'stars', 'useful', 'funny', 'cool', 'review',
            'review_count',
            'useful_user', 'funny_user', 'cool_user', 'fans',
            'average_stars',
            'compliment_hot', 'compliment_more', 'compliment_note',
            'compliment_plain', 'compliment_cool', 'compliment_funny',
            'compliment_writer', 'compliment_photos', 'friend', 'rev-age',
            'rev_freq_month', 'rev_freq_std',
            # 'check-in',
            'tip_count',
            'label'
        ]
        df = df[features]
        df = df.drop(columns=['user_id', 'useful'])
    else:
        df, top_cats = review_feature(
            review_fp="data/review.json",
            business_fp="data/business.json",
            user_fp="data/user.json",
            checkin_fp="data/checkin.json",
            tip_fp="data/tip.json",
            top_k=top_k,
            min_useful=min_useful,
        )

        top_cats.extend(['label'])
        df = df[top_cats]

    helpful_df, helpful_rules = mine_pattern(
        df=df[df['label'] == 1],
        min_sup=min_sup,
        k=k,
    )

    print(f"💾 [INFO] Saving to `{task}-itemset-helpful.csv`...")
    os.makedirs(name='results', exist_ok=True)
    helpful_df.to_csv(f"results/{task}-itemset-helpful.csv", index=False)
    helpful_rules.to_csv(f"results/{task}-rule-helpful.csv", index=False)

    unhelpful_df, unhelpful_rules = mine_pattern(
        df=df[df['label'] == 0],
        min_sup=min_sup,
        k=k,
    )

    print(f"💾 [INFO] Saving to `{task}-itemset-unhelpful.csv`...")
    os.makedirs(name='results', exist_ok=True)
    unhelpful_df.to_csv(f"results/{task}-itemset-unhelpful.csv", index=False)
    unhelpful_rules.to_csv(f"results/{task}-rule-unhelpful.csv", index=False)
