"""
🎯 Goal
Understand which business category combinations are more common in:

✅ Helpful reviews (useful ≥ 2)

❌ Non-helpful reviews (useful == 0)

This gives insight into behavioral and contextual differences behind useful vs.
non-useful feedback.
"""

from pandas import DataFrame

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from utils.utils import user_feature, binarize


def apriori_analysis(
        df: DataFrame,
        min_sup: float,
        label: str,
) -> DataFrame:
    print("🔍 [INFO] Filtering user features...")
    df = df.select_dtypes(include='number')
    if label == 'helpful':
        df = df[df['label'] == 1]
    else:
        df = df[df['label'] == 0]
    df = df.drop(columns=['label', 'userful', ], errors='ignore')
    # print(df.shape)

    print(f"📊 Running Apriori for {label} reviews...")
    # freq_itemsets = apriori(df=df, min_support=min_sup, use_colnames=True)
    freq_itemsets = fpgrowth(df=df, min_support=min_sup, use_colnames=True, max_len=6)
    freq_itemsets['group'] = label

    return freq_itemsets


def user_helpfulness(
        df: DataFrame,
        min_sup: float = 0.10,
        min_len: int = 3,
):
    features = [
        'user_id', 'stars', 'useful', 'funny', 'cool', 'review', 'review_count',
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

    # Apply apriori to helpful & non-helpful separately
    helpful_patterns = apriori_analysis(df=df, min_sup=min_sup, label='helpful')
    unhelpful_patterns = apriori_analysis(
        df=df, min_sup=min_sup, label='unhelpful'
    )

    # Merge both
    df = pd.concat([helpful_patterns, unhelpful_patterns])

    print("🔍 [INFO] Filtering by itemset length...")
    df = df[df['itemsets'].apply(lambda x: len(x) >= min_len)]

    print("📏 [INFO] Sorting by itemset length (descending)...")
    df['itemset_len'] = df['itemsets'].apply(len)
    df = df.sort_values(by='itemset_len', ascending=False)

    # # Pivot: itemsets as index, columns = support per group
    # df = df.pivot_table(
    #     index='itemsets', columns='group', values='support', fill_value=0
    # )
    # df = df.reset_index()
    #
    # # Confidence metrics
    # total = df['helpful'] + df['unhelpful'] + 1e-9  # avoid divide-by-zero
    # df['conf_helpful'] = df['helpful'] / total
    # df['conf_unhelpful'] = df['unhelpful'] / total
    # df['total_support'] = total
    #
    # # Sort by most discriminative helpful patterns
    # df = df.sort_values(
    #     by=['conf_helpful', 'total_support'], ascending=[False, False]
    # )

    print("📌 Top patterns overrepresented in helpful reviews:")
    print(df.head(10))
    print(df.tail(10))

    print("💾 [INFO] Saving to `itemset-helpfulness.csv`...")
    os.makedirs(name='results', exist_ok=True)
    df.to_csv("results/itemset-helpfulness.csv", index=False)

    print(f"✅ [INFO] Final pattern table shape: {df.shape}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pattern_mining", description="Mine frequent pattern"
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
        "--min_len",
        "-l",
        type=float,
        required=True,
        help="Minimum length of itemset",
    )

    args = parser.parse_args()
    min_useful = args.min
    top_k = args.top_k
    min_sup = args.min_sup
    min_len = args.min_len

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

    user_helpfulness(
        df=df,
        min_sup=min_sup,
        min_len=min_len,
    )
