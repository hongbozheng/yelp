"""
🎯 Goal
Understand which business category combinations are more common in:

✅ Helpful reviews (useful ≥ 3)

❌ Non-helpful reviews (useful == 0)

This gives insight into behavioral and contextual differences behind useful vs.
non-useful feedback.
"""

from pandas import DataFrame

import argparse
import pandas as pd
from mlxtend.frequent_patterns import apriori


def load_merge(review_fp: str, business_fp: str) -> DataFrame:
    print("📂 [INFO] Loading reviews...")
    review_df = pd.read_json(path_or_buf=review_fp, lines=True)
    print("📂 [INFO] Loading businesses...")
    business_df = pd.read_json(path_or_buf=business_fp, lines=True)

    print("🔗 [INFO] Merging on business_id...")
    df = review_df.merge(
        business_df[['business_id', 'categories']],
        on='business_id',
        how='inner',
    )
    df = df.dropna(subset=['categories'])
    print("🧹 [INFO] Splitting and cleaning categories...")
    df['categories'] = df['categories'].str.split(', ')

    return df


def one_hot_encode(df: DataFrame, top_k: int = 50) -> DataFrame:
    print("📊 [INFO] One-hot encoding categories...")
    all_cats = df['categories'].explode().dropna()
    top_cats = all_cats.value_counts().head(top_k).index.tolist()
    print(f"✅ [INFO] Using top {top_k} categories.")

    df['filtered_cats'] = df['categories'].apply(
        lambda x: list(set(x).intersection(top_cats))
    )

    ohe_df = pd.DataFrame(data=False, index=df.index, columns=top_cats)
    for i, cats in df['filtered_cats'].items():
        for cat in cats:
            ohe_df.at[i, cat] = True

    return ohe_df


def run_apriori_analysis(
        df: DataFrame,
        top_k: int,
        min_sup: float,
        label: str,
) -> DataFrame:
    print(f"📊 Running Apriori for {label} reviews...")
    ohe_df = one_hot_encode(df=df, top_k=top_k)
    freq = apriori(ohe_df, min_support=min_sup, use_colnames=True)
    freq['group'] = label

    return freq


def helpful_vs_unhelpful(min_useful: int, top_k: int = 50, min_sup: float = 0.01):
    df = load_merge(
        review_fp="../data/review.json", business_fp="../data/business.json"
    )

    print("🔍 [INFO] Filtering helpful reviews...")
    helpful_df = df[df['useful'] >= min_useful].copy()
    unhelpful_df = df[df['useful'] == 0].copy()

    helpful_patterns = run_apriori_analysis(
        df=helpful_df, top_k=top_k, min_sup=min_sup, label='helpful'
    )
    unhelpful_patterns = run_apriori_analysis(
        df=unhelpful_df, top_k=top_k, min_sup=min_sup, label='non_helpful'
    )

    df = pd.concat([helpful_patterns, unhelpful_patterns])

    pivot = df.pivot_table(
        index='itemsets', columns='group', values='support'
    )
    pivot = pivot.fillna(0)
    pivot['lift_ratio'] = (pivot['helpful'] + 1e-6) / (
                pivot['non_helpful'] + 1e-6)
    pivot = pivot.sort_values(by='lift_ratio', ascending=False)

    print("📌 Top patterns overrepresented in helpful reviews:")
    print(pivot.shape)
    print(pivot.head(10))
    print(pivot.tail(10))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pattern_mining", description="Mine frequent pattern"
    )
    parser.add_argument(
        "--min",
        "-m",
        type=int,
        required=True,
        help="Minimum number of useful needed to retain a pattern",
    )
    parser.add_argument(
        "--top_k",
        "-k",
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

    args = parser.parse_args()
    min_useful = args.min
    min_sup = args.min_sup
    top_k = args.top_k

    helpful_vs_unhelpful(
        min_useful=min_useful,
        top_k=top_k,
        min_sup=min_sup,
    )
