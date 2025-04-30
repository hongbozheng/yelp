"""
🎯 Goal
Discover which business category combinations are most common in each city. This helps answer:

What kinds of businesses are popular together in each city?

How does user behavior differ across regions?

This also shows that you understand subgroup analysis, an advanced application of pattern mining.
"""

from pandas import DataFrame

import argparse
import pandas as pd
from mlxtend.frequent_patterns import apriori
from tqdm import tqdm


def load_merge(review_fp: str, business_fp: str) -> DataFrame:
    print("📂 [INFO] Loading reviews...")
    review_df = pd.read_json(path_or_buf=review_fp, lines=True)
    print("📂 [INFO] Loading businesses...")
    business_df = pd.read_json(path_or_buf=business_fp, lines=True)

    print("🔗 Merging to attach city and categories...")
    business_df = business_df[['business_id', 'city', 'categories']].dropna()
    business_df['categories'] = business_df['categories'].str.split(', ')
    df = review_df.merge(business_df, on='business_id', how='inner')

    return df


def one_hot_encode(df: DataFrame, top_k: int = 50) -> DataFrame:
    all_cats = df['categories'].explode().dropna()
    top_cats = all_cats.value_counts().head(top_k).index.tolist()

    df['filtered_cats'] = df['categories'].apply(
        lambda x: list(set(x).intersection(top_cats)))

    ohe_df = pd.DataFrame(data=False, index=df.index, columns=top_cats)
    for i, cats in df['filtered_cats'].items():
        for cat in cats:
            ohe_df.at[i, cat] = True

    return ohe_df


def run_apriori_per_city(df,  top_k: int, min_sup: float) -> DataFrame:
    cities = df['city'].value_counts().index.tolist()
    city_rules = []

    for city in tqdm(iterable=cities, desc="🏙️ [INFO] Running Apriori per city"):
        city_df = df[df['city'] == city].copy()

        # if len(city_df) < 200:  # Skip very small samples
        #     continue

        ohe_df = one_hot_encode(city_df, top_k=top_k)

        freq_itemsets = apriori(
            df=ohe_df, min_support=min_sup, use_colnames=True
        )
        freq_itemsets['city'] = city
        city_rules.append(freq_itemsets)

    return pd.concat(city_rules, ignore_index=True)


def city_specific_pattern(top_k: int = 50, min_sup: float = 0.02):
    df = load_merge(
        review_fp="../data/review.json", business_fp="../data/business.json"
    )
    all_city_freqs = run_apriori_per_city(df=df, top_k=top_k, min_sup=min_sup)

    print(f"📌 [INFO] Top 5 itemsets per city:")
    top_per_city = (
        all_city_freqs.sort_values(by=['city', 'support'], ascending=[True, False])
        .groupby('city')
        .head(5)
    )
    print(top_per_city[['city', 'itemsets', 'support']])

    return all_city_freqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="region-freq-pattern", description="Mine city-specific patterns"
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
    min_sup = args.min_sup
    top_k = args.top_k

    city_specific_pattern(
        top_k=top_k,
        min_sup=min_sup,
    )
