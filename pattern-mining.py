from pandas import DataFrame

import argparse
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules



def one_hot_encode_category(df: DataFrame) -> DataFrame:
    print("📐 [INFO] One-hot encoding categories...")
    categories = df['categories']
    categories = categories.explode().dropna()
    top_cats = categories.value_counts().head(50).index.tolist()
    print(f"✅ [INFO] Using top {len(top_cats)} categories.")

    # Filter only top categories in each row
    df['filtered_cats'] = df['categories'].apply(
        lambda cats: list(set(cats).intersection(top_cats))
    )

    # Binary encode
    ohe_df = pd.DataFrame(data=0, index=df.index, columns=top_cats)
    for i, row in df.iterrows():
        for cat in row['filtered_cats']:
            ohe_df.at[i, cat] = 1

    return ohe_df


def run_apriori(
        ohe_df: DataFrame,
        min_sup: float,
        min_conf: float,
) -> DataFrame:
    print("📊 [INFO] Running Apriori...")
    frequent_itemsets = apriori(
        df=ohe_df, min_support=min_sup,use_colnames=True
    )
    print(f"✅ [INFO] Found {len(frequent_itemsets)} frequent itemsets.")

    print("📈 [INFO] Generating association rules...")
    rules = association_rules(
        df=frequent_itemsets, metric="confidence", min_threshold=min_conf
    )
    rules = rules.sort_values(by='lift', ascending=False)

    return rules


def pattern_mining(
        min_useful: int,
        min_sup: float,
        min_conf: float,
):
    print("📂 [INFO] Loading reviews...")
    review_df = pd.read_json(path_or_buf="data/review.json", lines=True)
    print("📂 [INFO] Loading businesses...")
    business_df = pd.read_json(path_or_buf="data/business.json", lines=True)

    print("🔍 [INFO] Filtering helpful reviews...")
    review_df = review_df[review_df['useful'] >= min_useful]

    print("🔗 [INFO] Merging reviews with business categories...")
    df = review_df.merge(
        business_df[['business_id', 'categories']], on='business_id', how='inner'
    )
    df = df.dropna(subset=['categories'])

    print("🧹 [INFO] Splitting and cleaning categories...")
    df['categories'] = df['categories'].str.split(', ')

    ohe_df = one_hot_encode_category(df=df)

    rules = run_apriori(
        ohe_df=ohe_df, min_sup=min_sup, min_conf=min_conf
    )

    print("📌 [INFO] Sample Top 10 Rules:")
    print(rules[
              ['antecedents', 'consequents', 'support', 'confidence', 'lift']
          ].head(10))

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
        "--min_sup",
        "-s",
        type=float,
        required=True,
        help="Minimum support",
    )
    parser.add_argument(
        "--min_conf",
        "-c",
        type=float,
        required=True,
        help="Minimum confidence",
    )
    args = parser.parse_args()
    min_useful = args.min
    min_sup = args.min_sup
    min_conf = args.min_confidence
    pattern_mining(
        min_useful=min_useful,
        min_sup=min_sup,
        min_conf=min_conf,
    )
