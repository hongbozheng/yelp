from pandas import DataFrame

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from mlxtend.frequent_patterns import apriori, association_rules
from utils.utils import load_merge, one_hot_encode, draw_sankey_from_rules, \
    draw_lift_support_heatmap, draw_category_network


def constrained_apriori(df: DataFrame, min_sup: float):
    print("📊 [INFO] Running Apriori...")
    freq_itemsets = apriori(df=df, min_support=min_sup, use_colnames=True)

    return freq_itemsets.sort_values(by='support', ascending=False)


def freq_itemset(
        top_k: int = 50,
        min_sup: float = 0.01,
        min_len: int = 2,
        k: int = 10,
):
    df = load_merge(
        review_fp="data/review.json", business_fp="data/business.json"
    )

    ohe_df = one_hot_encode(df=df, top_k=top_k)

    freq_itemsets = constrained_apriori(df=ohe_df, min_sup=min_sup)

    rules = association_rules(freq_itemsets, metric='lift', min_threshold=1.2)

    freq_itemsets = freq_itemsets[
        freq_itemsets['itemsets'].apply(lambda x: len(x) >= min_len)
    ]

    print(f"📌 [INFO] Top {k} Frequent Itemsets")
    print(freq_itemsets.head(k))

    draw_sankey_from_rules(rules)
    draw_lift_support_heatmap(rules)

    # For NetworkX, use frequent itemsets (not rules)
    draw_category_network(freq_itemsets)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="freq-itemset", description="Mine frequent itemsets"
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

    freq_itemset(
        top_k=top_k,
        min_sup=min_sup,
        min_len=min_len,
        k=k,
    )
