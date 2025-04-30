from pandas import DataFrame

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from config import FEATURES
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import classification_report, confusion_matrix
from utils.utils import load_merge


def mine_patterns(
        df: DataFrame,
        min_useful: int,
        top_k: int,
        min_sup: float,
        min_conf: float,
        min_len: int,
        kulc: float,
):
    print("📊 [INFO] Mining helpful rules...")

    df['label'] = (df['useful'] >= min_useful).astype(int)
    print("📐 [INFO] Selecting features...")
    df = df[FEATURES].fillna(0)

    label_counts = df['label'].value_counts(normalize=True) * 100
    print(f"📊 [INFO] Label Distribution:")
    print(f"[INFO] ➤ Helpful (label=1):     {label_counts.get(1, 0):.4f}%")
    print(f"[INFO] ➤ Not helpful (label=0): {label_counts.get(0, 0):.4f}%")

    print("📊 [INFO] One-hot encoding categories...")
    all_cats = df['categories'].explode().dropna()
    top_cats = all_cats.value_counts().head(top_k).index.tolist()
    print(f"✅ [INFO] Using top {top_k} categories.")

    for cat in top_cats:
        df[cat] = df['categories'].apply(lambda x: int(cat in x))

    # Only use helpful reviews for rule mining
    ohe_df = df[df['label'] == 1][top_cats].astype(bool)
    freq_itemsets = apriori(df=ohe_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(
        df=freq_itemsets, metric='confidence', min_threshold=min_conf
    )

    # Filter rules with multiple antecedents and strong lift
    rules = rules[
        (rules['antecedents'].apply(lambda x: len(x) >= min_len)) &
        (0.5 * (rules['support'] / rules['antecedent support'] +
            rules['support'] / rules['consequent support']
        ) > kulc)
    ]
    print(rules.head(10))
    print(f"✅ [INFO] Mined {len(rules)} rules for classification.")

    return rules, top_cats


def rules_classifier(df, rules, top_cats, min_useful):
    print("🧠 [INFO] Applying rules to classify reviews...")

    # Convert rules to list of antecedent sets
    rule_antecedents = [set(rule) for rule in rules['antecedents']]

    # Precompute review category sets
    review_cat_sets = df[top_cats].astype(bool).apply(
        lambda row: set(row[row].index), axis=1
    )

    # For each review, check if it matches any antecedent
    def match_review(cats):
        return any(rule.issubset(cats) for rule in rule_antecedents)

    print("⚡ [INFO] Classifying reviews via rule match...")
    df['rule_pred'] = [int(match_review(cats)) for cats in review_cat_sets]
    df['label'] = (df['useful'] >= min_useful).astype(int)

    print("📊 [INFO] Rule-Based Classification Report:")
    print(classification_report(df['label'], df['rule_pred']))
    print(confusion_matrix(df['label'], df['rule_pred']))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--min_conf",
        "-c",
        type=float,
        required=True,
        help="Minimum confidence",
    )
    parser.add_argument(
        "--min_len",
        "-l",
        type=int,
        required=True,
        help="Minimum length of itemsets",
    )
    parser.add_argument(
        "--kulc",
        "-k",
        type=float,
        required=True,
        help="Kulczynski Measure",
    )
    args = parser.parse_args()
    min_useful = args.min
    top_k = args.top_k
    min_sup = args.min_sup
    min_conf = args.min_conf
    min_len = args.min_len
    kulc = args.kulc

    df = load_merge(
        review_fp="data/review.json", business_fp="data/business.json"
    )

    rules, top_cats = mine_patterns(
        df=df,
        min_useful=min_useful,
        top_k=top_k,
        min_sup=min_sup,
        min_conf=min_conf,
        min_len=min_len,
        kulc=kulc,
    )

    rules_classifier(
        df=df, rules=rules, top_cats=top_cats, min_useful=min_useful
    )
