from pandas import DataFrame
from typing import List

import argparse
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import classification_report, confusion_matrix
from utils import review_feature


def rule_feature(
        df: DataFrame,
        exclude_cols: List[str],
):
    print("🔁 [INFO] Performing median binarization...")
    exclude_cols.extend(['label', 'useful'])
    cols = df.select_dtypes(include='number').columns.difference(exclude_cols)

    print("📊 [INFO] Distribution of 0s and 1s in binarized columns:")
    for col in cols:
        if col in exclude_cols:
            continue
        df[col] = (df[col] >= df[col].median()).astype(int)
        counts = df[col].value_counts().sort_index()
        count_0 = counts.get(0, 0)
        count_1 = counts.get(1, 0)
        print(f"{col:<25}: 0s = {count_0:<6} | 1s = {count_1}")

    print(df.head(5))
    # print(df.columns)
    print(f"✅ [INFO] Binarized {len(cols)} columns.")


def mine_rule(
        df: DataFrame,
        top_cats: List[str],
        min_sup: float = 0.02,
        min_conf: float = 0.40,
        min_len: int = 2,
        kulc: float = 0.60,
):
    df = rule_feature(df=df, exclude_cols=top_cats)
    return
    print("📊 [INFO] Mining helpful rules...")
    total = df['label'].shape[0]
    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    print(f"📊 [INFO] Label Distribution:")
    print(f"[INFO] ➤ Helpful     [1]: {pos} {pos/total:.4%}")
    print(f"[INFO] ➤ Not helpful [0]: {neg} {neg/total:.4%}")

    df = df.select_dtypes(include='number')
    df = df[df['label'] == 1][top_cats].astype(bool)
    df = df.drop(columns=['label', 'userful', ], errors='ignore')

    print(f"🧠 [INFO] Running Apriori...")
    freq_itemsets = apriori(df=df, min_support=min_sup, use_colnames=True)
    print(f"✅ [INFO] Found {len(freq_itemsets)} frequent itemsets.")

    rules = association_rules(
        df=freq_itemsets, metric='confidence', min_threshold=min_conf
    )

    # Filter rules with multiple antecedents and strong Kulczynski Measure
    rules = rules[
        (rules['antecedents'].apply(lambda x: len(x) >= min_len)) &
        (0.5 * (rules['support'] / rules['antecedent support'] +
            rules['support'] / rules['consequent support']
        ) > kulc)
    ]
    print(rules.head(10))
    print(f"✅ [INFO] Mined {len(rules)} rules for classification.")

    return rules, top_cats


def rule_classifier(df, rules, top_cats, min_useful):
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
        help="Minimum number of useful",
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
    top_k = args.top_k
    min_useful = args.min
    min_sup = args.min_sup
    min_conf = args.min_conf
    min_len = args.min_len
    kulc = args.kulc

    df, top_cats = review_feature(
        review_fp="data/review.json",
        business_fp="data/business.json",
        user_fp="data/user.json",
        checkin_fp="data/checkin.json",
        tip_fp="data/tip.json",
        min_useful=min_useful,
        top_k=top_k,
    )

    rules, top_cats = mine_rule(
        df=df,
        top_cats=top_cats,
        min_sup=min_sup,
        min_conf=min_conf,
        min_len=min_len,
        kulc=kulc,
    )

    rule_classifier(
        df=df, rules=rules, top_cats=top_cats, min_useful=min_useful
    )
