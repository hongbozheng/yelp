"""
python3 classification/rule-classification.py -t 50 -u 2 -s 0.10 -k 3 -c 0.75
"""

from pandas import DataFrame
from typing import List

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.metrics import classification_report, confusion_matrix
from utils.utils import review_feature, rule_feature


FEATURES = [
        'stars', 'cool', 'review', 'Restaurants',
        'Food', 'Nightlife', 'Bars', 'American (New)', 'Breakfast & Brunch',
        'American (Traditional)', 'Sandwiches', 'Coffee & Tea', 'Mexican',
        'Event Planning & Services', 'Seafood', 'Cocktail Bars', 'Shopping',
        'Burgers', 'Pizza', 'Desserts', 'Arts & Entertainment', 'Italian',
        'Salad', 'Cafes', 'Bakeries', 'Specialty Food', 'Beer',
        'Wine & Spirits', 'Japanese', 'Wine Bars', 'Chinese', 'Fast Food',
        'Asian Fusion', 'Sushi Bars', 'Beauty & Spas', 'Venues & Event Spaces',
        'Pubs', 'Caterers', 'Steakhouses', 'Active Life', 'Southern',
        'Chicken Wings', 'Ice Cream & Frozen Yogurt', 'Vegetarian', 'Beer Bar',
        'Sports Bars', 'Juice Bars & Smoothies', 'Grocery', 'Barbeque',
        'Cajun/Creole', 'Tacos', 'Vegan', 'Mediterranean',
        'review_count', 'useful_user', 'funny_user',
        'cool_user', 'friend', 'fans', 'average_stars', 'rev-age',
        # 'compliment_hot', 'compliment_more', 'compliment_note',
        # 'compliment_plain', 'compliment_cool', 'compliment_funny',
        # 'compliment_writer', 'compliment_photos',
        # 'days_since_1st_rev', 'days_since_dataset_start',
        'rev_freq_month', 'rev_freq_std',
    ]


def mine_rule(
        df: DataFrame,
        top_cats: List[str],
        min_sup: float = 0.10,
        k: int = 3,
        kulc: float = 0.80,
):
    print("📊 [INFO] Mining helpful rules...")
    total = df['label'].shape[0]
    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    print(f"📊 [INFO] Label Distribution:")
    print(f"[INFO] ➤ Helpful     [1]: {pos} {pos/total:.4%}")
    print(f"[INFO] ➤ Not helpful [0]: {neg} {neg/total:.4%}")

    df = df.select_dtypes(include='number')
    df = df[df['label'] == 1].astype(bool)
    df = df.drop(columns=['label', 'userful', ], errors='ignore')

    df = df[FEATURES]

    print(f"🧠 [INFO] Running Apriori...")
    # freq_itemsets = apriori(df=df, min_support=min_sup, use_colnames=True)
    freq_itemsets = fpgrowth(df=df, min_support=min_sup, use_colnames=True, max_len=8)
    print(f"✅ [INFO] Found {len(freq_itemsets)} frequent itemsets.")

    rules = association_rules(
        df=freq_itemsets, metric='lift', min_threshold=1.2
    )

    # Filter rules with multiple antecedents and strong Kulczynski Measure
    rules = rules[
        (rules['antecedents'].apply(lambda x: len(x) >= k)) &
        (0.5 * (rules['support'] / rules['antecedent support'] +
            rules['support'] / rules['consequent support']
        ) > kulc)
    ]
    print(rules.head(10))
    print(f"✅ [INFO] Mined {len(rules)} rules for classification.")

    os.makedirs(name='results', exist_ok=True)
    rules.to_csv("results/rules.csv", index=False)

    return rules, top_cats


def rule_classifier(df, rules):
    print("🧠 [INFO] Applying rules to classify reviews...")

    # Convert rules to list of antecedent sets
    rule_antecedents = rules['antecedents'].apply(set).tolist()

    # Precompute review category sets
    review_feature_sets = df[FEATURES].astype(bool).apply(
        lambda row: set(row[row].index), axis=1
    )

    # For each review, check if it matches any antecedent
    def match_review(feat_set):
        return any(rule.issubset(feat_set) for rule in rule_antecedents)

    print(f"⚡ [INFO] Classifying {len(df)} reviews using {len(rule_antecedents)} rules...")
    df['rule_pred'] = review_feature_sets.apply(match_review).astype(int)

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
        "--k_itemset",
        "-k",
        type=int,
        required=True,
        help="k-itemset",
    )
    parser.add_argument(
        "--kulc",
        "-c",
        type=float,
        required=True,
        help="Kulczynski Measure",
    )
    args = parser.parse_args()
    top_k = args.top_k
    min_useful = args.min
    min_sup = args.min_sup
    k = args.k_itemset
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

    df = rule_feature(df=df, exclude_cols=top_cats, percentile=0.80)

    rules, top_cats = mine_rule(
        df=df,
        top_cats=top_cats,
        min_sup=min_sup,
        k=k,
        kulc=kulc,
    )

    rule_classifier(df=df, rules=rules)
