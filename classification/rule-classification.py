from pandas import DataFrame
from typing import List

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.metrics import classification_report, confusion_matrix
from utils.utils import review_feature, rule_feature


def mine_rule(
        df: DataFrame,
        top_cats: List[str],
        min_sup: float = 0.10,
        min_conf: float = 0.50,
        min_len: int = 2,
        kulc: float = 0.80,
):
    print("📊 [INFO] Mining helpful rules...")
    total = df['label'].shape[0]
    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    print(f"📊 [INFO] Label Distribution:")
    print(f"[INFO] ➤ Helpful     [1]: {pos} {pos/total:.4%}")
    print(f"[INFO] ➤ Not helpful [0]: {neg} {neg/total:.4%}")
    # print(df.columns)
    # exit(0)
    df = df.select_dtypes(include='number')
    df = df[df['label'] == 1].astype(bool)
    df = df.drop(columns=['label', 'userful', ], errors='ignore')

    features = [
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

    df = df[features]

    print(f"🧠 [INFO] Running Apriori...")
    # freq_itemsets = apriori(df=df, min_support=min_sup, use_colnames=True)
    freq_itemsets = fpgrowth(df=df, min_support=min_sup, use_colnames=True, max_len=5)
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

    os.makedirs(name='results', exist_ok=True)
    rules.to_csv("results/rules.csv", index=False)

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

    df = rule_feature(df=df, exclude_cols=top_cats, percentile=0.80)

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
