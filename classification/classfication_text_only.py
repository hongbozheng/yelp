from pandas import DataFrame
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection        import train_test_split
from sklearn.metrics               import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from sklearn.ensemble               import RandomForestClassifier
from sklearn.linear_model           import LogisticRegression
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.svm                    import SVC
from xgboost                        import XGBClassifier

from utils.utils import review_feature

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


MODELS = {
    'Logistic':    LogisticRegression(max_iter=1000, C=1.0, penalty='l2'),
    'RandomForest':RandomForestClassifier(max_depth=10, n_estimators=100),
    'SVM':         SVC(kernel='linear', C=1.0, probability=True),
    'KNN':         KNeighborsClassifier(n_neighbors=5),
    'XGBoost':     XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                 max_depth=6, n_estimators=100)
}


def classification_text_only(
    df: DataFrame,
    random_state: int,
    model_name: str,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: float = 0.01,
    max_df: float = 0.9
):
    """
    Train and evaluate a classifier using only TF-IDF features from `text`,
    printing both train and test accuracy to check for overfitting.
    """

    print("📐 [INFO] Extracting text & labels...")
    y = df['label']
    texts = df['text'].fillna("")

    print(f"📊 [INFO] Vectorizing text (ngram_range={ngram_range}, max_features={max_features}, "
          f"min_df={min_df}, max_df={max_df})...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    X_tfidf = vectorizer.fit_transform(texts)
    print(f"✅ [INFO] TF-IDF matrix shape: {X_tfidf.shape}")

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y,
        stratify=y,
        random_state=random_state
    )
    print(f"✅ [INFO] Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # 3) Choose and train model
    clf = MODELS[model_name]
    print(f"🧠 [INFO] Training {model_name} on TF-IDF features...")
    clf.fit(X_train, y_train)

    # 4) Evaluate on train set
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"🧪 [TRAIN] Accuracy: {train_acc:.4f}")
    print("🔍 [TRAIN] Classification Report:")
    print(classification_report(y_train, y_train_pred, digits=4))

    # 5) Predict & evaluate on test set
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    print(f"🧪 [TEST ] Accuracy: {test_acc:.4f}")
    print("🔍 [TEST ] Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("📈 ROC AUC:", roc_auc_score(y_test, y_prob))

    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text-only classification using TF-IDF features"
    )
    parser.add_argument(
        "-f", "--max_features",
        type=int, default=5000,
        help="Maximum number of TF-IDF features"
    )
    parser.add_argument(
        "-m", "--model",
        choices=list(MODELS.keys()), default="RandomForest",
        help="Which classifier to use"
    )
    parser.add_argument(
        "-n", "--ngram_range",
        nargs=2, type=int, default=[1, 4],
        metavar=('MIN_N', 'MAX_N'),
        help="The n-gram range for TF-IDF (e.g. 1 2)"
    )
    parser.add_argument(
        "-u", "--min",
        type=int, default=2,
        help="Minimum 'useful' threshold (passed to review_feature)"
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int, default=50,
        help="Top-k for one-hot encoding (passed to review_feature)"
    )
    # new args to help control overfitting
    parser.add_argument(
        "--min_df", type=float, default=0.01,
        help="TF-IDF min_df (ignore very rare terms)"
    )
    parser.add_argument(
        "--max_df", type=float, default=0.9,
        help="TF-IDF max_df (ignore too-frequent terms)"
    )

    args = parser.parse_args()

    # build initial DataFrame (with text & label)
    df, _ = review_feature(
        review_fp="../data/review.json",
        business_fp="../data/business.json",
        user_fp="../data/user.json",
        checkin_fp="../data/checkin.json",
        tip_fp="../data/tip.json",
        top_k=args.top_k,
        min_useful=args.min,
    )

    classification_text_only(
        df=df,
        random_state=42,
        model_name=args.model,
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        min_df=args.min_df,
        max_df=args.max_df
    )

    # texts = df['text'].fillna("").tolist()
    # vec = TfidfVectorizer(stop_words='english', ngram_range=(2, 2))
    # X = vec.fit_transform(texts)
    # names = vec.get_feature_names_out()
    #
    # mean_h = X[df['label'] == 1].mean(axis=0).A1
    # mean_u = X[df['label'] == 0].mean(axis=0).A1
    # diff = mean_h - mean_u
    #
    # k = 50
    # idx_h = np.argsort(diff)[-k:]
    # idx_u = np.argsort(diff)[:k]
    #
    # help_freq = {names[i]: diff[i] for i in idx_h}
    # unhelp_freq = {names[i]: -diff[i] for i in idx_u}
    #
    # # --- Option A: built-in colormap ---
    # wc1 = WordCloud(
    #     width=800, height=400,
    #     background_color='white',
    #     colormap='Set2',  # ← nice pastel set of colors
    # ).generate_from_frequencies(help_freq)
    #
    # wc2 = WordCloud(
    #     width=800, height=400,
    #     background_color='white',
    #     colormap='Set2',
    # ).generate_from_frequencies(unhelp_freq)
    #
    # # --- Option B: custom palette via color_func ---
    # # colors = ["#2ca02c", "#9467bd", "#1f77b4"]  # green, purple, blue
    # # def my_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # #     return random.choice(colors)
    # # wc1 = WordCloud(width=800, height=400, background_color='white',
    # #                 color_func=my_color_func) \
    # #           .generate_from_frequencies(help_freq)
    # # wc2 = WordCloud(width=800, height=400, background_color='white',
    # #                 color_func=my_color_func) \
    # #           .generate_from_frequencies(unhelp_freq)
    #
    # # 3) plot
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # axes[0].imshow(wc1, interpolation='bilinear')
    # axes[0].axis('off')
    # axes[0].set_title("Helpful", fontsize=32, weight='bold')
    #
    # axes[1].imshow(wc2, interpolation='bilinear')
    # axes[1].axis('off')
    # axes[1].set_title("Unhelpful", fontsize=32, weight='bold')
    #
    # plt.tight_layout()
    # plt.show()
