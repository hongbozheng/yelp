from pandas import DataFrame

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.utils import review_feature, balance_classes
from xgboost import XGBClassifier


MODELS = {
    'Logistic': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(kernel='linear', probability=True),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


def classification(df: DataFrame, random_state: int, model: str):
    print("📐 [INFO] Creating training data & labels...")
    y = df['label']
    df = df.select_dtypes(include='number')
    df = df.drop(columns=['label', 'useful', ], errors='ignore')
    X = df.fillna(0)
    print(df.head(10))

    total = len(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    print("🔍 [INFO] Label Distribution:")
    print(f"[INFO] Total samples:   {total}")
    print(f"[INFO] Helpful     [1]: {pos} ({pos / total:.4%})")
    print(f"[INFO] Not helpful [0]: {neg} ({neg / total:.4%})")
    print(f"✅ [INFO] Feature matrix: {X.shape}, Target: {y.shape}")

    print(f"🧠 [INFO] Running model: {model}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=random_state
    )

    # Normalize for linear models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MODELS[model]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("📈 ROC AUC:", roc_auc_score(y_test, y_prob))

    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        required=True,
        help="Top-k for one-hot encoding",
    )
    parser.add_argument(
        "--min",
        "-u",
        type=float,
        required=True,
        help="Minimum number of useful",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=['Logistic', 'KNN', 'RandomForest', 'SVM', 'XGBoost'],
        type=str,
        required=True,
        help="Model to use",
    )
    args = parser.parse_args()
    top_k = args.top_k
    min_useful = args.min
    model = args.model

    df, _ = review_feature(
        review_fp="data/review.json",
        business_fp="data/business.json",
        user_fp="data/user.json",
        checkin_fp="data/checkin.json",
        tip_fp="data/tip.json",
        top_k=top_k,
        min_useful=min_useful,
    )

    df = balance_classes(df=df, seed=42)

    classification(df=df, random_state=42, model=model)
