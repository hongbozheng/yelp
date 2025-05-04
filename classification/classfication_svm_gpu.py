# classification_thundersvm.py

from pandas import DataFrame
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np

from sklearn.ensemble       import RandomForestClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.preprocessing  import StandardScaler
from xgboost                import XGBClassifier

from thundersvm             import SVC as ThunderSVC
from utils.utils            import review_feature, balance_classes

MODELS = {
    'Logistic'    : LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'SVM'         : ThunderSVC(kernel='linear', probability=True, verbose=True),
    'KNN'         : KNeighborsClassifier(),
    'XGBoost'     : XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


def classification(df: DataFrame, random_state: int, model_name: str):
    print("📐 [INFO] Preparing data...")
    y = df['label']

    df_num = (
        df
        .select_dtypes(include='number')
        .drop(columns=['label', 'useful'], errors='ignore')
        .fillna(0)
    )
    X = df_num.values

    total = len(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    print(f"🔍 [INFO] Samples: total={total}, helpful={pos}, not helpful={neg}")
    print(f"✅ [INFO] Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=random_state
    )

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = MODELS[model_name]
    print(f"🧠 [INFO] Training {model_name}...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("📈 ROC AUC:", roc_auc_score(y_test, y_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classification with ThunderSVM (GPU-accelerated SVM)"
    )
    parser.add_argument(
        "-k", "--top_k", type=int, default=50,
        help="Top-k for one-hot encoding"
    )
    parser.add_argument(
        "-u", "--min", type=float, default=2,
        help="Minimum number of useful"
    )
    parser.add_argument(
        "-m", "--model",
        choices=list(MODELS.keys()), default='SVM',
        help="Which model to use"
    )
    args = parser.parse_args()

    df, _ = review_feature(
        review_fp="../data/review.json",
        business_fp="../data/business.json",
        user_fp="../data/user.json",
        checkin_fp="../data/checkin.json",
        tip_fp="../data/tip.json",
        top_k=args.top_k,
        min_useful=args.min,
    )

    df = balance_classes(df=df, seed=42)
    classification(df=df, random_state=42, model_name=args.model)
