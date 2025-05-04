# classification_gpu_svm.py

from pandas import DataFrame
import os
import sys
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

from utils.utils import review_feature, balance_classes

# Try to import cuML's GPU-accelerated SVM
try:
    from cuml.svm import SVC as cuSVC
    GPU_SVM_AVAILABLE = True
except ImportError:
    cuSVC = None
    GPU_SVM_AVAILABLE = False

# Choose which SVM to use
if GPU_SVM_AVAILABLE:
    print("🔗 [INFO] Using cuML SVM (GPU-accelerated)")
    SVM_MODEL = cuSVC(kernel='linear', max_iter=1000, verbose=True)
else:
    from sklearn.svm import SVC
    print("⚠️ [WARN] cuML not available, falling back to scikit-learn SVM (CPU)")
    SVM_MODEL = SVC(kernel='linear', probability=True, verbose=True)

MODELS = {
    'Logistic'    : LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'SVM'         : SVM_MODEL,
    'KNN'         : KNeighborsClassifier(),
    'XGBoost'     : XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


def classification(df: DataFrame, random_state: int, model_name: str):
    print("📐 [INFO] Creating training data & labels...")
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
    print("🔍 [INFO] Label Distribution:")
    print(f"[INFO] Total samples:   {total}")
    print(f"[INFO] Helpful     [1]: {pos} ({pos/total:.2%})")
    print(f"[INFO] Not helpful [0]: {neg} ({neg/total:.2%})")
    print(f"✅ [INFO] Feature matrix: {X.shape}, Target: {y.shape}")

    print(f"🧠 [INFO] Running model: {model_name}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=random_state
    )

    # Normalize numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = MODELS[model_name]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # cuML SVM does not support predict_proba; use decision_function for ranking
    if model_name == 'SVM' and GPU_SVM_AVAILABLE:
        # decision_function returns distance to hyperplane; can use as score
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("📈 ROC AUC:", roc_auc_score(y_test, y_score))

    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classification with optional GPU-accelerated SVM"
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int, default=50,
        help="Top-k for one-hot encoding"
    )
    parser.add_argument(
        "--min", "-u",
        type=float, default=2,
        help="Minimum number of useful"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default='SVM',
        help="Model to use"
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
