from pandas import DataFrame
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix
from utils.utils import review_feature, balance_classes


MODELS = {
    'Logistic':    LogisticRegression(max_iter=1000),
    'RandomForest':RandomForestClassifier(),
    'SVM':         SVC(kernel='linear', probability=True),
    'KNN':         KNeighborsClassifier(),
    'XGBoost':     XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


def classification(df: DataFrame, random_state: int, model: str):
    # 1) 标签 + 原始数值特征
    y = df['label']
    df_num = (
        df
        .select_dtypes(include='number')
        .drop(columns=[
            'label',
            'useful'], errors='ignore')
        .fillna(0)
    )
    texts = df['text'].fillna("")

    # 2) 划分 train/test（保持 text 对齐）
    X_full = df_num.copy()
    X_full['text'] = texts
    X_train_full, X_test_full, y_train, y_test = \
        train_test_split(X_full, y, stratify=y, random_state=random_state)

    X_train_num = X_train_full.drop(columns=['text']).values
    X_test_num  = X_test_full .drop(columns=['text']).values
    X_train_txt = X_train_full['text']
    X_test_txt  = X_test_full ['text']

    # 3) 全量 2-gram TF-IDF
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 5))
    X_train_tfidf = vec.fit_transform(X_train_txt)
    X_test_tfidf  = vec.transform(X_test_txt)

    # 4) 计算有用/非有用类的质心
    cent_useful     = np.asarray(X_train_tfidf[y_train==1].mean(axis=0)).ravel()[None,:]
    cent_not_useful = np.asarray(X_train_tfidf[y_train==0].mean(axis=0)).ravel()[None,:]

    # 5) 计算余弦相似度
    sim_tr_useful = cosine_similarity(X_train_tfidf,     cent_useful)
    sim_tr_not    = cosine_similarity(X_train_tfidf,     cent_not_useful)
    sim_te_useful = cosine_similarity(X_test_tfidf,      cent_useful)
    sim_te_not    = cosine_similarity(X_test_tfidf,      cent_not_useful)

    # — DEBUG: 保存一下 sim 特征 + 标签 —
    pd.DataFrame({
        'sim_useful':   sim_tr_useful.ravel(),
        'sim_notuseful':sim_tr_not.ravel(),
        'label':        y_train.values
    }).to_csv("sim_train_debug.csv", index=False)
    print("🔧 [DEBUG] Saved sim_train_debug.csv")

    # 6) 合并原始数值特征 + sim 特征
    X_train_combined = np.hstack([X_train_num, sim_tr_useful, sim_tr_not])
    X_test_combined  = np.hstack([X_test_num,  sim_te_useful, sim_te_not])

    # 7) 统一归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_combined)
    X_test  = scaler.transform(X_test_combined)

    print(f"✅ [INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 8) 训练并评估
    clf = MODELS[model]
    print(f"🧠 [INFO] Training {model}…")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("📈 ROC AUC:", roc_auc_score(y_test, y_prob))

    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k","--top_k",type=int,default=50,help="(unused)")
    parser.add_argument("-u","--min",  type=int,default=2, help="min useful")
    parser.add_argument("-m","--model",choices=list(MODELS.keys()),
                        default='RandomForest', help="which model")
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
    classification(df=df, random_state=42, model=args.model)
