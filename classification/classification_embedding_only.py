# classification_embed_only.py

from pandas import DataFrame
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.svm                import SVC
from xgboost                    import XGBClassifier

from sentence_transformers      import SentenceTransformer
from utils.utils                import review_feature, balance_classes

EMBED_MODEL = 'all-MiniLM-L6-v2'

MODELS = {
    'Logistic'    : LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'SVM'         : SVC(kernel='linear', probability=True),
    'KNN'         : KNeighborsClassifier(),
    'XGBoost'     : XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


def classification_embed_only(
    df: DataFrame,
    random_state: int,
    model_name: str,
    embed_model: str = EMBED_MODEL,
    batch_size: int = 64
):
    # 1) 提取文本和标签
    texts = df['text'].fillna("").tolist()
    y     = df['label']

    # 2) 划分训练/测试
    X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
        texts, y, stratify=y, random_state=random_state
    )

    # 3) 加载编码器（优先用 GPU，如果有的话）
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer(embed_model, device=device)

    # 4) 编码（返回 numpy array）
    emb_tr = encoder.encode(
        X_tr_txt,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    emb_te = encoder.encode(
        X_te_txt,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # 5) （可选）L2 归一化向量
    from sklearn.preprocessing import normalize
    emb_tr = normalize(emb_tr, norm='l2', axis=1)
    emb_te = normalize(emb_te, norm='l2', axis=1)

    # 6) 标准化所有维度，使每个维度分布更均匀
    scaler = StandardScaler()
    emb_tr_s = scaler.fit_transform(emb_tr)
    emb_te_s = scaler.transform(emb_te)

    print(f"✅ [INFO] Embedding features shape: train={emb_tr_s.shape}, test={emb_te_s.shape}")

    # 7) 训练 & 评估
    clf = MODELS[model_name]
    print(f"🧠 [INFO] Training {model_name} on {embed_model} embeddings…")
    clf.fit(emb_tr_s, y_tr)

    # 训练集性能
    y_tr_pred = clf.predict(emb_tr_s)
    print(f"🧪 [TRAIN] Acc={accuracy_score(y_tr, y_tr_pred):.4f}")
    print(classification_report(y_tr, y_tr_pred, digits=4))

    # 测试集性能
    y_te_pred = clf.predict(emb_te_s)
    y_te_prob = clf.predict_proba(emb_te_s)[:, 1]
    print(f"🧪 [TEST ] Acc={accuracy_score(y_te, y_te_pred):.4f}")
    print(classification_report(y_te, y_te_pred, digits=4))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_te, y_te_pred))
    print("📈 ROC AUC:", roc_auc_score(y_te, y_te_prob))

    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="High-dimensional embedding only classification"
    )
    parser.add_argument("-u", "--min",    type=int, default=2,
                        help="minimum useful threshold")
    parser.add_argument("-k", "--top_k",  type=int, default=50,
                        help="(unused) top_k for review_feature")
    parser.add_argument("-m", "--model",  choices=list(MODELS.keys()),
                        default='RandomForest', help="classifier")
    parser.add_argument("-e", "--embed_model",
                        type=str, default=EMBED_MODEL,
                        help="sentence-transformers model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=64,
                        help="encoding batch size")
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

    classification_embed_only(
        df=df,
        random_state=42,
        model_name=args.model,
        embed_model=args.embed_model,
        batch_size=args.batch_size
    )
