# classification_text.py

from pandas import DataFrame
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np

from sklearn.model_selection        import train_test_split
from sklearn.preprocessing          import StandardScaler
from sklearn.metrics.pairwise       import cosine_similarity
from sklearn.metrics                import classification_report, confusion_matrix, roc_auc_score

from sklearn.ensemble               import RandomForestClassifier
from sklearn.linear_model           import LogisticRegression
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.svm                    import SVC
from xgboost                        import XGBClassifier

from sentence_transformers          import SentenceTransformer

from utils.utils import review_feature

# pick a fast small model for encoding
EMBED_MODEL = 'all-MiniLM-L6-v2'

MODELS = {
    'Logistic':     LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'SVM':          SVC(kernel='linear', probability=True),
    'KNN':          KNeighborsClassifier(),
    'XGBoost':      XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


def classification(df: DataFrame, random_state: int, model_name: str):
    # 1) prepare labels + numeric features
    y = df['label']
    df_num = (
        df
        .select_dtypes(include='number')
        .drop(columns=[
            'label',
            'useful','useful_user',
            'funny','funny_user',
            'cool','cool_user'
        ], errors='ignore')
        .fillna(0)
    )
    texts = df['text'].fillna("")

    # 2) split train/test, keeping text aligned
    X_full = df_num.copy()
    X_full['text'] = texts
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, stratify=y, random_state=random_state
    )
    X_tr_num = X_tr.drop(columns=['text']).values
    X_te_num = X_te.drop(columns=['text']).values
    X_tr_txt = X_tr['text'].tolist()
    X_te_txt = X_te['text'].tolist()

    # 3) load and run encoder
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    encoder = SentenceTransformer(EMBED_MODEL, device=device)

    emb_tr = encoder.encode(X_tr_txt, convert_to_numpy=True, show_progress_bar=True)
    emb_te = encoder.encode(X_te_txt, convert_to_numpy=True, show_progress_bar=True)

    # 4) L2-normalize embeddings
    from sklearn.preprocessing import normalize
    emb_tr = normalize(emb_tr, norm='l2', axis=1)
    emb_te = normalize(emb_te, norm='l2', axis=1)

    # 5) compute class centroids
    cent_useful     = emb_tr[y_tr == 1].mean(axis=0, keepdims=True)
    cent_notuseful  = emb_tr[y_tr == 0].mean(axis=0, keepdims=True)

    # 6) cosine sims
    sim_tr_useful = cosine_similarity(emb_tr,    cent_useful).ravel()
    sim_tr_not    = cosine_similarity(emb_tr,    cent_notuseful).ravel()
    sim_te_useful = cosine_similarity(emb_te,     cent_useful).ravel()
    sim_te_not    = cosine_similarity(emb_te,     cent_notuseful).ravel()

    # 7) debug: save train sims + label, and test sims + label
    pd.DataFrame({
        'sim_useful':    sim_tr_useful,
        'sim_notuseful': sim_tr_not,
        'label':         y_tr.values
    }).to_csv("sim_train_embedding_debug.csv", index=False)
    print("🔧 [DEBUG] Saved training sims to sim_train_debug.csv")

    pd.DataFrame({
        'sim_useful':    sim_te_useful,
        'sim_notuseful': sim_te_not,
        'label':         y_te.values
    }).to_csv("sim_test_embedding_debug.csv", index=False)
    print("🔧 [DEBUG] Saved test sims to sim_test_debug.csv")

    # 8) combine numeric + sim features
    X_train_combined = np.hstack([X_tr_num, sim_tr_useful[:, None], sim_tr_not[:, None]])
    X_test_combined  = np.hstack([X_te_num, sim_te_useful[:, None], sim_te_not[:, None]])

    # 9) scale all features together
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_combined)
    X_test  = scaler.transform(X_test_combined)

    print(f"✅ [INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 10) train & evaluate
    clf = MODELS[model_name]
    print(f"🧠 [INFO] Training {model_name}…")
    clf.fit(X_train, y_tr)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("📊 Classification Report:")
    print(classification_report(y_te, y_pred))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))
    print("📈 ROC AUC:", roc_auc_score(y_te, y_prob))

    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--min",
        type=int,
        default=2,
        help="minimum 'useful' threshold",
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=50,
        help="(unused) top_k for TF-IDF",
    )
    parser.add_argument(
        "-m", "--model",
        choices=list(MODELS.keys()),
        default='Logistic',
        help="which model to train",
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

    classification(df=df, random_state=42, model_name=args.model)
