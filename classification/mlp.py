import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import review_feature


class MLP(nn.Module):
    def __init__(self, in_ch, out_ch, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_ch, out_features=256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=p),
            # nn.Linear(in_features=128, out_features=64),
            # nn.BatchNorm1d(64),
            # nn.GELU(),
            # nn.Dropout(p=p),
            # nn.Linear(in_features=64, out_features=32),
            # nn.BatchNorm1d(32),
            # nn.GELU(),
            # nn.Dropout(p=p),
            nn.Linear(in_features=64, out_features=out_ch),
        )
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


def train_mlp(df, epochs=20, batch_size=64, lr=1e-2):
    print("📐 [INFO] Preparing dataset...")
    y = df['label'].values.astype(np.float32)
    df = df.select_dtypes(include='number')
    df = df.drop(columns=['label', 'useful', ], errors='ignore')
    X = df.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Normalize for linear models
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = MLP(in_ch=X.shape[1], out_ch=1, p=0.3)
    weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y)
    pos_weight = torch.tensor([weights[1] / weights[0]])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = SGD(model.parameters(), lr=lr)

    print("🧠 [INFO] Training model...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📈 [INFO] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    print("🧪 [INFO] Evaluating on test set...")
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            probs = model(xb).squeeze()
            preds = (probs >= 0.5).int()
            y_true.extend(yb.squeeze().int().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    print("📊 [INFO] Classification Report:")
    print(classification_report(y_true, y_pred))

    print("📉 [INFO] Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("📈 [INFO] ROC AUC:", roc_auc_score(y_true, y_prob))

    return model


if __name__ == "__main__":
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
        type=int,
        required=True,
        help="Minimum number of useful",
    )
    args = parser.parse_args()
    top_k = args.top_k
    min_useful = args.min

    df, _ = review_feature(
        review_fp="data/review.json",
        business_fp="data/business.json",
        user_fp="data/user.json",
        checkin_fp="data/checkin.json",
        tip_fp="data/tip.json",
        top_k=top_k,
        min_useful=min_useful,
    )

    _ = train_mlp(df=df, epochs=20, batch_size=1024, lr=5e-1)
