import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.utils import compute_class_weight
from utils.utils import review_feature, balance_classes


class MLP(nn.Module):
    def __init__(self, in_ch, out_ch, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_ch, out_features=128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(in_features=32, out_features=out_ch),
        )
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        return self.net(x)


def train_mlp(df, epochs=20, batch_size=64, lr=1e-2):
    # 1) pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 [INFO] Using device: {device}")

    # 2) Prepare data
    print("📐 [INFO] Preparing dataset...")
    y = df['label'].values.astype(np.float32)
    df_num = df.select_dtypes(include='number') \
               .drop(columns=['label', 'useful'], errors='ignore')
    X = df_num.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1)
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).unsqueeze(1)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # 3) Build model & move to device
    model = MLP(in_ch=X.shape[1], out_ch=1, p=0.1).to(device)

    # 4) Compute class weights for imbalance
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y
    )
    pos_weight = torch.tensor([weights[1] / weights[0]], dtype=torch.float, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = MLP(in_ch=X.shape[1], out_ch=1, p=0.3)
    # uncomment if class balance is needed
    # weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y)
    # pos_weight = torch.tensor([weights[1] / weights[0]])
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=None)
    optimizer = SGD(model.parameters(), lr=lr)

    # 5) Training loop
    print("🧠 [INFO] Training model...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        if epoch % 256 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 5
            print(f"🔧 [INFO] Epoch {epoch} — Reduced LR to {optimizer.param_groups[0]['lr']:.6f}")
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"📈 [INFO] Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

    # 6) Evaluation
    print("🧪 [INFO] Evaluating on test set...")
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb).squeeze(1)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).int()

            y_true.extend(yb.cpu().int().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    print("📊 [INFO] Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
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
    df = balance_classes(df=df, seed=42)
    # call with your chosen hyperparameters
    _ = train_mlp(
        df=df,
        epochs=1024,
        batch_size=1024,
        lr=0.4
    )
