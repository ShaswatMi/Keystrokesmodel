
# proctor_transformer_train.py
import math
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
SEED = 42
torch.manual_seed(SEED);
np.random.seed(SEED)
random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "proctor_transformer.pt"
# ONNX_PATH = "proctor_transformer.onnx"
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-4
SEQ_LEN = 120          # sequence length in timesteps (e.g., 120 sec with 1s sampling)
FEATURE_DIM = 6        # tabSwitch, focusLost, copy, paste, cut, fullScreenExit
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 4
D_FF = 256
DROPOUT = 0.2
PATIENCE = 5           # early stopping patience


# ----------------------------
# Synthetic dataset generator
# ----------------------------
class ProctoringSequenceDataset(Dataset):
    """
    Generates synthetic time-series sequences of shape (seq_len, 6).
    Label=1 indicates cheating-like behavior pattern present.
    Label=0 indicates benign behavior (small random noise).
    """

    def __init__(self, n_samples: int, seq_len: int = SEQ_LEN, fraud_prob: float = 0.5):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.fraud_prob = fraud_prob

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = 1 if random.random() < self.fraud_prob else 0
        x = self._generate_sequence(label)
        # Normalize features roughly (per-synthetic-design)
        x = self._normalize(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _generate_sequence(self, label: int):
        seq = np.zeros((self.seq_len, FEATURE_DIM), dtype=float)

        # Baseline benign noise
        # tabSwitch : rare spikes (0/1)
        # focusLostSeconds : small random drift per timestep
        # copy/paste/cut : rare events
        # fullScreenExit : rare spikes
        for t in range(self.seq_len):
            # benign: mostly zeros with occasional tiny noise
            seq[t, 0] = np.random.binomial(1, 0.01)  # tabSwitch event
            seq[t, 1] = np.random.binomial(1, 0.02) * np.random.uniform(1, 3)  # focus lost seconds chunk
            seq[t, 2] = np.random.binomial(1, 0.005)  # copy
            seq[t, 3] = np.random.binomial(1, 0.005)  # paste
            seq[t, 4] = np.random.binomial(1, 0.002)  # cut
            seq[t, 5] = np.random.binomial(1, 0.003)  # fullscreen exit

        if label == 1:
            # Inject cheating patterns:
            # - bursts: copy -> tabSwitch -> paste within short window
            # - repeated focus loss bursts
            # - frequent fullscreen exits
            num_bursts = random.randint(1, 4)
            for _ in range(num_bursts):
                center = random.randint(5, self.seq_len - 6)
                # burst length 3-8
                blen = random.randint(3, 8)
                for i in range(center, min(center + blen, self.seq_len)):
                    # spike probability much higher within burst
                    seq[i, 0] = seq[i, 0] or np.random.binomial(1, 0.6)  # tabSwitch
                    seq[i, 1] += np.random.uniform(2, 7)                 # focusLostSeconds
                    # copy->paste pattern
                    if random.random() < 0.6:
                        seq[i, 2] = 1 if random.random() < 0.6 else seq[i, 2]
                        if i + 1 < self.seq_len:
                            seq[i + 1, 3] = 1
                    # fullscreen exit bursts
                    if random.random() < 0.4:
                        seq[i, 5] = 1

            # occasional long focus loss
            if random.random() < 0.6:
                start = random.randint(10, self.seq_len - 20)
                length = random.randint(5, 20)
                seq[start:start + length, 1] += np.random.uniform(5, 20)

        return seq

    def _normalize(self, x):
        # Basic normalization tuned to synthetic ranges:
        # - tabSwitch, copy, paste, cut, fullscreen : keep 0/1
        # - focusLostSeconds: scale to ~0..1
        x[:, 1] = np.clip(x[:, 1] / 20.0, 0.0, 1.0)  # scale focus lost
        return x


# ----------------------------
# Positional encoding
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


# ----------------------------
# Transformer encoder model
# ----------------------------
# ----------------------------------------------
# LITE TRANSFORMER (fast training, high accuracy)
# ----------------------------------------------
class ProctorTransformer(nn.Module):
    def __init__(self,
                 feature_dim=6,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=128,
                 dropout=0.1,
                 max_len=1000):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)

        # lightweight positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        # only 2 stacked layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, 6)
        """
        x = self.input_proj(x)            # (B, L, 64)
        x = self.pos_encoder(x)           # (B, L, 64)
        x = self.transformer_encoder(x)   # (B, L, 64)

        # pooling over time dimension
        x = x.permute(0, 2, 1)            # (B, 64, L)
        pooled = self.pool(x).squeeze(-1) # (B, 64)

        logits = self.classifier(pooled)  # (B, 1)
        return logits



# ----------------------------
# Utilities
# ----------------------------
def collate_fn(batch):
    # batch: list of (tensor(seq_len, feat), label)
    seqs, labels = zip(*batch)
    seqs = torch.stack(seqs)  # require same seq_len; our synthetic data uses fixed SEQ_LEN
    labels = torch.tensor(labels, dtype=torch.float32)
    return seqs, labels


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }
    return metrics


# ----------------------------
# Training & validation
# ----------------------------
def train_and_evaluate():
    # dataset
    dataset = ProctoringSequenceDataset(n_samples=8000, seq_len=SEQ_LEN, fraud_prob=0.5)
    n_val = int(0.15 * len(dataset))
    n_test = int(0.15 * len(dataset))
    n_train = len(dataset) - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ProctorTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0.0
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

        # validation
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(DEVICE)
                logits = model(x_val)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs.tolist())
                all_labels.extend(y_val.numpy().tolist())

        val_metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
        print(f"Validation metrics epoch {epoch}: {val_metrics}")

        # early stopping based on AUC
        if val_metrics["auc"] > best_val_auc + 1e-4:
            best_val_auc = val_metrics["auc"]
            patience_left = PATIENCE
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch}, MODEL_PATH)
            print(f"Saved best model (AUC={best_val_auc:.4f}) to {MODEL_PATH}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # load best model and evaluate on test set
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(DEVICE)
            logits = model(x_test)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_test.numpy().tolist())

    test_metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    print("Final test metrics:", test_metrics)

    return model


# ----------------------------
# Inference helper
# ----------------------------
def risk_score_from_sequence(model: nn.Module, sequence: np.ndarray) -> float:
    """
    sequence: np.ndarray shape (seq_len, 6)
    returns: risk score 0..100
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy().item()
    return round(float(prob) * 100.0, 2)


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    model = train_and_evaluate()

    # test example sequences
    ds = ProctoringSequenceDataset(n_samples=2, seq_len=SEQ_LEN, fraud_prob=0.5)
    seq0, label0 = ds[0]
    seq1, label1 = ds[1]

    print("Sample labels:", label0.item(), label1.item())
    print("Risk seq0:", risk_score_from_sequence(model, seq0.numpy()))
    print("Risk seq1:", risk_score_from_sequence(model, seq1.numpy()))
