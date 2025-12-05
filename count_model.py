import torch
import torch.nn as nn
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
class CountModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------
# STABLE LABEL GENERATOR
# ---------------------------------------------------
def generate_stable_data(n=6000):
    X = np.random.randint(0, 30, size=(n, 6)).astype(np.float32)

    tab = X[:,0]
    focus = X[:,1]
    copy = X[:,2]
    paste = X[:,3]
    cut = X[:,4]
    fs = X[:,5]

    # stable risk formula (0–1 range)
    risk = (
        0.4 * np.minimum(tab / 12, 1) +
        0.2 * np.minimum(copy / 5, 1) +
        0.2 * np.minimum(paste / 5, 1) +
        0.2 * np.minimum(fs / 3, 1)
    )

    # threshold to make labels 0 or 1
    y = (risk > 0.5).astype(np.float32)

    return X, y

# ---------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------
def train_count_model():
    model = CountModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # and now training is stable
    X, y = generate_stable_data(6000)

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    print("Training Stable Count Model...")
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/15  Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "count_model.pt")
    print("Saved as count_model_stable.pt")

    return model

# ---------------------------------------------------
# INFERENCE
# ---------------------------------------------------
def risk_from_counts(model, counts):
    x = torch.tensor([counts], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy().item()
    return round(prob * 100, 2)

# ---------------------------------------------------
# TEST
# ---------------------------------------------------
if __name__ == "__main__":
    model = train_count_model()
    test_counts = [8, 30, 2, 1, 0, 3]
    score = risk_from_counts(model, test_counts)
    print("Risk score:", score)
