from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SeriesWindow:
    series: np.ndarray
    lookback: int
    horizon: int

    def make_xy(self):
        X, y = [], []
        for i in range(len(self.series) - self.lookback - self.horizon + 1):
            X.append(self.series[i : i + self.lookback])
            y.append(self.series[i + self.lookback + self.horizon - 1])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X = np.expand_dims(X, -1)
        return X, y

    def get_train_val_split(self, val_ratio: float = 0.2):
        X, y = self.make_xy()
        min_windows = 3
        if len(X) < min_windows:
            raise ValueError(
                f"Not enough data to train: windows={len(X)} (<{min_windows}), lookback={self.lookback}, "
                f"horizon={self.horizon}, series_len={len(self.series)}."
            )
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        return X_train, y_train, X_val, y_val

    def get_last_window(self):
        if len(self.series) < self.lookback:
            raise ValueError("Series shorter than lookback window.")
        x = self.series[-self.lookback:]
        x = np.array(x, dtype=np.float32)[None, :, None]
        return x


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, 1)
        out, (h_n, c_n) = self.lstm(x)
        h = h_n[-1]
        h = self.dropout(h)
        y = self.fc(h)
        return y


def build_lstm_model(input_steps: int, units: int = 32, dropout: float = 0.2):
    model = LSTMRegressor(input_size=1, hidden_size=units, dropout=dropout)
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=20):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    
    # Convert to float32 for MPS compatibility (Apple Silicon)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(-1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(-1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_ds)

        model.eval()
        with torch.no_grad():
            total_val = 0.0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total_val += loss.item() * xb.size(0)
        val_loss = total_val / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    class H:
        def __init__(self, hist):
            self.history = hist
    return H(history)


def mc_dropout_predict(model, X, mc_samples: int = 50):
    device = next(model.parameters()).device
    # Convert to float32 for MPS compatibility
    X = X.astype(np.float32)
    x = torch.from_numpy(X).to(device)
    preds = []
    model.train()  # enable dropout
    with torch.no_grad():
        for _ in range(mc_samples):
            y = model(x).squeeze(-1).cpu().numpy()
            preds.append(y)
    preds = np.stack(preds, axis=0)  # (mc_samples, B)
    # For single-item batch, return a 1D vector of MC samples; else return (mc_samples, B)
    if preds.shape[1] == 1:
        return preds[:, 0]
    return preds
