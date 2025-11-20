"""
Advanced Models: CNN, CatBoost for Drought Prediction
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class CNN1DRegressor(nn.Module):
    """1D CNN for time series regression"""
    
    def __init__(self, input_steps, input_features=1):
        super(CNN1DRegressor, self).__init__()
        
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        
        # Calculate flattened size
        conv_output_size = (input_steps // 4) * 32
        
        self.fc1 = nn.Linear(conv_output_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_cnn_model(X_train, y_train, X_val, y_val, batch_size=64, epochs=10):
    """Train CNN model - returns model, history, and predictions"""
    
    # Build model
    input_steps = X_train.shape[1]
    input_features = X_train.shape[2] if len(X_train.shape) > 2 else 1
    model = CNN1DRegressor(input_steps, input_features)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(-1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(-1))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        
        train_loss = total / len(train_loader.dataset)
        
        # Validation
        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total += loss.item() * xb.size(0)
        
        val_loss = total / len(val_loader.dataset)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
    
    # Generate predictions on validation set
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.from_numpy(X_val).to(device)
        y_pred_val = model(X_val_tensor).cpu().numpy().flatten()
    
    class H:
        def __init__(self, hist):
            self.history = hist
    
    return model, H(history), y_pred_val


def build_cnn_model(input_steps, input_features=1):
    """Build CNN model"""
    return CNN1DRegressor(input_steps, input_features)


def train_catboost_model(X_train, y_train, X_val, y_val, iterations=100):
    """Train CatBoost model - returns model and predictions"""
    try:
        from catboost import CatBoostRegressor
        
        # Flatten sequences for CatBoost
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            verbose=False,
            random_seed=42
        )
        
        model.fit(
            X_train_flat, y_train,
            eval_set=(X_val_flat, y_val),
            use_best_model=True,
            verbose=False
        )
        
        # Generate predictions
        y_pred_val = model.predict(X_val_flat)
        
        return model, y_pred_val
    
    except ImportError:
        return None, None, None
