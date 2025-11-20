"""
Agriculture Dataset Loader and Processor
Handles the large agriculture dataset with satellite imagery features
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Removed RandomForest and SVR - using only CatBoost, CNN, and LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Any
import streamlit as st

# We intentionally do NOT import TensorFlow here anymore.
# All sequence modelling is done via PyTorch (see LSTMRegressor in core.model).
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class AgricultureDataProcessor:
    """Process the agriculture dataset for drought prediction"""
    
    def __init__(self, csv_path: str = 'agriculture_dataset.csv'):
        self.csv_path = csv_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = 'Crop_Stress_Indicator'
        
    @st.cache_data
    def load_data(_self, sample_size: int = 10000) -> pd.DataFrame:
        """
        Load and sample the agriculture dataset
        
        Args:
            sample_size: Number of samples to load (for performance)
        """
        try:
            # Load a sample of the data for performance
            total_lines = 212020  # We know from wc -l
            if sample_size >= total_lines:
                # Load all data
                df = pd.read_csv(_self.csv_path)
            else:
                # Sample random rows
                skip_rows = np.random.choice(range(1, total_lines), 
                                           size=total_lines - sample_size - 1, 
                                           replace=False)
                df = pd.read_csv(_self.csv_path, skiprows=skip_rows)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            
            # Convert categorical columns
            if 'Crop_Type' in df.columns:
                _self.label_encoder = LabelEncoder()
                df['Crop_Type_Encoded'] = _self.label_encoder.fit_transform(df['Crop_Type'])
            
            _self.df = df
            return df
            
        except Exception as e:
            st.error(f"Error loading agriculture data: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for ML models based on the Colab notebook approach
        
        Returns:
            X: Feature matrix
            y: Target vector (Crop_Stress_Indicator)
        """
        # Select relevant features for drought/stress prediction
        feature_cols = [
            'NDVI', 'SAVI', 'Chlorophyll_Content', 'Leaf_Area_Index',
            'Temperature', 'Humidity', 'Rainfall', 'Soil_Moisture',
            'Soil_pH', 'Organic_Matter', 'Canopy_Coverage',
            'Elevation_Data', 'Wind_Speed', 'Expected_Yield'
        ]
        
        # Add encoded crop type if available
        if 'Crop_Type_Encoded' in df.columns:
            feature_cols.append('Crop_Type_Encoded')
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        # Prepare features and target
        X = df[available_cols].fillna(df[available_cols].mean())
        y = df[self.target_column].fillna(df[self.target_column].mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def create_sequences_for_lstm(self, X: np.ndarray, y: np.ndarray, 
                                lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training (from Colab notebook)
        
        Args:
            X: Feature matrix
            y: Target vector
            lookback: Number of time steps to look back
        """
        if len(X) < lookback:
            # If not enough data, repeat the pattern
            n_repeats = (lookback // len(X)) + 1
            X = np.tile(X, (n_repeats, 1))
            y = np.tile(y, n_repeats)
        
        X_seq, y_seq = [], []
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i+lookback])
            y_seq.append(y[i+lookback])
        
        return np.array(X_seq), np.array(y_seq)


class DroughtPredictionModels:
    """Implementation of ML models from the Colab notebook"""
    
    def __init__(self):
        self.lstm_model = None
        # Removed rf_model - using CatBoost, CNN, and LSTM only
        # Using CatBoost, CNN, and LSTM only
        self.models_trained = False
        
    def build_lstm_model(self, input_shape: tuple):
        """
        Build LSTM model using PyTorch (avoids TensorFlow mutex issues)
        
        Args:
            input_shape: (timesteps, features)
        """
        # Import PyTorch LSTM from the existing model.py
        from ..core.model import LSTMRegressor
        
        # Create PyTorch LSTM model
        timesteps, n_features = input_shape
        model = LSTMRegressor(input_size=n_features, hidden_size=64, dropout=0.2)
        
        return model
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train all three models from the Colab notebook
        
        Returns:
            Dictionary with training results and metrics
        """
        # Train-test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        results = {}
        
        # 1. LSTM Model (using PyTorch to avoid TensorFlow mutex issues)
        try:
            import torch
            from ..core.model import train_model as train_pytorch_model
            
            processor = AgricultureDataProcessor()
            X_seq, y_seq = processor.create_sequences_for_lstm(X, y, lookback=10)
            
            if len(X_seq) > 0:
                split_seq = int(len(X_seq) * (1 - test_size))
                X_train_seq = X_seq[:split_seq]
                X_test_seq = X_seq[split_seq:]
                y_train_seq = y_seq[:split_seq]
                y_test_seq = y_seq[split_seq:]
                
                # Build PyTorch LSTM
                self.lstm_model = self.build_lstm_model((10, X_train_seq.shape[2]))
                
                # Convert to float32 for compatibility
                X_train_seq = X_train_seq.astype(np.float32)
                X_test_seq = X_test_seq.astype(np.float32)
                y_train_seq = y_train_seq.astype(np.float32)
                y_test_seq = y_test_seq.astype(np.float32)
                
                # Train with PyTorch
                history_dict = train_pytorch_model(
                    self.lstm_model,
                    X_train_seq, y_train_seq,
                    X_test_seq, y_test_seq,
                    batch_size=32,
                    epochs=4
                )
                
                # Get predictions
                device = torch.device("mps" if torch.backends.mps.is_available() else 
                                    ("cuda" if torch.cuda.is_available() else "cpu"))
                self.lstm_model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.from_numpy(X_test_seq).to(device)
                    y_pred_lstm = self.lstm_model(X_test_tensor).cpu().numpy().reshape(-1)
                
                # Metrics
                results['LSTM'] = {
                    'rmse': np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm)),
                    'mae': mean_absolute_error(y_test_seq, y_pred_lstm),
                    'r2': r2_score(y_test_seq, y_pred_lstm),
                    'predictions': y_pred_lstm,
                    'actual': y_test_seq,
                    'history': history_dict
                }
                
        except Exception as e:
            st.warning(f"LSTM training failed: {e}")
            results['LSTM'] = None
        
        # Note: Removed RandomForest and SVR - focusing on CatBoost, CNN, and LSTM only
        
        self.models_trained = True
        return results
    
    def predict_crop_stress(self, features: np.ndarray) -> Dict[str, float]:
        """
        Make predictions using trained models
        
        Args:
            features: Feature vector
        
        Returns:
            Predictions from all models
        """
        predictions = {}
        
        # Removed RF and SVR predictions - using CatBoost, CNN, and LSTM only
        
        if self.lstm_model:
            # For LSTM (PyTorch), we need sequence data
            import torch
            
            seq_features = np.tile(features, (10, 1)).reshape(1, 10, -1).astype(np.float32)
            
            device = torch.device("mps" if torch.backends.mps.is_available() else 
                                ("cuda" if torch.cuda.is_available() else "cpu"))
            
            self.lstm_model.eval()
            with torch.no_grad():
                seq_tensor = torch.from_numpy(seq_features).to(device)
                pred = self.lstm_model(seq_tensor).cpu().numpy()
                predictions['LSTM'] = float(pred[0][0])
        
        return predictions


def calculate_nddi_from_agriculture_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate NDDI from agriculture dataset features
    
    The dataset already has NDVI, so we estimate NDWI from other features
    and calculate NDDI = (NDVI - NDWI) / (NDVI + NDWI)
    """
    df = df.copy()
    
    # Estimate NDWI from soil moisture and water-related features
    # NDWI typically correlates with water content
    if 'Soil_Moisture' in df.columns and 'NDVI' in df.columns:
        # Normalize soil moisture to [0, 1] range
        soil_moisture_norm = (df['Soil_Moisture'] - df['Soil_Moisture'].min()) / \
                           (df['Soil_Moisture'].max() - df['Soil_Moisture'].min())
        
        # Estimate NDWI (typically lower than NDVI for vegetated areas)
        df['NDWI_estimated'] = soil_moisture_norm * 0.5 + np.random.normal(0, 0.05, len(df))
        df['NDWI_estimated'] = np.clip(df['NDWI_estimated'], 0, 1)
        
        # Calculate NDDI
        df['NDDI'] = (df['NDVI'] - df['NDWI_estimated']) / (df['NDVI'] + df['NDWI_estimated'] + 1e-10)
        df['NDDI'] = np.clip(df['NDDI'], -1, 1)
    
    return df


def get_drought_severity_from_nddi(nddi: float) -> str:
    """Classify drought severity based on NDDI value"""
    if nddi < -0.5:
        return "Extreme Drought"
    elif nddi < -0.2:
        return "Severe Drought" 
    elif nddi < 0.0:
        return "Moderate Drought"
    elif nddi < 0.2:
        return "Normal"
    else:
        return "Wet Conditions"


def integrate_satellite_with_agriculture(
    agriculture_df: pd.DataFrame,
    satellite_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Integrate real-time satellite data with agriculture dataset
    
    Args:
        agriculture_df: Agriculture dataset
        satellite_df: Real-time satellite data
        
    Returns:
        Integrated dataset
    """
    # Add satellite NDDI to agriculture data
    if len(satellite_df) > 0:
        latest_satellite = satellite_df.iloc[-1]
        
        agriculture_df = agriculture_df.copy()
        agriculture_df['Satellite_NDDI'] = latest_satellite['NDDI']
        agriculture_df['Satellite_NDVI'] = latest_satellite['NDVI']
        agriculture_df['Satellite_NDWI'] = latest_satellite['NDWI']
        agriculture_df['Satellite_Temp'] = latest_satellite['Temperature_C']
        agriculture_df['Last_Update'] = latest_satellite['Date']
        
        # Update drought severity
        agriculture_df['Drought_Severity'] = agriculture_df['Satellite_NDDI'].apply(
            get_drought_severity_from_nddi
        )
    
    return agriculture_df