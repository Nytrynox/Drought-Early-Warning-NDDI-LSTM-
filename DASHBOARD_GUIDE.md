# AI-Powered Drought Early Warning System
## Integrated Multi-Model Dashboard with Real-time Satellite Data

### 🎯 Overview
This enhanced dashboard combines:
- **LSTM Deep Learning** for time series prediction
- **Random Forest** ensemble learning
- **Support Vector Regression (SVR)** for pattern recognition
- **Real-time NDDI satellite data** fetching
- **Interactive visualizations** and model comparisons
- **Agriculture dataset integration**

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

#### 1. Clone/Navigate to Project
```bash
cd /Users/karthik/Sync/All\ Projects/Drought-Early-Warning-NDDI-LSTM
```

#### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: The agriculture_dataset.csv file is too large to load in the editor. The dashboard will handle it dynamically.

#### 4. Run the Application
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### 🤖 AI Dashboard Tab
The main integrated dashboard with:

#### Data Sources:
1. **Upload CSV** - Upload your own agricultural data
2. **Real-time Satellite** - Fetch NDDI data from satellite imagery
3. **Use Agriculture Dataset** - Load the agriculture_dataset.csv
4. **Session Data** - Use preprocessed data from other tabs

#### Models Trained:
- **LSTM (Long Short-Term Memory)** - Deep learning for sequential patterns
- **Random Forest** - Ensemble decision trees
- **SVR (Support Vector Regression)** - Kernel-based regression

#### Visualizations:
- **Predictions vs Actual** - Time series comparison
- **Regression Plots** - Scatter plots with R² scores
- **Model Comparison** - Combined performance analysis
- **LSTM Training History** - Loss curves

#### Metrics Displayed:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score (Coefficient of Determination)

### 📊 Data Tab
- Upload CSV files
- Load from URL
- Download from Kaggle
- Generate synthetic demo data

### 🔧 Preprocess Tab
- Calculate NDDI from NDVI and NDWI
- Handle missing values
- Date/time processing
- Data normalization

### 🧠 Model Tab
- LSTM training with configurable parameters
- Monte Carlo Dropout for uncertainty estimation
- Probabilistic drought forecasting
- Alert threshold configuration

### 📈 Visualize Tab
- Time series plots
- Forecast visualization
- Uncertainty bands
- Drought severity indicators

---

## 🛰️ Real-time Satellite Data Integration

### Supported Data Sources

#### 1. **Google Earth Engine** (Recommended)
```bash
# Install Earth Engine
pip install earthengine-api

# Authenticate
earthengine authenticate
```

**Features**:
- Sentinel-2 imagery
- MODIS data
- Landsat collections
- Cloud filtering
- Automatic NDVI/NDWI calculation

#### 2. **Sentinel Hub API**
```bash
pip install sentinelsat
```

**Usage**:
- Requires Copernicus account
- High-resolution imagery
- Custom band math

#### 3. **NASA MODIS**
- Direct API access
- 16-day composite data
- Global coverage

#### 4. **Simulated Data** (Default)
- No authentication required
- Realistic patterns
- Great for testing

### Configuration

The satellite fetcher automatically detects available APIs:

```python
from src.drought_app.utils.satellite import SatelliteDataFetcher

fetcher = SatelliteDataFetcher()
sources = fetcher.get_available_sources()
print(sources)
# {
#     'Google Earth Engine': True/False,
#     'Sentinel Hub': True/False,
#     'Simulated Data': True
# }
```

---

## 📁 Project Structure

```
Drought-Early-Warning-NDDI-LSTM/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── agriculture_dataset.csv         # Your dataset (large file)
├── AI.ipynb                       # Colab notebook code
├── src/
│   └── drought_app/
│       ├── ui.py                  # Main UI coordinator
│       ├── core/
│       │   ├── data.py           # Data loading utilities
│       │   ├── model.py          # LSTM model definitions
│       │   └── session.py        # Session state management
│       ├── views/
│       │   ├── ai_dashboard.py   # 🆕 AI multi-model dashboard
│       │   ├── data_ingestion.py # Data loading interface
│       │   ├── preprocessing.py  # Data preprocessing
│       │   ├── modeling.py       # Model training
│       │   └── visualization.py  # Results visualization
│       ├── utils/
│       │   ├── nddi.py          # NDDI calculations
│       │   ├── satellite.py     # 🆕 Satellite data fetching
│       │   └── sample.py        # Sample data generation
│       └── components/
│           └── map_viewer.py    # 🆕 Interactive maps
└── docs/                         # Documentation
```

---

## 🔧 Configuration Options

### Model Hyperparameters

**LSTM**:
- Lookback window: 5-30 steps (default: 10)
- Epochs: 2-20 (default: 4 for speed)
- Hidden units: 32-128
- Dropout: 0.0-0.8

**Random Forest**:
- n_estimators: 100 (default)
- Automatic feature selection
- Parallel processing enabled

**SVR**:
- Kernel: RBF (default)
- Auto gamma selection
- Sample limiting for large datasets

### Satellite Data

**Fetch Settings**:
- Latitude: -90 to 90
- Longitude: -180 to 180
- Days of history: 30-365
- Cloud cover threshold: < 20%

---

## 📊 Using Your Agriculture Dataset

The `agriculture_dataset.csv` file is automatically detected by the dashboard:

### Option 1: Through AI Dashboard
1. Go to "🤖 AI Dashboard" tab
2. Select "Use Agriculture Dataset" in sidebar
3. Click "Train All Models"

### Option 2: Through Data Tab
1. Go to "📊 Data" tab
2. Upload the CSV manually
3. Process in other tabs

### Data Requirements
Your CSV should contain numeric columns. The dashboard will:
- Auto-detect numeric columns
- Handle missing values
- Scale features automatically
- Create time sequences for prediction

---

## 🎨 Example Workflows

### Workflow 1: Quick Demo with Simulated Data
```
1. Open app → AI Dashboard tab
2. Sidebar: Select "Real-time Satellite"
3. Set coordinates (e.g., 28.6139, 77.2090 for Delhi)
4. Click "Fetch Satellite Data"
5. Click "Train All Models"
6. View results in all visualization tabs
```

### Workflow 2: Upload Custom Data
```
1. AI Dashboard tab
2. Sidebar: Select "Upload CSV"
3. Upload your agricultural data
4. Select target column (e.g., NDDI, Yield, Moisture)
5. Click "Train All Models"
6. Download predictions as CSV
```

### Workflow 3: Full Pipeline with NDDI Calculation
```
1. Data tab → Upload CSV with NDVI and NDWI columns
2. Preprocess tab → Calculate NDDI
3. Model tab → Train LSTM
4. Visualize tab → View forecasts
5. AI Dashboard tab → Compare all models
```

---

## 📈 Interpreting Results

### NDDI Values
- **< -0.5**: Extreme drought ⚠️
- **-0.5 to -0.2**: Severe drought 🔴
- **-0.2 to 0.0**: Moderate drought 🟠
- **0.0 to 0.2**: Normal conditions 🟢
- **> 0.2**: Wet conditions 🔵

### Model Metrics
- **RMSE**: Lower is better (typical: 0.01-0.1 for scaled data)
- **MAE**: Average prediction error (lower is better)
- **R² Score**: 
  - 1.0 = Perfect fit
  - 0.8-1.0 = Excellent
  - 0.6-0.8 = Good
  - < 0.6 = Needs improvement

### Which Model to Use?
- **LSTM**: Best for sequential patterns, seasonal trends
- **Random Forest**: Best for complex non-linear relationships
- **SVR**: Best for smooth, continuous predictions

Compare all three and ensemble for best results!

---

## 🐛 Troubleshooting

### TensorFlow Not Available
```bash
# macOS with Apple Silicon
pip install tensorflow-macos tensorflow-metal

# Other systems
pip install tensorflow
```

### Large Dataset Slow Training
- Reduce lookback window (10 → 5)
- Reduce epochs (20 → 4)
- Enable simple mode in Model tab
- Dataset is automatically capped at 3000 rows

### Satellite Data Not Loading
- Check internet connection
- Verify API credentials (for GEE/Sentinel)
- Use "Simulated Data" as fallback

### Agriculture CSV Not Found
```bash
# Ensure file is in project root
ls -lh agriculture_dataset.csv

# If too large, upload through UI instead
```

---

## 🔐 API Keys & Authentication

### Google Earth Engine
```bash
earthengine authenticate
```
Visit: https://signup.earthengine.google.com/

### Sentinel Hub / Copernicus
Visit: https://scihub.copernicus.eu/

### NASA MODIS
Visit: https://urs.earthdata.nasa.gov/

*Note*: All APIs have free tiers for research use

---

## 📦 Export & Integration

### Download Predictions
1. Train models in AI Dashboard
2. Scroll to "Export Results" section
3. Click "Download Predictions as CSV"

Output format:
```csv
Actual,LSTM_Predicted,Random Forest_Predicted,SVR_Predicted
0.234,0.241,0.238,0.236
...
```

### API Integration
The models can be saved and deployed:
```python
# Save trained models
import pickle
pickle.dump(model_lstm, open('lstm_model.pkl', 'wb'))
pickle.dump(rf, open('rf_model.pkl', 'wb'))
pickle.dump(svr, open('svr_model.pkl', 'wb'))
```

---

## 🚀 Advanced Features

### Custom Satellite Integration
Edit `src/drought_app/utils/satellite.py`:
```python
def fetch_custom_api(lat, lon, start, end):
    # Your API integration
    response = requests.get(YOUR_API_URL)
    return process_response(response)
```

### Add New Models
Edit `src/drought_app/views/ai_dashboard.py`:
```python
# Add new model in train_and_compare_models()
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train_flat, y_train)
y_pred_ridge = ridge.predict(X_test_flat)
```

---

## 📞 What You Need to Provide

### For Full Satellite Integration:
1. **Google Earth Engine account** (free for research)
   - Sign up at: https://earthengine.google.com/
   - Run: `earthengine authenticate`

2. **Copernicus account** (optional, for Sentinel Hub)
   - Sign up at: https://scihub.copernicus.eu/

3. **NASA Earthdata account** (optional, for MODIS)
   - Sign up at: https://urs.earthdata.nasa.gov/

### For Agriculture Dataset:
- The `agriculture_dataset.csv` file is already in your project
- Just select "Use Agriculture Dataset" in the dashboard

### System Requirements:
- ✅ Already installed: Python, VS Code
- ✅ Need to install: Dependencies via `pip install -r requirements.txt`
- ⚡ Recommended: 8GB+ RAM for large datasets

---

## 🎯 Next Steps

1. **Install dependencies**:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. **Test with simulated data** first (no API keys needed)

4. **Upload your agriculture dataset** and train models

5. **Set up satellite APIs** for real-time data (optional)

---

## 📚 References

- **NDDI Paper**: Normalized Difference Drought Index
- **LSTM**: Hochreiter & Schmidhuber, 1997
- **Random Forest**: Breiman, 2001
- **Google Earth Engine**: https://earthengine.google.com/
- **Sentinel-2**: https://sentinel.esa.int/

---

## 💡 Tips

- Start with **Simple Mode** for faster experimentation
- Use **Simulated Data** to understand the dashboard
- **Compare all models** - ensemble often performs best
- **Download results** for further analysis in Excel/Python
- **Monitor R² scores** - aim for > 0.7 for good predictions

---

**Dashboard is ready to use! Let me know if you need help with:**
- API authentication
- Custom model integration
- Data preprocessing
- Performance optimization
