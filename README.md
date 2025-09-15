# Drought Early Warning (NDDI + LSTM)

End-to-end, software-based pipeline to detect and forecast agricultural drought using NDDI and an LSTM model, wrapped in a Streamlit GUI.

## Features
- Ingest data via:
  - Kaggle dataset (requires API token)
  - URL (HTTP CSV)
  - File upload (CSV)
- Preprocess: resample, clean, compute NDDI from NDVI/NDWI or preprovided columns
- LSTM forecasting with Monte Carlo dropout for probabilistic outputs
- Visualizations: time series, probability, optional geospatial map (if lat/lon)
- Export results and models

## Quickstart
1. Create a virtual environment and install deps.
2. Obtain Kaggle API credentials if using Kaggle downloads.
3. Run Streamlit.

### Setup (macOS, zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Kaggle credentials (optional)
Place your Kaggle API key at `~/.kaggle/kaggle.json` with permissions `600`.
```
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

### Run the app
```bash
streamlit run app.py
```

## Dataset
Suggested: Crop Health and Environmental Stress Dataset (Kaggle)
https://www.kaggle.com/datasets/datasetengineer/crop-health-and-environmental-stress-dataset

## Notes
- This app assumes a time column and NDVI/NDWI or NDDI columns. You can map columns in the GUI.
- For probabilistic forecasts, MC Dropout is used at inference.
# Drought-Early-Warning-NDDI-LSTM-
