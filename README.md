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

## Deployment

This is a Python Streamlit app. Hosts that serve static sites (like Netlify or Vercel’s static mode) will show 404/NOT_FOUND because there’s no static build output. Use a Python-friendly host:

- Streamlit Community Cloud: connect repo and set "app.py" as the entry file.
- Render: repo deploy using the provided `render.yaml`.
- Hugging Face Spaces: choose Space type "Streamlit" and point to `app.py`.
- Railway/Heroku: use the provided `Procfile` and `runtime.txt`.

### Render (recommended)
Render will detect `render.yaml` and create a Web Service automatically.

If creating manually:
- Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- Environment: Python 3.11 (set `PYTHON_VERSION=3.11` if needed)

### Why Vercel/Netlify 404?
They expect either a static site build or a Node/Edge serverless entrypoint. Streamlit is a stateful Python server and doesn’t produce a static build, so you’ll see `404: NOT_FOUND`. Deploy to one of the Python hosts above.
i want to make a project on agentic app that is used to control the whole laptop using commands whith commands it have to control the whole laptop likeeg open whatsapp is level 1 and level 2 complexity is open whatsapp and send hi message to this person ok like this and how it works is when i tell command it have to think and plan steps to execute and while perfroming tasks executions for coorrect task perfromation it have to take screenshorts and analyse and check wether the operation is working correctly or not see we are using openrouter ai keyps for this whole project u need to select best models free large models and do this project 