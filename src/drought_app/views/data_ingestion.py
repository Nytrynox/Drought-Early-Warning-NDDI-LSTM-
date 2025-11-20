import os
import io
import zipfile
from typing import Optional
import streamlit as st
import pandas as pd
import requests

from ..core.session import get_session_state
from ..core.data import load_csv_from_bytes, kaggle_download_dataset
from ..utils.sample import make_synthetic_dataset


DATA_DIR = ".data"


def render_data_tab():
    st.subheader("Data ingestion")
    ss = get_session_state()

    with st.expander("Upload CSV", expanded=True):
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"]) 
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            ss["raw_df"] = df
            ss["source_name"] = getattr(uploaded, 'name', 'uploaded.csv')
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head(50))

    with st.expander("Load from URL"):
        url = st.text_input("HTTP(s) CSV URL")
        if st.button("Fetch CSV", use_container_width=True, type="primary") and url:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                df = load_csv_from_bytes(r.content)
                ss["raw_df"] = df
                try:
                    from urllib.parse import urlparse
                    ss["source_name"] = os.path.basename(urlparse(url).path) or "remote.csv"
                except Exception:
                    ss["source_name"] = "remote.csv"
                st.success(f"Loaded {len(df)} rows from URL")
                st.dataframe(df.head(50))
            except Exception as e:
                st.error(f"Failed to fetch: {e}")

    with st.expander("Kaggle dataset (optional)"):
        st.caption("Requires Kaggle API credentials at ~/.kaggle/kaggle.json")
        ds_slug = st.text_input("Kaggle dataset slug", value="datasetengineer/crop-health-and-environmental-stress-dataset")
        subpath = st.text_input("Optional file filter (e.g., .csv)", value=".csv")
        if st.button("Download from Kaggle"):
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                files = kaggle_download_dataset(ds_slug, DATA_DIR)
                csv_files = [f for f in files if f.endswith(subpath)] if subpath else files
                if not csv_files and files:
                    st.warning("No files matched filter; showing all")
                    csv_files = files
                st.success(f"Downloaded {len(csv_files)} files")
                # if a single CSV, load it
                loaded_any = False
                for f in csv_files:
                    if f.lower().endswith(".csv"):
                        df = pd.read_csv(f)
                        ss["raw_df"] = df
                        ss["source_name"] = os.path.basename(f)
                        st.info(f"Loaded {os.path.basename(f)} -> {len(df)} rows")
                        st.dataframe(df.head(50))
                        loaded_any = True
                        break
                if not loaded_any:
                    st.info("No CSV loaded automatically; please upload or specify exact file.")
            except Exception as e:
                st.error(f"Kaggle download failed: {e}")

    with st.expander("Synthetic sample (quick demo)"):
        if st.button("Load synthetic NDVI/NDWI demo"):
            df = make_synthetic_dataset()
            ss["raw_df"] = df
            ss["source_name"] = "synthetic.csv"
            st.success(f"Loaded synthetic dataset: {len(df)} rows")
            st.dataframe(df.head(50))

    if "raw_df" in ss:
        st.success("Data is in session. Proceed to Preprocess tab.")
