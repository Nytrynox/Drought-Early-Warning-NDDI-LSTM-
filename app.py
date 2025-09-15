import os
import sys
import streamlit as st

# Ensure src/ is on the path when running `streamlit run app.py`
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
	sys.path.insert(0, SRC_DIR)

from src.drought_app.ui import run_app

st.set_page_config(page_title="Drought Early Warning (NDDI + LSTM)", layout="wide")
run_app()
