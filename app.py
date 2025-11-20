import os
import sys

# Suppress ALL TensorFlow warnings and logs BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Suppress gRPC logs
os.environ['GLOG_minloglevel'] = '3'  # Suppress Google logging
import warnings
warnings.filterwarnings('ignore')

import streamlit as st

# Additional TensorFlow logging suppression after import
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except:
    pass

# Ensure src/ is on the path when running `streamlit run app.py`
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
	sys.path.insert(0, SRC_DIR)

# Show initial loading state
with st.spinner("🚀 Initializing Drought Early Warning System..."):
    from src.drought_app.ui import run_app

st.set_page_config(page_title="Drought Early Warning (NDDI + LSTM)", layout="wide")
run_app()
