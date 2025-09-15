import streamlit as st
from .views.data_ingestion import render_data_tab
from .views.preprocessing import render_preprocess_tab
from .views.modeling import render_model_tab
from .views.visualization import render_visualization_tab


def run_app():
    st.title("Drought Early Warning (NDDI + LSTM)")

    tabs = st.tabs([
        "1) Data",
        "2) Preprocess",
        "3) Model",
        "4) Visualize",
    ])

    with tabs[0]:
        render_data_tab()
    with tabs[1]:
        render_preprocess_tab()
    with tabs[2]:
        render_model_tab()
    with tabs[3]:
        render_visualization_tab()
