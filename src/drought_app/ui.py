import streamlit as st
from .views.data_ingestion import render_data_tab
from .views.preprocessing import render_preprocess_tab
from .views.modeling import render_model_tab
from .views.visualization import render_visualization_tab
from .views.ai_dashboard import render_ai_dashboard


def run_app():
    st.title("🌾 Drought Early Warning System (NDDI + LSTM)")

    tabs = st.tabs([
        "🤖 AI Dashboard",
        "📊 Data",
        "🔧 Preprocess",
        "🧠 Model",
        "📈 Visualize",
    ])

    with tabs[0]:
        render_ai_dashboard()
    with tabs[1]:
        render_data_tab()
    with tabs[2]:
        render_preprocess_tab()
    with tabs[3]:
        render_model_tab()
    with tabs[4]:
        render_visualization_tab()
