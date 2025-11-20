import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ..core.session import get_session_state


def render_visualization_tab():
    st.subheader("Visualizations & Alerts")
    ss = get_session_state()
    if "proc_df" not in ss:
        st.info("Please preprocess data first.")
        return
    if "meta" not in ss:
        st.info("Missing metadata from preprocessing.")
        return

    df: pd.DataFrame = ss["proc_df"]
    meta = ss["meta"]
    nddi_col = meta["nddi_col"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[nddi_col], mode="lines", name="NDDI"))
    thresh = meta.get("drought_thresh", 0.2)
    fig.add_hline(y=thresh, line=dict(color="red", dash="dash"), annotation_text=f"Threshold {thresh}")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if "model_artifacts" in ss:
        arts = ss["model_artifacts"]
        st.markdown("### Forecast summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean forecast (NDDI)", f"{arts['mean_pred']:.3f}")
        col2.metric("Uncertainty (std)", f"{arts['std_pred']:.3f}")
        col3.metric("Prob. drought", f"{arts['prob']*100:.1f}%")

        alert = arts.get("alert", False)
        if alert:
            st.error(f"ALERT: Probability exceeds {arts['prob_alert_threshold']}% threshold")
        else:
            st.success("No alert triggered.")

        # Show distribution of MC samples (English-labeled histogram)
        samples = np.array(arts["pred_samples"]) if isinstance(arts["pred_samples"], list) else arts["pred_samples"]
        st.markdown("#### Forecast distribution")
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(x=samples, nbinsx=20, name="Forecast samples", marker_color="#4C78A8")
        )
        # Mean line
        mean_pred = float(arts.get("mean_pred", np.mean(samples)))
        hist_fig.add_vline(x=mean_pred, line_width=2, line_dash="dash", line_color="green", annotation_text="Mean")
        # Drought threshold line
        thresh_line = meta.get("drought_thresh", 0.2)
        hist_fig.add_vline(x=thresh_line, line_width=2, line_dash="dot", line_color="red", annotation_text="Threshold")
        hist_fig.update_layout(
            title_text="Forecast distribution (NDDI)",
            xaxis_title="Predicted NDDI",
            yaxis_title="Count",
            bargap=0.02,
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    else:
        st.info("Train a model to view forecast and probabilities.")
