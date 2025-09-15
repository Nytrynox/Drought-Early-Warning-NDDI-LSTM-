import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
from ..core.session import get_session_state
from ..core.model import build_lstm_model, SeriesWindow, train_model, mc_dropout_predict


def render_model_tab():
    st.subheader("LSTM Modeling and Probabilistic Forecast")
    ss = get_session_state()

    if "proc_df" not in ss:
        st.info("Please preprocess data first.")
        return

    df: pd.DataFrame = ss["proc_df"]
    meta = ss["meta"]
    nddi_col = meta["nddi_col"]
    # Show how many points we can train on
    st.caption(f"Data points available: {len(df)}")

    simple = st.checkbox("Simple mode (auto settings)", value=True)
    st.caption("Sequence settings")
    if simple:
        lookback = 12
        horizon = 1
        st.write({"lookback": lookback, "horizon": horizon})
    else:
        lookback = st.slider("Lookback window (timesteps)", 6, 48, 12)
        horizon = st.slider("Forecast horizon (timesteps ahead)", 1, 12, 1)

    st.caption("Model hyperparameters")
    if simple:
        units, dropout, batch_size, epochs = 32, 0.2, 32, 20
    else:
        units = st.slider("LSTM units", 8, 128, 32, step=8)
        dropout = st.slider("Dropout", 0.0, 0.8, 0.2, step=0.05)
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
        epochs = st.slider("Epochs", 5, 100, 20)

    st.caption("Probability & Alerts")
    if simple:
        mc_samples = 50
        prob_alert_threshold = 70
    else:
        mc_samples = st.slider("MC Dropout samples", 10, 200, 50, step=10)
        prob_alert_threshold = st.slider("Alert threshold (%)", 50, 95, 70, step=5)

    # Data cap settings for responsiveness
    cap_default = 3000
    if simple:
        use_last_n = None  # will default to cap_default inside handler
    else:
        use_last_n = st.slider("Use last N points for training", min_value=200, max_value=int(max(200, len(df))), value=int(min(len(df), cap_default)), step=100)

    if st.button("Train & Forecast", type="primary"):
        try:
            series = df[nddi_col].astype(float).values
            # Limit series length to keep training fast
            cap_n = use_last_n if use_last_n else min(len(series), cap_default)
            if len(series) > cap_n:
                series = series[-cap_n:]
                st.caption(f"Using last {cap_n} points for training (of {len(df)} available) for speed.")
            # Drop NaNs if any
            orig_len = len(series)
            series = series[~np.isnan(series)]
            if len(series) < orig_len:
                st.caption(f"Dropped {orig_len - len(series)} NaN values from series.")
            if len(series) < 8:
                # Baseline fallback: use last value with small noise as MC predictions
                last = float(series[-1])
                samples = np.random.normal(loc=last, scale=max(1e-6, np.std(series[-min(5,len(series)):]) * 0.1 or 1e-3), size=mc_samples)
                mean_pred = float(np.mean(samples))
                std_pred = float(np.std(samples))
                drought_thresh = float(meta["drought_thresh"])
                prob = float((samples >= drought_thresh).mean())
                ss["model_artifacts"] = {
                    "history": {"train_loss": [], "val_loss": []},
                    "mean_pred": mean_pred,
                    "std_pred": std_pred,
                    "pred_samples": samples.tolist(),
                    "prob": prob,
                    "alert": prob * 100.0 >= prob_alert_threshold,
                    "prob_alert_threshold": prob_alert_threshold,
                    "mode": "baseline",
                }
                st.info("Dataset is very small; showing baseline forecast from last value.")
                return
            # Ensure we have enough windows for training; require at least 10 windows by default.
            n = len(series)
            feasible_max_lookback = n - horizon - 2  # allow as few as 3 windows
            if feasible_max_lookback < 1:
                # Not enough data even with minimal lookback; guide the user
                needed = horizon + 3
                st.error(
                    f"Not enough data to train. Need >= {needed} points for horizon={horizon}. "
                    f"You currently have {n}. Try decreasing horizon or resampling to a finer frequency."
                )
                return
            adjusted_lookback = min(lookback, feasible_max_lookback)
            if adjusted_lookback != lookback:
                st.info(f"Adjusted lookback from {lookback} to {adjusted_lookback} to fit available data.")

            sw = SeriesWindow(series, lookback=adjusted_lookback, horizon=horizon)
            X_train, y_train, X_val, y_val = sw.get_train_val_split()

            # If series is large and in simple mode, reduce epochs for speed
            eff_epochs = epochs
            if simple and len(series) > 1500:
                eff_epochs = min(epochs, 10)
                st.caption(f"Large dataset detected; reducing epochs to {eff_epochs} for faster training.")

            with st.spinner("Training model and generating forecast..."):
                model = build_lstm_model(input_steps=adjusted_lookback, units=units, dropout=dropout)
                history = train_model(model, X_train, y_train, X_val, y_val, batch_size=batch_size, epochs=eff_epochs)

            # MC dropout predictions for latest window
            last_window = sw.get_last_window()
            preds = mc_dropout_predict(model, last_window, mc_samples=mc_samples)
            mean_pred = float(np.mean(preds))
            std_pred = float(np.std(preds))

            # Probability of drought based on threshold
            drought_thresh = float(meta["drought_thresh"])
            prob = float((preds >= drought_thresh).mean())

            ss["model_artifacts"] = {
                "history": history.history,
                "mean_pred": mean_pred,
                "std_pred": std_pred,
                "pred_samples": preds.tolist(),
                "prob": prob,
                "alert": prob * 100.0 >= prob_alert_threshold,
                "prob_alert_threshold": prob_alert_threshold,
            }
            st.success("Model trained and forecast computed.")
            st.json({
                "mean_pred": mean_pred,
                "std_pred": std_pred,
                "prob_drought": prob,
                "alert": prob * 100.0 >= prob_alert_threshold,
            })
        except Exception as e:
            st.error(f"Training or prediction failed: {e}")

    if "model_artifacts" in ss:
        st.success("Forecast ready. Proceed to Visualize tab.")
