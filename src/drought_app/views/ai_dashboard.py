"""
AI Dashboard - Integrated Multi-Model Comparison with Real-time NDDI Satellite Data
Combines LSTM, Random Forest, and SVR models for comprehensive drought prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# TensorFlow/Keras imports - Skip to avoid mutex lock issues
TENSORFLOW_AVAILABLE = False
# Note: TensorFlow disabled due to mutex lock on macOS
# The dashboard will use PyTorch LSTM from the existing model.py instead

from ..core.session import get_session_state
from ..utils.nddi import compute_nddi_column
from ..utils.export import (
    create_comprehensive_csv_export,
    record_satellite_data_to_csv,
    generate_download_button
)
from ..components.advanced_viz import (
    render_advanced_visualizations,
    create_drought_severity_gauge,
    create_nddi_heatmap,
    create_correlation_heatmap
)
from ..components.live_tracking import (
    create_live_tracking_dashboard,
    create_realtime_nddi_chart,
    create_ndvi_ndwi_scatter,
    create_drought_timeline,
    create_environmental_dashboard
)
from ..components.comprehensive_viz import (
    create_model_comparison_plot,
    create_water_stress_map_2d,
    create_water_stress_map_3d,
    create_detailed_heatmap,
    create_linear_comparison_plot,
    create_3d_terrain_map,
    create_water_stress_gauge
)
from ..core.advanced_models import (
    train_cnn_model,
    train_catboost_model
)


def fetch_satellite_nddi_data(lat, lon, start_date, end_date):
    """
    Fetch real-time NDDI data from satellite imagery using Google Earth Engine
    Falls back to simulated data if GEE not available
    """
    from ..utils.satellite import fetch_real_time_satellite_data
    
    st.info(f"📡 Fetching satellite data for coordinates: ({lat:.4f}, {lon:.4f})")
    
    try:
        # Try to fetch real satellite data
        days = (end_date - start_date).days
        satellite_df = fetch_real_time_satellite_data(lat, lon, days_back=days, source='auto')
        
        # Check if we got real data or simulated
        if 'Data_Quality' in satellite_df.columns and satellite_df['Data_Quality'].iloc[0] == 'Good':
            if len(satellite_df) < 50:
                st.success(f"✅ Fetched {len(satellite_df)} real satellite observations from Google Earth Engine!")
            else:
                st.info(f"ℹ️  Using simulated realistic satellite data ({len(satellite_df)} points)")
        
        return satellite_df
        
    except Exception as e:
        st.warning(f"⚠️  Could not fetch real satellite data: {e}")
        st.info("Using simulated realistic satellite data for demonstration")
        
        # Fallback to simulated data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        time_idx = np.arange(n)
        seasonal = 0.3 * np.sin(2 * np.pi * time_idx / 365)
        trend = -0.0001 * time_idx
        noise = np.random.normal(0, 0.05, n)
        
        nddi_values = 0.5 + seasonal + trend + noise
        ndvi_values = 0.6 + 0.2 * np.sin(2 * np.pi * time_idx / 365) + np.random.normal(0, 0.03, n)
        ndwi_values = 0.3 + 0.1 * np.sin(2 * np.pi * time_idx / 365 + np.pi/2) + np.random.normal(0, 0.03, n)
        
        satellite_df = pd.DataFrame({
            'Date': dates,
            'NDVI': ndvi_values.clip(0, 1),
            'NDWI': ndwi_values.clip(0, 1),
            'NDDI': nddi_values.clip(-1, 1),
            'Latitude': lat,
            'Longitude': lon,
            'Temperature': 25 + 10 * np.sin(2 * np.pi * time_idx / 365) + np.random.normal(0, 2, n),
            'Precipitation': np.maximum(0, 50 + 30 * np.sin(2 * np.pi * time_idx / 365 + np.pi) + np.random.normal(0, 10, n)),
            'Soil_Moisture': 0.4 + 0.1 * np.sin(2 * np.pi * time_idx / 365) + np.random.normal(0, 0.05, n)
        })
        
        return satellite_df


def create_sequences(data, lookback=10):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, 0])  # First column as target
    return np.array(X), np.array(y)


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def render_ai_dashboard():
    """Main AI Dashboard rendering function"""
    st.title("🌾 AI-Powered Drought Prediction Dashboard")
    st.markdown("### Multi-Model Analysis with Real-time Satellite Data Integration")
    
    ss = get_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Upload CSV", "Real-time Satellite", "Use Agriculture Dataset", "Session Data"]
        )
        
        st.divider()
        
        # Model configuration
        st.subheader("Model Settings")
        lookback = st.slider("Lookback Window", 3, 20, 5, help="Fewer steps = faster training")
        epochs = st.slider("LSTM Epochs", 1, 10, 3, help="Fewer epochs = faster training")
        sample_size = st.slider("Data Sample Size (%)", 10, 100, 100, step=5, help="Use 100% for full dataset or reduce for faster testing")
        
        st.divider()
        
        # Satellite data configuration
        if data_source == "Real-time Satellite":
            st.subheader("📡 Satellite Config")
            lat = st.number_input("Latitude", -90.0, 90.0, 28.6139, format="%.4f")
            lon = st.number_input("Longitude", -180.0, 180.0, 77.2090, format="%.4f")
            days_back = st.slider("Days of History", 30, 365, 180)
    
    # Main content area
    df_to_use = None
    
    # Data loading based on source
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Agricultural Data CSV", type=['csv'])
        if uploaded_file:
            df_to_use = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_to_use)} rows, {len(df_to_use.columns)} columns")
    
    elif data_source == "Real-time Satellite":
        if st.button("🛰️ Fetch Satellite Data", type="primary"):
            with st.spinner("Fetching satellite imagery data..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                df_to_use = fetch_satellite_nddi_data(lat, lon, start_date, end_date)
                ss["satellite_df"] = df_to_use
                
                # Auto-record to CSV
                record_satellite_data_to_csv(df_to_use, lat, lon)
                
                st.success(f"✅ Fetched {len(df_to_use)} days of satellite data")
                st.info("📝 Data automatically recorded to satellite_data_records/")
                
                # Live tracking dashboard
                st.divider()
                create_live_tracking_dashboard(lat, lon, days_back)
                
                # Real-time visualizations
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 NDDI Tracking",
                    "🌿 NDVI vs NDWI",
                    "⏱️ Drought Timeline",
                    "🌡️ Environmental"
                ])
                
                with tab1:
                    fig_nddi = create_realtime_nddi_chart(df_to_use)
                    st.plotly_chart(fig_nddi, use_container_width=True, key="realtime_nddi_chart")
                    
                    # Current status
                    latest = df_to_use.iloc[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current NDDI", f"{latest.get('NDDI', 0):.3f}")
                    col2.metric("7-Day Avg", f"{df_to_use['NDDI'].tail(7).mean():.3f}")
                    col3.metric("30-Day Avg", f"{df_to_use['NDDI'].tail(30).mean():.3f}")
                    col4.metric("Trend", "📉" if df_to_use['NDDI'].tail(7).mean() < df_to_use['NDDI'].tail(30).mean() else "📈")
                
                with tab2:
                    fig_scatter = create_ndvi_ndwi_scatter(df_to_use)
                    if fig_scatter:
                        st.plotly_chart(fig_scatter, use_container_width=True, key="ndvi_ndwi_scatter")
                        st.info("💡 Points colored by NDDI - Green = wet, Red = dry")
                
                with tab3:
                    fig_timeline = create_drought_timeline(df_to_use)
                    if fig_timeline:
                        st.plotly_chart(fig_timeline, use_container_width=True, key="drought_timeline")
                
                with tab4:
                    fig_env = create_environmental_dashboard(df_to_use)
                    if fig_env:
                        st.plotly_chart(fig_env, use_container_width=True, key="environmental_dashboard")
                    else:
                        st.info("Environmental data not available in current dataset")
                
                # Show map
                st.divider()
                st.markdown("### 🗺️ Real-time Satellite Location Map")
                from ..components.map_viewer import create_nddi_map
                if 'Latitude' in df_to_use.columns and 'Longitude' in df_to_use.columns and 'NDDI' in df_to_use.columns:
                    map_fig = create_nddi_map(df_to_use)
                    st.plotly_chart(map_fig, use_container_width=True, key="satellite_location_map")
                    
                    # Show latest data point
                    latest = df_to_use.iloc[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📍 Location", f"{lat:.2f}, {lon:.2f}")
                    col2.metric("🌱 NDVI", f"{latest.get('NDVI', 0):.3f}")
                    col3.metric("💧 NDWI", f"{latest.get('NDWI', 0):.3f}")
                    col4.metric("🌾 NDDI", f"{latest.get('NDDI', 0):.3f}")
        
        if "satellite_df" in ss:
            df_to_use = ss["satellite_df"]
            
            # Download button for satellite data
            st.markdown("### 💾 Export Satellite Data")
            generate_download_button(
                df_to_use,
                filename=f"satellite_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                button_text="📥 Download Satellite Data CSV"
            )
    
    elif data_source == "Use Agriculture Dataset":
        try:
            # Try to load the agriculture dataset
            df_to_use = pd.read_csv("agriculture_dataset.csv")
            st.success(f"✅ Loaded agriculture dataset: {len(df_to_use)} rows")
        except Exception as e:
            st.error(f"Could not load agriculture_dataset.csv: {e}")
            st.info("Please ensure agriculture_dataset.csv is in the project root")
    
    elif data_source == "Session Data":
        if "proc_df" in ss or "raw_df" in ss:
            df_to_use = ss.get("proc_df", ss.get("raw_df"))
            st.success(f"✅ Using session data: {len(df_to_use)} rows")
        else:
            st.warning("No data in session. Please load data from other tabs first.")
    
    # Process and analyze data if available
    if df_to_use is not None and len(df_to_use) > 0:
        # Display data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df_to_use.head(20))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df_to_use))
            col2.metric("Total Columns", len(df_to_use.columns))
            col3.metric("Numeric Columns", len(df_to_use.select_dtypes(include=[np.number]).columns))
        
        # Column selection
        numeric_cols = df_to_use.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in dataset")
            return
        
        # Target column selection
        target_col = st.selectbox(
            "Select Target Column for Prediction",
            numeric_cols,
            index=0 if 'NDDI' not in numeric_cols else numeric_cols.index('NDDI')
        )
        
        # Model training section
        st.divider()
        st.header("🤖 Multi-Model Training & Comparison")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"⚡ Training with {sample_size}% of data ({int(len(df_to_use) * sample_size / 100):,} rows) for faster results")
        with col2:
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                train_and_compare_models(df_to_use, target_col, lookback, epochs, sample_size, ss)
        
        # Display results if available
        if "ai_results" in ss:
            display_model_results(ss["ai_results"], df_to_use)
        
        # Visualization Tabs - Show only after data is loaded
        st.divider()
        st.header("📊 Data Visualizations & Analysis")
        
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "� Data Overview", 
            "🔥 Heatmaps", 
            "🎯 Metrics",
            "🗺️ Geo Maps"
        ])
        
        with viz_tab1:
            st.subheader("📊 Dataset Statistics")
            
            # Show data info
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", f"{len(df_to_use):,}")
            col2.metric("Features", len(df_to_use.columns))
            col3.metric("Numeric Cols", len(df_to_use.select_dtypes(include=[np.number]).columns))
            col4.metric("Date Range", f"{len(df_to_use)} days" if 'Date' in df_to_use.columns else "N/A")
            
            # Time series plots for key features
            st.markdown("### 📈 Key Features Over Time")
            numeric_cols = df_to_use.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                # Create multi-line plot
                selected_features = st.multiselect(
                    "Select features to visualize",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if selected_features:
                    fig = go.Figure()
                    for feature in selected_features:
                        fig.add_trace(go.Scatter(
                            y=df_to_use[feature],
                            mode='lines',
                            name=feature
                        ))
                    
                    fig.update_layout(
                        title="Feature Trends",
                        xaxis_title="Index" if 'Date' not in df_to_use.columns else "Time",
                        yaxis_title="Value",
                        height=500,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True, key="feature_trends")
        
        with viz_tab2:
            st.subheader("🔥 Correlation & Distribution Heatmaps")
            
            # Correlation matrix
            st.markdown("#### Feature Correlation Matrix")
            fig_corr = create_correlation_heatmap(df_to_use)
            st.plotly_chart(fig_corr, use_container_width=True, key="correlation_heatmap")
            
            # Temporal heatmap if Date and NDDI exist
            if 'Date' in df_to_use.columns and 'NDDI' in df_to_use.columns:
                st.markdown("#### Temporal NDDI Heatmap")
                fig_temp = create_nddi_heatmap(df_to_use)
                if fig_temp:
                    st.plotly_chart(fig_temp, use_container_width=True, key="temporal_nddi_heatmap")
            else:
                st.info("ℹ️ Temporal heatmap requires 'Date' and 'NDDI' columns")
        
        with viz_tab3:
            st.subheader("🎯 Feature Statistics & Distributions")
            
            numeric_cols = df_to_use.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select feature for detailed analysis", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=df_to_use[selected_col],
                        nbinsx=50,
                        name=selected_col
                    ))
                    fig_hist.update_layout(
                        title=f"{selected_col} Distribution",
                        xaxis_title=selected_col,
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"histogram_{selected_col}")
                
                with col2:
                    # Box plot
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=df_to_use[selected_col],
                        name=selected_col
                    ))
                    fig_box.update_layout(
                        title=f"{selected_col} Box Plot",
                        yaxis_title=selected_col,
                        height=400
                    )
                    st.plotly_chart(fig_box, use_container_width=True, key=f"boxplot_{selected_col}")
                
                # Statistics table
                st.markdown("#### Statistical Summary")
                stats_df = df_to_use[numeric_cols].describe()
                st.dataframe(stats_df, use_container_width=True)
        
        with viz_tab4:
            st.subheader("🗺️ Geographical Visualization")
            
            # Check for geo columns
            has_lat = 'Latitude' in df_to_use.columns or 'latitude' in df_to_use.columns or 'lat' in df_to_use.columns
            has_lon = 'Longitude' in df_to_use.columns or 'longitude' in df_to_use.columns or 'lon' in df_to_use.columns
            
            if has_lat and has_lon:
                lat_col = next((col for col in df_to_use.columns if 'lat' in col.lower()), None)
                lon_col = next((col for col in df_to_use.columns if 'lon' in col.lower()), None)
                
                # Create map
                from ..components.map_viewer import create_nddi_map
                value_col = 'NDDI' if 'NDDI' in df_to_use.columns else numeric_cols[0]
                
                st.markdown(f"#### Map showing {value_col}")
                
                # Prepare data for map
                map_df = df_to_use[[lat_col, lon_col, value_col]].copy()
                map_df.columns = ['Latitude', 'Longitude', 'NDDI']
                
                map_fig = create_nddi_map(map_df)
                st.plotly_chart(map_fig, use_container_width=True, key="geo_visualization_map")
                
                # Show location stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Unique Locations", df_to_use[[lat_col, lon_col]].drop_duplicates().shape[0])
                col2.metric("Avg Latitude", f"{df_to_use[lat_col].mean():.4f}")
                col3.metric("Avg Longitude", f"{df_to_use[lon_col].mean():.4f}")
            else:
                st.info("ℹ️ Geographical visualization requires Latitude and Longitude columns")
                st.markdown("**Available columns:**")
                st.write(df_to_use.columns.tolist())
        
        # Export section
        st.divider()
        st.header("💾 Export Data & Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export current data
            generate_download_button(
                df_to_use,
                filename=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                button_text="📥 Download Current Dataset"
            )
        
        with col2:
            # Comprehensive export with results
            if "ai_results" in ss:
                if st.button("📦 Create Comprehensive Export Package", use_container_width=True):
                    with st.spinner("Creating export package..."):
                        zip_bytes = create_comprehensive_csv_export(
                            df_to_use,
                            ss.get("ai_results", {}),
                            metadata={
                                'target_column': target_col,
                                'lookback': lookback,
                                'data_source': data_source
                            }
                        )
                        
                        st.download_button(
                            label="⬇️ Download ZIP Package",
                            data=zip_bytes,
                            file_name=f"drought_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                        st.success("✅ Export package ready!")


def train_and_compare_models(df, target_col, lookback, epochs, sample_size, ss):
    """Train LSTM, Random Forest, and SVR models and compare results"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Sample data for faster training
        status_text.text(f"📊 Sampling {sample_size}% of data for faster training...")
        progress_bar.progress(5)
        
        # Use random sampling to get representative subset
        if sample_size < 100:
            df_sampled = df.sample(frac=sample_size/100, random_state=42)
            st.info(f"⚡ Using {len(df_sampled):,} rows (sampled from {len(df):,}) for speed")
        else:
            df_sampled = df
        
        # Prepare data
        status_text.text("📊 Preparing features...")
        progress_bar.progress(10)
        
        df_numeric = df_sampled.select_dtypes(include=[np.number])
        df_numeric = df_numeric.fillna(df_numeric.mean())
        
        # Scale data and convert to float32 for MPS compatibility
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_numeric.values).astype(np.float32)
        
        # Create sequences
        status_text.text("🔄 Creating sequences...")
        progress_bar.progress(20)
        
        X, y = create_sequences(data_scaled, lookback=lookback)
        
        # Train-test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        results = {
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'target_col': target_col,
            'models': {}
        }
        
        # 1. Train LSTM using PyTorch (from existing model.py)
        status_text.text("🧠 Training PyTorch LSTM model...")
        progress_bar.progress(30)
        
        # Use the existing PyTorch LSTM from core.model
        from ..core.model import build_lstm_model, train_model as train_pytorch_model
        import torch
        
        # Reshape for PyTorch (needs channels last)
        X_train_torch = X_train[:, :, 0:1]  # Take first feature
        X_test_torch = X_test[:, :, 0:1]
        
        # Smaller model for faster training
        model_lstm = build_lstm_model(input_steps=lookback, units=32, dropout=0.1)
        history_dict = train_pytorch_model(
            model_lstm, 
            X_train_torch, y_train,
            X_test_torch, y_test,
            batch_size=64,  # Larger batch = faster
            epochs=epochs
        )
        
        # Get predictions - use same device detection as training
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        model_lstm.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test_torch).to(device)
            y_pred_lstm = model_lstm(X_test_tensor).cpu().numpy().reshape(-1)
        
        rmse_lstm, mae_lstm, r2_lstm = calculate_metrics(y_test, y_pred_lstm)
        
        results['models']['LSTM'] = {
            'predictions': y_pred_lstm,
            'rmse': rmse_lstm,
            'mae': mae_lstm,
            'r2': r2_lstm,
            'history': history_dict,
            'model': model_lstm
        }
        
        # 2. Train Random Forest
        status_text.text("🌲 Training Random Forest...")
        progress_bar.progress(60)
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Faster Random Forest with fewer trees
        rf = RandomForestRegressor(
            n_estimators=50,  # Reduced from 100
            max_depth=10,  # Limit depth for speed
            random_state=42, 
            n_jobs=-1,  # Use all CPU cores
            min_samples_split=5
        )
        rf.fit(X_train_flat, y_train)
        y_pred_rf = rf.predict(X_test_flat)
        
        rmse_rf, mae_rf, r2_rf = calculate_metrics(y_test, y_pred_rf)
        
        results['models']['Random Forest'] = {
            'predictions': y_pred_rf,
            'rmse': rmse_rf,
            'mae': mae_rf,
            'r2': r2_rf,
            'model': rf
        }
        
        # 3. Train CNN
        status_text.text("🔥 Training CNN model...")
        progress_bar.progress(50)
        
        model_cnn, history_cnn, y_pred_cnn = train_cnn_model(
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_test, 
            y_val=y_test,
            epochs=epochs,
            batch_size=64
        )
        
        rmse_cnn, mae_cnn, r2_cnn = calculate_metrics(y_test, y_pred_cnn)
        
        results['models']['CNN'] = {
            'predictions': y_pred_cnn,
            'rmse': rmse_cnn,
            'mae': mae_cnn,
            'r2': r2_cnn,
            'history': history_cnn,
            'model': model_cnn
        }
        
        # 4. Train CatBoost
        status_text.text("🚀 Training CatBoost model...")
        progress_bar.progress(65)
        
        model_catboost, y_pred_catboost = train_catboost_model(
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_test, 
            y_val=y_test
        )
        
        rmse_catboost, mae_catboost, r2_catboost = calculate_metrics(y_test, y_pred_catboost)
        
        results['models']['CatBoost'] = {
            'predictions': y_pred_catboost,
            'rmse': rmse_catboost,
            'mae': mae_catboost,
            'r2': r2_catboost,
            'model': model_catboost
        }
        
        # 5. Train Random Forest
        status_text.text("🌲 Training Random Forest...")
        progress_bar.progress(80)
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Faster Random Forest with fewer trees
        rf = RandomForestRegressor(
            n_estimators=50,  # Reduced from 100
            max_depth=10,  # Limit depth for speed
            random_state=42, 
            n_jobs=-1,  # Use all CPU cores
            min_samples_split=5
        )
        rf.fit(X_train_flat, y_train)
        y_pred_rf = rf.predict(X_test_flat)
        
        rmse_rf, mae_rf, r2_rf = calculate_metrics(y_test, y_pred_rf)
        
        results['models']['Random Forest'] = {
            'predictions': y_pred_rf,
            'rmse': rmse_rf,
            'mae': mae_rf,
            'r2': r2_rf,
            'model': rf
        }
        
        # 6. Train SVR (faster with linear kernel and subset)
        status_text.text("🎯 Training SVR...")
        progress_bar.progress(90)
        
        # Use subset for SVR (it's slow on large datasets)
        max_svr_samples = 5000
        if len(X_train_flat) > max_svr_samples:
            idx = np.random.choice(len(X_train_flat), max_svr_samples, replace=False)
            X_train_svr = X_train_flat[idx]
            y_train_svr = y_train[idx]
        else:
            X_train_svr = X_train_flat
            y_train_svr = y_train
        
        svr = SVR(kernel='linear', C=1.0)  # Linear kernel is much faster
        svr.fit(X_train_svr, y_train_svr)
        y_pred_svr = svr.predict(X_test_flat)
        
        rmse_svr, mae_svr, r2_svr = calculate_metrics(y_test, y_pred_svr)
        
        results['models']['SVR'] = {
            'predictions': y_pred_svr,
            'rmse': rmse_svr,
            'mae': mae_svr,
            'r2': r2_svr,
            'model': svr
        }
        
        # Store results
        ss["ai_results"] = results
        
        status_text.text("✅ All 5 models trained successfully!")
        progress_bar.progress(100)
        
        st.success("🎉 All models trained successfully!")
        
    except Exception as e:
        st.error(f"Training failed: {e}")
        import traceback
        st.code(traceback.format_exc())


def display_model_results(results, df_source=None):
    """Display comprehensive model comparison results"""
    
    st.divider()
    st.header("📈 Model Performance Comparison")
    
    # Metrics comparison table
    metrics_data = []
    for model_name, model_info in results['models'].items():
        metrics_data.append({
            'Model': model_name,
            'RMSE': f"{model_info['rmse']:.4f}",
            'MAE': f"{model_info['mae']:.4f}",
            'R² Score': f"{model_info['r2']:.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        # Best model
        best_model = min(results['models'].items(), key=lambda x: x[1]['rmse'])
        st.metric("🏆 Best Model", best_model[0])
        st.metric("Best RMSE", f"{best_model[1]['rmse']:.4f}")
    
    # Visualizations
    st.divider()
    st.header("📊 Comprehensive Model Visualizations")
    
    tabs = st.tabs([
        "📈 Linear Comparison",
        "🔬 Detailed Comparison",
        "📊 Regression Plots",
        "📉 Training History",
        "🗺️ Water Stress Maps",
        "🔥 3D Visualizations"
    ])
    
    y_test = results['y_test']
    
    # Tab 1: Linear comparison for all models
    with tabs[0]:
        st.markdown("### Linear Comparison: All Models")
        fig_linear = create_linear_comparison_plot(results, y_test)
        st.plotly_chart(fig_linear, use_container_width=True, key="linear_comparison_all")
        
        # Show separate plots for each model
        st.markdown("### Individual Model Details")
        
        model_names = list(results['models'].keys())
        num_models = len(model_names)
        cols = st.columns(min(3, num_models))
        
        for idx, (model_name, model_info) in enumerate(results['models'].items()):
            with cols[idx % 3]:
                sample_size = min(200, len(y_test))
                indices = sorted(random.sample(range(len(y_test)), sample_size))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(indices))),
                    y=y_test[indices],
                    mode='lines',
                    name='Actual',
                    line=dict(color='black', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(indices))),
                    y=model_info['predictions'][indices],
                    mode='lines',
                    name=model_name,
                    line=dict(dash='dot', width=2)
                ))
                
                fig.update_layout(
                    title=f"{model_name}<br>RMSE={model_info['rmse']:.4f}, R²={model_info['r2']:.4f}",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True, key=f"linear_detail_{model_name}")
    
    # Tab 2: Detailed multi-model comparison
    with tabs[1]:
        st.markdown("### Comprehensive Multi-Model Performance Analysis")
        fig_comparison = create_model_comparison_plot(results, y_test, model_names=list(results['models'].keys()))
        st.plotly_chart(fig_comparison, use_container_width=True, key="detailed_comparison")
        
        # Performance metrics table
        st.markdown("### Performance Metrics Comparison")
        metrics_df = pd.DataFrame([
            {
                'Model': name,
                'RMSE': f"{info['rmse']:.6f}",
                'MAE': f"{info['mae']:.6f}",
                'R² Score': f"{info['r2']:.6f}",
                'Rank (by RMSE)': ''
            }
            for name, info in results['models'].items()
        ])
        
        # Add ranking
        sorted_models = sorted(results['models'].items(), key=lambda x: x[1]['rmse'])
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            metrics_df.loc[metrics_df['Model'] == model_name, 'Rank (by RMSE)'] = f"#{rank}"
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Tab 3: Regression scatter plots
    with tabs[2]:
        st.markdown("### Predicted vs Actual Regression Analysis")
        cols = st.columns(3)
        
        for idx, (model_name, model_info) in enumerate(results['models'].items()):
            with cols[idx % 3]:
                sample_size = min(300, len(y_test))
                indices = sorted(random.sample(range(len(y_test)), sample_size))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=y_test[indices],
                    y=model_info['predictions'][indices],
                    mode='markers',
                    name=model_name,
                    marker=dict(size=6, opacity=0.6, color='blue')
                ))
                
                # Perfect fit line
                min_val = min(y_test.min(), model_info['predictions'].min())
                max_val = max(y_test.max(), model_info['predictions'].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Fit',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{model_name}<br>R² = {model_info['r2']:.4f}",
                    xaxis_title="Actual",
                    yaxis_title="Predicted",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"regression_{model_name}")
        
        # Combined comparison
        st.markdown("### All Models Combined")
        fig = go.Figure()
        
        sample_size = min(300, len(y_test))
        indices = sorted(random.sample(range(len(y_test)), sample_size))
        
        colors_map = {'LSTM': 'blue', 'CNN': 'red', 'CatBoost': 'green', 'Random Forest': 'orange', 'SVR': 'purple'}
        
        for model_name, model_info in results['models'].items():
            fig.add_trace(go.Scatter(
                x=y_test[indices],
                y=model_info['predictions'][indices],
                mode='markers',
                name=f"{model_name} (R²={model_info['r2']:.3f})",
                marker=dict(size=6, opacity=0.5, color=colors_map.get(model_name, 'gray'))
            ))
        
        # Perfect fit line
        min_val = y_test.min()
        max_val = y_test.max()
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='black', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="All Models Comparison: Predicted vs Actual",
            xaxis_title="Actual Value",
            yaxis_title="Predicted Value",
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="regression_combined")
    
    # Tab 4: Training history for neural models
    with tabs[3]:
        st.markdown("### Neural Network Training History")
        
        neural_models = ['LSTM', 'CNN']
        available_neural = [m for m in neural_models if m in results['models'] and 'history' in results['models'][m]]
        
        if available_neural:
            for model_name in available_neural:
                st.markdown(f"#### {model_name} Training History")
                history = results['models'][model_name]['history']
                
                if isinstance(history, dict):
                    fig = go.Figure()
                    
                    if 'train_loss' in history:
                        fig.add_trace(go.Scatter(
                            y=history['train_loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='blue')
                        ))
                    
                    if 'val_loss' in history:
                        fig.add_trace(go.Scatter(
                            y=history['val_loss'],
                            mode='lines',
                            name='Validation Loss',
                            line=dict(color='red')
                        ))
                    
                    fig.update_layout(
                        title=f"{model_name} Training Progress",
                        xaxis_title="Epoch",
                        yaxis_title="Loss (MSE)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"training_history_{model_name}")
        else:
            st.info("No training history available for neural models")
    
    # Tab 5: Water stress maps (2D/3D)
    with tabs[4]:
        st.markdown("### Water Stress Level Visualization")
        
        if df_source is not None and len(df_source) > 0:
            # Check for required columns
            has_geo = all(col in df_source.columns for col in ['Latitude', 'Longitude'])
            
            if has_geo:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 2D Water Stress Map")
                    fig_2d = create_water_stress_map_2d(df_source)
                    st.plotly_chart(fig_2d, use_container_width=True, key="water_stress_2d")
                
                with col2:
                    st.markdown("#### 3D Water Stress Visualization")
                    fig_3d = create_water_stress_map_3d(df_source)
                    st.plotly_chart(fig_3d, use_container_width=True, key="water_stress_3d")
                
                # Current water stress gauge
                st.markdown("#### Current Water Stress Level")
                latest_stress = 0
                if 'NDDI' in df_source.columns:
                    latest_nddi = df_source['NDDI'].iloc[-1]
                    if latest_nddi < -0.5:
                        latest_stress = 10
                    elif latest_nddi < -0.2:
                        latest_stress = 7
                    elif latest_nddi < 0:
                        latest_stress = 4
                    else:
                        latest_stress = 2
                
                fig_gauge = create_water_stress_gauge(latest_stress)
                st.plotly_chart(fig_gauge, use_container_width=True, key="water_stress_gauge")
            else:
                st.warning("⚠️ Water stress maps require Latitude and Longitude columns")
                st.info("Available columns: " + ", ".join(df_source.columns.tolist()))
        else:
            st.info("No geographic data available for water stress mapping")
    
    # Tab 6: 3D terrain and detailed heatmaps
    with tabs[5]:
        st.markdown("### 3D Terrain & Advanced Heatmaps")
        
        if df_source is not None and len(df_source) > 0:
            # 3D terrain map
            if all(col in df_source.columns for col in ['Latitude', 'Longitude', 'NDDI']):
                st.markdown("#### 3D Terrain Map (NDDI)")
                fig_terrain = create_3d_terrain_map(df_source)
                if fig_terrain:
                    st.plotly_chart(fig_terrain, use_container_width=True, key="3d_terrain_map")
                else:
                    st.info("Could not generate 3D terrain map")
            
            # Detailed correlation heatmap
            st.markdown("#### Detailed Feature Correlation Heatmap")
            fig_heatmap = create_detailed_heatmap(df_source)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True, key="detailed_correlation_heatmap")
            else:
                st.info("Heatmap requires multiple numeric features")
        else:
            st.info("No data available for 3D visualizations")
    
    # Metrics bar chart
    st.divider()
    st.markdown("### Performance Metrics Bar Chart")
    
    metrics_comparison = pd.DataFrame([
        {
            'Model': name,
            'RMSE': info['rmse'],
            'MAE': info['mae'],
            'R² Score': info['r2']
        }
        for name, info in results['models'].items()
    ])
    
    fig_metrics = go.Figure()
    
    for metric in ['RMSE', 'MAE']:
        fig_metrics.add_trace(go.Bar(
            name=metric,
            x=metrics_comparison['Model'],
            y=metrics_comparison[metric]
        ))
    
    fig_metrics.update_layout(
        title="Model Metrics Comparison (Lower is Better)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig_metrics, use_container_width=True, key="metrics_bar_chart")
    
    # Download predictions
    st.divider()
    st.header("💾 Export Results")
    
    if st.button("Download Predictions as CSV"):
        export_df = pd.DataFrame({
            'Actual': y_test,
            **{
                f'{model_name}_Predicted': model_info['predictions']
                for model_name, model_info in results['models'].items()
            }
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"model_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
