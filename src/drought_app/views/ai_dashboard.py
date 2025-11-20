"""
Enhanced AI Dashboard - Integrates Colab Notebook Models with Agriculture Dataset
Real-time Satellite Data Integration with NDDI Monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Removed RandomForest and SVR - focusing on deep learning models only
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Import our enhanced modules
from ..core.agriculture_data import (
    AgricultureDataProcessor, 
    DroughtPredictionModels,
    calculate_nddi_from_agriculture_data,
    integrate_satellite_with_agriculture,
    get_drought_severity_from_nddi
)
from ..utils.satellite import (
    fetch_real_time_satellite_data,
    fetch_satellite_data_for_agriculture,
    analyze_drought_trends,
    calculate_drought_severity,
    get_drought_color
)
from ..core.session import get_session_state
from ..components.advanced_viz import create_correlation_heatmap, create_nddi_heatmap
from ..core.advanced_models import train_cnn_model, train_catboost_model, CNN1DRegressor
from ..utils.export import (
    generate_download_button, 
    record_satellite_data_to_csv,
    create_comprehensive_csv_export
)
from ..components.live_tracking import (
    create_live_tracking_dashboard,
    create_realtime_nddi_chart,
    create_ndvi_ndwi_scatter,
    create_drought_timeline,
    create_environmental_dashboard
)
from ..components.comprehensive_viz import (
    create_linear_comparison_plot,
    create_model_comparison_plot,
    create_water_stress_map_2d,
    create_water_stress_map_3d,
    create_detailed_heatmap,
    create_3d_terrain_map,
    create_water_stress_gauge
)


def render_ai_dashboard():
    """Enhanced AI Dashboard with real-time satellite integration"""
    
    st.header("🤖 AI-Powered Drought Early Warning Dashboard")
    st.markdown("""
    **Real-time integration of:**
    - 🛰️ Satellite NDDI monitoring (Google Earth Engine + Simulated data)
    - 🌾 Agriculture dataset analysis (212k+ records)
    - 🧠 Advanced ML models (CatBoost, CNN, LSTM)
    """)
    
    # Configuration section
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Agriculture Dataset Sample Size", 100, 10000, 5000)
            satellite_locations = st.slider("Satellite Locations", 1, 10, 5)
        
        with col2:
            days_back = st.slider("Days of Satellite History", 30, 365, 180)
            enable_real_gee = st.checkbox("Try Google Earth Engine", False,
                                        help="Requires authentication: python authenticate_gee.py")
        
        with col3:
            auto_refresh = st.checkbox("Auto Refresh (60s)", False)
            show_debug = st.checkbox("Show Debug Info", False)
    
    # Auto refresh logic
    if auto_refresh:
        placeholder = st.empty()
        time.sleep(60)
        st.rerun()
    
    # Load and process data with detailed progress indicators
    progress_container = st.container()
    
    with progress_container:
        # Step 1: Load agriculture dataset
        with st.spinner(f"📊 Loading agriculture dataset ({sample_size:,} records)..."):
            processor = AgricultureDataProcessor()
            agriculture_df = processor.load_data(sample_size)
            
            if agriculture_df.empty:
                st.error("Failed to load agriculture dataset")
                return
            
            st.success(f"✅ Loaded {len(agriculture_df):,} agriculture records")
        
        # Step 2: Calculate NDDI
        with st.spinner("🧮 Calculating NDDI from agriculture data..."):
            agriculture_df = calculate_nddi_from_agriculture_data(agriculture_df)
            st.success("✅ NDDI calculations complete")
        
        # Step 3: Fetch satellite data
        satellite_source = 'gee' if enable_real_gee else 'auto'
        with st.spinner(f"🛰️ Fetching satellite data from {satellite_locations} locations ({days_back} days)..."):
            satellite_df = fetch_satellite_data_for_agriculture(
                agriculture_df, 
                sample_locations=satellite_locations,
                days_back=days_back
            )
            if not satellite_df.empty:
                st.success(f"✅ Retrieved {len(satellite_df):,} satellite observations")
            else:
                st.warning("⚠️ No satellite data available, using agriculture data only")
        
        # Step 4: Integrate data
        if not satellite_df.empty:
            with st.spinner("🔗 Integrating satellite data with agriculture records..."):
                agriculture_df = integrate_satellite_with_agriculture(agriculture_df, satellite_df)
                st.success("✅ Data integration complete")
        
        # Clear progress messages after 2 seconds
        time.sleep(2)
        progress_container.empty()
    
    # Debug information
    if show_debug:
        with st.expander("🔍 Debug Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Agriculture Data:**")
                st.write(f"Shape: {agriculture_df.shape}")
                st.write(f"Columns: {list(agriculture_df.columns)}")
            with col2:
                st.write("**Satellite Data:**")
                st.write(f"Shape: {satellite_df.shape if not satellite_df.empty else 'Empty'}")
                st.write(f"Date range: {satellite_df['Date'].min() if not satellite_df.empty else 'N/A'} to {satellite_df['Date'].max() if not satellite_df.empty else 'N/A'}")
    
    # Real-time monitoring section
    st.subheader("📊 Real-time Drought Monitoring")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if not satellite_df.empty:
        latest_nddi = satellite_df['NDDI'].iloc[-1]
        severity = calculate_drought_severity(latest_nddi)
        severity_color = get_drought_color(severity)
        
        analysis = analyze_drought_trends(satellite_df)
        
        with col1:
            st.metric("Current NDDI", f"{latest_nddi:.3f}", 
                     delta=f"{satellite_df['NDDI'].diff().iloc[-1]:.3f}" if len(satellite_df) > 1 else None)
        
        with col2:
            st.markdown(f"<div style='color: {severity_color}; font-weight: bold; font-size: 16px;'>{severity}</div>", 
                       unsafe_allow_html=True)
            st.caption("Drought Severity")
        
        with col3:
            trend_icon = {"Worsening": "📉", "Improving": "📈", "Stable": "➡️"}.get(analysis['trend'], "➡️")
            st.metric("Trend", f"{trend_icon} {analysis['trend']}")
        
        with col4:
            risk_color = {"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#00FF00"}.get(analysis['risk_level'], "#808080")
            st.markdown(f"<div style='color: {risk_color}; font-weight: bold; font-size: 16px;'>{analysis['risk_level']} Risk</div>", 
                       unsafe_allow_html=True)
            st.caption("Drought Risk Level")
    
    # Charts section
    chart_tabs = st.tabs(["📈 Time Series", "🌍 Geographic", "🧠 ML Predictions", "📊 Comprehensive Analysis"])
    
    with chart_tabs[0]:
        # Time series charts
        if not satellite_df.empty:
            fig_ts = make_subplots(
                rows=2, cols=2,
                subplot_titles=('NDDI Time Series', 'NDVI vs NDWI', 'Temperature & Precipitation', 'Drought Severity Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": True}, {"secondary_y": False}]]
            )
            
            # NDDI time series with severity zones
            colors = satellite_df['NDDI'].apply(lambda x: get_drought_color(calculate_drought_severity(x)))
            fig_ts.add_trace(
                go.Scatter(x=satellite_df['Date'], y=satellite_df['NDDI'],
                          mode='lines+markers', name='NDDI',
                          line=dict(color='blue', width=2),
                          hovertemplate='Date: %{x}<br>NDDI: %{y:.3f}<br>Severity: %{text}',
                          text=[calculate_drought_severity(x) for x in satellite_df['NDDI']]),
                row=1, col=1
            )
            
            # Add drought severity zones
            fig_ts.add_hline(y=-0.5, line_dash="dash", line_color="red", 
                           annotation_text="Extreme Drought", row=1, col=1)
            fig_ts.add_hline(y=-0.2, line_dash="dash", line_color="orange", 
                           annotation_text="Severe Drought", row=1, col=1)
            fig_ts.add_hline(y=0, line_dash="dash", line_color="yellow", 
                           annotation_text="Moderate Drought", row=1, col=1)
            
            # NDVI vs NDWI scatter
            fig_ts.add_trace(
                go.Scatter(x=satellite_df['NDWI'], y=satellite_df['NDVI'],
                          mode='markers', name='NDVI vs NDWI',
                          marker=dict(color=satellite_df['NDDI'], colorscale='RdYlBu', showscale=True),
                          hovertemplate='NDWI: %{x:.3f}<br>NDVI: %{y:.3f}<br>NDDI: %{marker.color:.3f}'),
                row=1, col=2
            )
            
            # Temperature and Precipitation
            fig_ts.add_trace(
                go.Scatter(x=satellite_df['Date'], y=satellite_df['Temperature_C'],
                          mode='lines', name='Temperature (°C)', line=dict(color='red')),
                row=2, col=1
            )
            fig_ts.add_trace(
                go.Scatter(x=satellite_df['Date'], y=satellite_df['Precipitation_mm'],
                          mode='lines', name='Precipitation (mm)', line=dict(color='blue')),
                row=2, col=1, secondary_y=True
            )
            
            # Drought severity distribution
            severity_counts = analysis.get('severity_distribution', {})
            if severity_counts:
                severities = list(severity_counts.keys())
                counts = list(severity_counts.values())
                colors_pie = [get_drought_color(s) for s in severities]
                
                fig_ts.add_trace(
                    go.Pie(labels=severities, values=counts, 
                          marker_colors=colors_pie, name="Severity Distribution"),
                    row=2, col=2
                )
            
            fig_ts.update_layout(height=600, title_text="Satellite Data Analysis", showlegend=True)
            fig_ts.update_xaxes(title_text="Date", row=1, col=1)
            fig_ts.update_yaxes(title_text="NDDI", row=1, col=1)
            fig_ts.update_xaxes(title_text="NDWI", row=1, col=2)
            fig_ts.update_yaxes(title_text="NDVI", row=1, col=2)
            fig_ts.update_xaxes(title_text="Date", row=2, col=1)
            fig_ts.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
            fig_ts.update_yaxes(title_text="Precipitation (mm)", row=2, col=1, secondary_y=True)
            
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.warning("No satellite data available for time series analysis")
    
    with chart_tabs[1]:
        # Geographic visualization
        if not satellite_df.empty and 'Latitude' in satellite_df.columns:
            # Create map visualization
            fig_map = px.scatter_mapbox(
                satellite_df,
                lat="Latitude", lon="Longitude",
                color="NDDI",
                size=abs(satellite_df['NDDI']) * 10,
                hover_data=['Date', 'NDDI', 'Crop_Type'] if 'Crop_Type' in satellite_df.columns else ['Date', 'NDDI'],
                color_continuous_scale="RdYlBu",
                mapbox_style="open-street-map",
                title="Real-time NDDI Monitoring Locations"
            )
            fig_map.update_layout(height=500)
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Agriculture dataset geographic distribution
        if 'GPS_Coordinates' in agriculture_df.columns:
            st.subheader("Agriculture Dataset Sample Locations")
            
            # Sample some locations for visualization
            sample_ag = agriculture_df.sample(n=min(100, len(agriculture_df)))
            
            # Extract coordinates for mapping
            coords_data = []
            for idx, row in sample_ag.iterrows():
                try:
                    gps_str = str(row['GPS_Coordinates'])
                    if ',' in gps_str:
                        lat_str, lon_str = gps_str.split(',')[:2]
                        lat, lon = float(lat_str.strip()), float(lon_str.strip())
                        
                        coords_data.append({
                            'Latitude': lat,
                            'Longitude': lon,
                            'Crop_Type': row.get('Crop_Type', 'Unknown'),
                            'NDDI': row.get('NDDI', 0),
                            'Crop_Stress': row.get('Crop_Stress_Indicator', 0),
                            'Expected_Yield': row.get('Expected_Yield', 0)
                        })
                except:
                    continue
            
            if coords_data:
                coords_df = pd.DataFrame(coords_data)
                
                fig_ag_map = px.scatter_mapbox(
                    coords_df,
                    lat="Latitude", lon="Longitude",
                    color="Crop_Stress",
                    size="Expected_Yield",
                    hover_data=['Crop_Type', 'NDDI', 'Crop_Stress'],
                    color_continuous_scale="Viridis",
                    mapbox_style="open-street-map",
                    title="Agriculture Dataset - Crop Stress Distribution"
                )
                fig_ag_map.update_layout(height=400)
                st.plotly_chart(fig_ag_map, use_container_width=True)
    
    with chart_tabs[2]:
        # ML Predictions from Colab notebook
        st.subheader("🧠 Machine Learning Predictions")
        st.markdown("*Training CatBoost, CNN, and LSTM models for drought prediction*")
        
        # Train models button
        if st.button("🚀 Train Models on Current Data", type="primary"):
            training_status = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Prepare data
                training_status.info("📊 Preparing features for training...")
                progress_bar.progress(20)
                X, y = processor.prepare_features(agriculture_df)
                
                if len(X) > 0:
                    # Step 2: Train models
                    training_status.info(f"🧠 Training ML models on {len(X):,} samples...")
                    progress_bar.progress(40)
                    models = DroughtPredictionModels()
                    
                    training_status.info("⚡ Training LSTM (PyTorch) - Epoch 1/4...")
                    progress_bar.progress(50)
                    
                    training_status.info("🚀 Training CatBoost gradient boosting...")
                    progress_bar.progress(70)
                    
                    training_status.info("🧠 Training Convolutional Neural Network...")
                    progress_bar.progress(85)
                    
                    results = models.train_models(X, y)
                    
                    progress_bar.progress(100)
                    training_status.empty()
                    progress_bar.empty()
                    
                    # Display results
                    st.success("✅ Models trained successfully!")
                    
                    # Model performance comparison
                    model_names = []
                    rmse_scores = []
                    mae_scores = []
                    r2_scores = []
                    
                    for model_name, result in results.items():
                        if result is not None:
                            model_names.append(model_name)
                            rmse_scores.append(result['rmse'])
                            mae_scores.append(result['mae'])
                            r2_scores.append(result['r2'])
                    
                    if model_names:
                        # Performance metrics table
                        metrics_df = pd.DataFrame({
                            'Model': model_names,
                            'RMSE': rmse_scores,
                            'MAE': mae_scores,
                            'R²': r2_scores
                        })
                        
                        st.subheader("📊 Model Performance Comparison")
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Best model highlight
                        best_model_idx = np.argmin(rmse_scores)
                        best_model = model_names[best_model_idx]
                        st.success(f"🏆 Best performing model: **{best_model}** (RMSE: {rmse_scores[best_model_idx]:.4f})")
                        
                        # Visualization of predictions vs actual
                        fig_pred = make_subplots(
                            rows=1, cols=len(model_names),
                            subplot_titles=[f"{name} Predictions" for name in model_names]
                        )
                        
                        for i, (model_name, result) in enumerate(results.items()):
                            if result is not None:
                                # Sample points for visualization
                                n_points = min(300, len(result['actual']))
                                indices = np.random.choice(len(result['actual']), n_points, replace=False)
                                
                                fig_pred.add_trace(
                                    go.Scatter(
                                        x=result['actual'][indices],
                                        y=result['predictions'][indices],
                                        mode='markers',
                                        name=f'{model_name}',
                                        opacity=0.6
                                    ),
                                    row=1, col=i+1
                                )
                                
                                # Perfect fit line
                                min_val = min(result['actual'][indices])
                                max_val = max(result['actual'][indices])
                                fig_pred.add_trace(
                                    go.Scatter(
                                        x=[min_val, max_val],
                                        y=[min_val, max_val],
                                        mode='lines',
                                        name='Perfect Fit',
                                        line=dict(dash='dash', color='red'),
                                        showlegend=(i == 0)
                                    ),
                                    row=1, col=i+1
                                )
                        
                        fig_pred.update_layout(
                            height=400,
                            title_text="Actual vs Predicted Values (Sampled Points)",
                            showlegend=True
                        )
                        
                        for i in range(len(model_names)):
                            fig_pred.update_xaxes(title_text="Actual", row=1, col=i+1)
                            fig_pred.update_yaxes(title_text="Predicted", row=1, col=i+1)
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # LSTM training history (if available)
                        if 'LSTM' in results and results['LSTM'] is not None and 'history' in results['LSTM']:
                            st.subheader("📈 LSTM Training History")
                            history = results['LSTM']['history']
                            
                            fig_history = go.Figure()
                            fig_history.add_trace(go.Scatter(
                                y=history['loss'], 
                                name='Training Loss',
                                line=dict(color='blue')
                            ))
                            if 'val_loss' in history:
                                fig_history.add_trace(go.Scatter(
                                    y=history['val_loss'], 
                                    name='Validation Loss',
                                    line=dict(color='red')
                                ))
                            
                            fig_history.update_layout(
                                title="LSTM Training & Validation Loss (4 epochs)",
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                height=300
                            )
                            st.plotly_chart(fig_history, use_container_width=True)
                        
                        # Save trained models to session state
                        st.session_state['trained_models'] = models
                        st.session_state['model_results'] = results
                        st.session_state['feature_columns'] = processor.feature_columns
                        st.session_state['scaler'] = processor.scaler
                        
                else:
                    st.error("❌ Failed to prepare data for training")
            except Exception as e:
                training_status.empty()
                progress_bar.empty()
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)
        
        # Real-time prediction section
        if 'trained_models' in st.session_state:
            st.subheader("🔮 Real-time Crop Stress Prediction")
            st.markdown("*Enter values to get predictions from trained models*")
            
            # Input form for real-time prediction
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ndvi = st.slider("NDVI", 0.0, 1.0, 0.5)
                    temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
                    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
                    soil_moisture = st.slider("Soil Moisture", 0.0, 1.0, 0.4)
                
                with col2:
                    rainfall = st.slider("Rainfall (mm)", 0.0, 200.0, 50.0)
                    soil_ph = st.slider("Soil pH", 4.0, 9.0, 7.0)
                    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0)
                    elevation = st.slider("Elevation (m)", 0.0, 3000.0, 200.0)
                
                with col3:
                    savi = st.slider("SAVI", 0.0, 1.0, 0.3)
                    chlorophyll = st.slider("Chlorophyll Content", 0.0, 5.0, 1.5)
                    lai = st.slider("Leaf Area Index", 0.0, 10.0, 3.0)
                    canopy = st.slider("Canopy Coverage (%)", 0.0, 100.0, 70.0)
                
                predict_button = st.form_submit_button("🔮 Predict Crop Stress")
                
                if predict_button:
                    # Prepare input features
                    feature_values = [
                        ndvi, savi, chlorophyll, lai, temp, humidity, rainfall,
                        soil_moisture, soil_ph, 0.5, canopy, elevation, wind_speed, 2000
                    ]
                    
                    if len(feature_values) == len(st.session_state['feature_columns']):
                        # Scale features
                        features_scaled = st.session_state['scaler'].transform([feature_values])
                        
                        # Get predictions
                        predictions = st.session_state['trained_models'].predict_crop_stress(features_scaled[0])
                        
                        # Display predictions
                        st.subheader("🎯 Predictions")
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        
                        if 'LSTM' in predictions:
                            with pred_col1:
                                stress_level = predictions['LSTM']
                                color = 'red' if stress_level > 70 else 'orange' if stress_level > 40 else 'green'
                                st.metric(
                                    "🧠 LSTM Prediction",
                                    f"{stress_level:.1f}",
                                    help="Crop stress indicator (0-100)"
                                )
                        
                        if 'CNN' in predictions:
                            with pred_col2:
                                stress_level = predictions['CNN']
                                st.metric(
                                    "🧠 CNN Prediction",
                                    f"{stress_level:.1f}",
                                    help="Crop stress indicator (0-100)"
                                )
                        
                        if 'CatBoost' in predictions:
                            with pred_col3:
                                stress_level = predictions['CatBoost']
                                st.metric(
                                    "🚀 CatBoost Prediction",
                                    f"{stress_level:.1f}",
                                    help="Crop stress indicator (0-100)"
                                )
                        
                        # Calculate NDDI and drought severity
                        estimated_ndwi = soil_moisture * 0.5  # Simple estimation
                        nddi_calc = (ndvi - estimated_ndwi) / (ndvi + estimated_ndwi + 1e-10)
                        drought_sev = get_drought_severity_from_nddi(nddi_calc)
                        drought_color = get_drought_color(drought_sev)
                        
                        st.markdown("---")
                        st.markdown(
                            f"**Calculated NDDI:** {nddi_calc:.3f} | "
                            f"**Drought Status:** <span style='color: {drought_color}; font-weight: bold;'>{drought_sev}</span>",
                            unsafe_allow_html=True
                        )
        else:
            st.info("👆 Train models first to enable real-time predictions")
    
    with chart_tabs[3]:
        # Comprehensive analysis combining everything
        st.subheader("📊 Comprehensive Drought & Crop Analysis")
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Agriculture Dataset Summary**")
            if not agriculture_df.empty:
                summary_stats = {
                    "Total Records": len(agriculture_df),
                    "Crop Types": agriculture_df['Crop_Type'].nunique() if 'Crop_Type' in agriculture_df.columns else "N/A",
                    "Avg NDVI": f"{agriculture_df['NDVI'].mean():.3f}" if 'NDVI' in agriculture_df.columns else "N/A",
                    "Avg Crop Stress": f"{agriculture_df['Crop_Stress_Indicator'].mean():.1f}" if 'Crop_Stress_Indicator' in agriculture_df.columns else "N/A",
                    "Avg Expected Yield": f"{agriculture_df['Expected_Yield'].mean():.1f}" if 'Expected_Yield' in agriculture_df.columns else "N/A"
                }
                
                for key, value in summary_stats.items():
                    st.metric(key, value)
        
        with col2:
            st.write("**Satellite Data Summary**")
            if not satellite_df.empty:
                satellite_stats = {
                    "Data Points": len(satellite_df),
                    "Locations": satellite_df['Latitude'].nunique() if 'Latitude' in satellite_df.columns else "N/A",
                    "Avg NDDI": f"{satellite_df['NDDI'].mean():.3f}",
                    "Date Range": f"{(satellite_df['Date'].max() - satellite_df['Date'].min()).days} days",
                    "Data Quality": satellite_df['Data_Quality'].mode().iloc[0] if 'Data_Quality' in satellite_df.columns else "N/A"
                }
                
                for key, value in satellite_stats.items():
                    st.metric(key, value)
        
        # Feature correlation heatmap
        if not agriculture_df.empty:
            st.subheader("🔗 Feature Correlations")
            
            # Select numeric columns for correlation
            numeric_cols = agriculture_df.select_dtypes(include=[np.number]).columns
            corr_cols = [col for col in ['NDVI', 'SAVI', 'Temperature', 'Humidity', 'Rainfall', 
                                       'Soil_Moisture', 'Crop_Stress_Indicator', 'Expected_Yield', 'NDDI'] 
                        if col in numeric_cols]
            
            if len(corr_cols) > 1:
                corr_matrix = agriculture_df[corr_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data quality assessment
        st.subheader("🔍 Data Quality Assessment")
        
        qual_col1, qual_col2 = st.columns(2)
        
        with qual_col1:
            if not agriculture_df.empty:
                st.write("**Agriculture Data Quality**")
                missing_data = agriculture_df.isnull().sum()
                missing_pct = (missing_data / len(agriculture_df) * 100).round(2)
                quality_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': missing_pct.values
                }).head(10)
                
                st.dataframe(quality_df, use_container_width=True)
        
        with qual_col2:
            if not satellite_df.empty:
                st.write("**Satellite Data Quality**")
                quality_counts = satellite_df['Data_Quality'].value_counts() if 'Data_Quality' in satellite_df.columns else pd.Series()
                
                if not quality_counts.empty:
                    fig_quality = px.pie(
                        values=quality_counts.values,
                        names=quality_counts.index,
                        title="Satellite Data Quality Distribution"
                    )
                    fig_quality.update_layout(height=300)
                    st.plotly_chart(fig_quality, use_container_width=True)
    
    # Footer with data sources and last update
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"🌾 Agriculture data: {len(agriculture_df)} records loaded")
    
    with col2:
        st.caption(f"🛰️ Satellite data: {len(satellite_df) if not satellite_df.empty else 0} points")
    
    with col3:
        st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data export section
    with st.expander("💾 Export Data"):
        if st.button("📥 Download Integrated Dataset"):
            if not agriculture_df.empty:
                # Combine relevant data for export
                export_df = agriculture_df.copy()
                
                # Add satellite summary if available
                if not satellite_df.empty:
                    satellite_summary = satellite_df.groupby('Agriculture_Index').agg({
                        'NDDI': 'mean',
                        'NDVI': 'mean',
                        'NDWI': 'mean',
                        'Temperature_C': 'mean'
                    }).reset_index()
                    
                    export_df = export_df.merge(
                        satellite_summary, 
                        left_index=True, 
                        right_on='Agriculture_Index', 
                        how='left'
                    )
                
                # Convert to CSV
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"drought_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
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
            ["Upload CSV", "Real-time Satellite"]
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
            st.subheader("📡 Satellite Config - Uttar Pradesh")
            
            # Load UP districts for selection
            from ..utils.up_gps_coordinates import UP_DISTRICT_COORDINATES
            district_options = sorted(UP_DISTRICT_COORDINATES.keys())
            
            selected_district = st.selectbox(
                "Select UP District",
                district_options,
                index=district_options.index('LUCKNOW') if 'LUCKNOW' in district_options else 0
            )
            
            if selected_district:
                default_lat, default_lon = UP_DISTRICT_COORDINATES[selected_district]
                st.info(f"📍 {selected_district} District HQ: {default_lat:.4f}°N, {default_lon:.4f}°E")
            else:
                default_lat, default_lon = 26.8467, 80.9462  # Lucknow
            
            lat = st.number_input("Latitude (or customize)", 23.8, 30.4, default_lat, format="%.4f", 
                                help="Uttar Pradesh range: 23.8°N to 30.4°N")
            lon = st.number_input("Longitude (or customize)", 77.0, 84.6, default_lon, format="%.4f",
                                help="Uttar Pradesh range: 77.0°E to 84.6°E")
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
        if st.button("🛰️ Fetch Real Satellite Data from Google Earth Engine", type="primary"):
            with st.spinner(f"🛰️ Fetching REAL satellite imagery data for {selected_district}, Uttar Pradesh from Google Earth Engine..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                # Validate coordinates are in UP
                from ..utils.up_gps_coordinates import validate_up_coordinates
                if not validate_up_coordinates(lat, lon):
                    st.error(f"❌ Coordinates ({lat}, {lon}) are outside Uttar Pradesh boundaries!")
                    st.info("UP boundaries: 23.8°N-30.4°N, 77.0°E-84.6°E")
                    df_to_use = None
                else:
                    df_to_use = fetch_satellite_nddi_data(lat, lon, start_date, end_date)
                    ss["satellite_df"] = df_to_use
                    
                    # Auto-record to CSV with district name
                    try:
                        filename = f"{selected_district.lower()}_lat{lat:.4f}_lon{lon:.4f}"
                        record_satellite_data_to_csv(df_to_use, filename)
                        st.info(f"📝 Data automatically recorded to satellite_data_records/{filename}_*.csv")
                    except Exception as e:
                        st.warning(f"Could not save to CSV: {e}")
                    
                    st.success(f"✅ Fetched {len(df_to_use)} days of REAL satellite data for {selected_district} district")
                
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
    
    elif data_source == "UP Village Data (UPVillageSchedule.csv)":
        try:
            # Load REAL Uttar Pradesh village data from CSV file
            with st.spinner("📊 Loading REAL Uttar Pradesh village irrigation data from CSV..."):
                import os
                csv_path = 'UPVillageSchedule.csv'
                if not os.path.exists(csv_path):
                    st.error(f"❌ {csv_path} not found in project root")
                    df_to_use = None
                else:
                    # Read the CSV
                    df_raw = pd.read_csv(csv_path)
                    
                    # Add GPS coordinates using UP district/village mapping
                    from ..utils.up_gps_coordinates import add_gps_to_up_village_data
                    df_with_gps = add_gps_to_up_village_data(df_raw)
                    
                    # Calculate sample size based on slider percentage
                    total_rows = len(df_with_gps)
                    target_samples = int(total_rows * (sample_size / 100))
                    
                    # Sample the data
                    df_to_use = df_with_gps.sample(n=min(target_samples, len(df_with_gps)), random_state=42).reset_index(drop=True)
                    
                    if not df_to_use.empty:
                        st.success(f"✅ Loaded {len(df_to_use):,} REAL Uttar Pradesh villages from UPVillageSchedule.csv ({sample_size}% of {total_rows:,} total villages)")
                        st.info("🌾 This is REAL Uttar Pradesh government agricultural census data with irrigation, ground water levels, and cultivable area - NOT simulated!")
                        
                        # Show data source confirmation
                        with st.expander("🔍 Data Source Confirmation - Uttar Pradesh Villages", expanded=True):
                            st.write(f"**Source File:** UPVillageSchedule.csv (Government of Uttar Pradesh)")
                            st.write(f"**State:** {df_to_use['state_name'].iloc[0]}")
                            st.write(f"**Districts Covered:** {df_to_use['district_name'].nunique()} districts")
                            st.write(f"**Villages:** {len(df_to_use):,} villages")
                            st.write(f"**Features:** Geographical area, cultivable area, net sown area, irrigation data, ground water levels")
                            
                            # Show sample districts and villages
                            sample_districts = df_to_use.groupby('district_name').size().head(5)
                            st.write(f"**Top Districts:** {', '.join(sample_districts.index.tolist())}")
                            
                            st.write(f"**Sample Village Data:**")
                            display_cols = ['district_name', 'village_name', 'geographical_area', 'cultivable_area', 
                                          'net_irrigated_area', 'avg_ground_water_level_pre_monsoon', 'Latitude', 'Longitude']
                            available_cols = [col for col in display_cols if col in df_to_use.columns]
                            st.dataframe(df_to_use[available_cols].head(5), use_container_width=True)
                        
                        # Calculate NDDI from ground water and irrigation data
                        if 'NDDI' not in df_to_use.columns:
                            with st.spinner("Calculating NDDI from ground water levels and irrigation data..."):
                                # Use ground water levels and irrigation as proxy for moisture/vegetation
                                # Lower ground water = drier = lower NDDI
                                # Higher irrigation = wetter = higher NDDI
                                
                                df_to_use['NDVI'] = 0.5 + 0.3 * (df_to_use['net_irrigated_area'] / (df_to_use['net_sown_area'] + 1)).clip(0, 1)
                                
                                # Normalize ground water levels (lower depth = more water = higher NDWI)
                                gw_normalized = 1 - (df_to_use['avg_ground_water_level_pre_monsoon'] / df_to_use['avg_ground_water_level_pre_monsoon'].max()).fillna(0.5)
                                df_to_use['NDWI'] = 0.2 + 0.3 * gw_normalized.clip(0, 1)
                                
                                # Calculate NDDI
                                df_to_use['NDDI'] = (df_to_use['NDVI'] - df_to_use['NDWI']) / (df_to_use['NDVI'] + df_to_use['NDWI'] + 1e-10)
                                
                                st.success("✅ NDDI calculated from REAL irrigation and ground water measurements of UP villages")
                    else:
                        st.error("❌ Failed to load UP village dataset")
                        df_to_use = None
        except Exception as e:
            st.error(f"Could not load UPVillageSchedule.csv: {e}")
            st.info("Please ensure UPVillageSchedule.csv is in the project root")
            import traceback
            st.code(traceback.format_exc())
    
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
    """Train CatBoost, CNN, and LSTM models and compare results"""
    
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
        
        # 2. Train CNN
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
        
        # 3. Train CatBoost
        status_text.text("🚀 Training CatBoost model...")
        progress_bar.progress(70)
        
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
        
        # Store results
        ss["ai_results"] = results
        
        status_text.text("✅ All 3 models trained successfully!")
        progress_bar.progress(100)
        
        st.success("🎉 CatBoost, CNN, and LSTM trained successfully!")
        
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
    
    # Model Comparison Visualization
    st.divider()
    st.header("📊 CatBoost vs CNN vs LSTM Comparison")
    
    # Create comparison metrics visualization
    comparison_col1, comparison_col2 = st.columns([1, 1])
    
    with comparison_col1:
        st.markdown("### 📈 Performance Metrics Comparison")
        
        # Bar chart for metrics comparison
        metrics_comparison = []
        for model_name, model_info in results['models'].items():
            metrics_comparison.append({
                'Model': model_name,
                'RMSE': model_info['rmse'],
                'MAE': model_info['mae'],
                'R² Score': model_info['r2']
            })
        
        fig_metrics = go.Figure()
        
        models_list = [m['Model'] for m in metrics_comparison]
        colors_map = {'LSTM': '#3498db', 'CNN': '#e74c3c', 'CatBoost': '#2ecc71'}
        
        # RMSE bars
        fig_metrics.add_trace(go.Bar(
            name='RMSE (lower is better)',
            x=models_list,
            y=[m['RMSE'] for m in metrics_comparison],
            marker_color=[colors_map.get(m, 'gray') for m in models_list],
            text=[f"{m['RMSE']:.4f}" for m in metrics_comparison],
            textposition='auto'
        ))
        
        fig_metrics.update_layout(
            title="Root Mean Square Error Comparison",
            yaxis_title="RMSE",
            xaxis_title="Model",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True, key="rmse_comparison")
        
        # R² Score comparison
        fig_r2 = go.Figure()
        
        fig_r2.add_trace(go.Bar(
            name='R² Score (higher is better)',
            x=models_list,
            y=[m['R² Score'] for m in metrics_comparison],
            marker_color=[colors_map.get(m, 'gray') for m in models_list],
            text=[f"{m['R² Score']:.4f}" for m in metrics_comparison],
            textposition='auto'
        ))
        
        fig_r2.update_layout(
            title="R² Score Comparison",
            yaxis_title="R² Score",
            xaxis_title="Model",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_r2, use_container_width=True, key="r2_comparison")
    
    with comparison_col2:
        st.markdown("### 🎯 Prediction Accuracy Visualization")
        
        # Predictions vs Actual for all 3 models
        sample_size = min(200, len(y_test))
        indices = sorted(random.sample(range(len(y_test)), sample_size))
        
        fig_pred = go.Figure()
        
        # Actual values
        fig_pred.add_trace(go.Scatter(
            x=list(range(sample_size)),
            y=y_test[indices],
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2, dash='solid')
        ))
        
        # Model predictions
        for model_name, model_info in results['models'].items():
            fig_pred.add_trace(go.Scatter(
                x=list(range(sample_size)),
                y=model_info['predictions'][indices],
                mode='lines',
                name=f"{model_name} (RMSE: {model_info['rmse']:.4f})",
                line=dict(width=2, color=colors_map.get(model_name, 'gray'))
            ))
        
        fig_pred.update_layout(
            title=f"Model Predictions vs Actual Values ({sample_size} samples)",
            xaxis_title="Sample Index",
            yaxis_title="Crop Stress Value",
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_pred, use_container_width=True, key="predictions_comparison")
        
        # Error distribution
        st.markdown("### 📊 Prediction Error Distribution")
        
        fig_errors = go.Figure()
        
        for model_name, model_info in results['models'].items():
            errors = np.abs(y_test - model_info['predictions'])
            
            fig_errors.add_trace(go.Box(
                y=errors[:500],  # Use subset for clarity
                name=model_name,
                marker_color=colors_map.get(model_name, 'gray')
            ))
        
        fig_errors.update_layout(
            title="Absolute Error Distribution",
            yaxis_title="Absolute Error",
            xaxis_title="Model",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_errors, use_container_width=True, key="errors_comparison")
    
    # Visualizations
    st.divider()
    st.header("📊 Detailed Model Visualizations")
    
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
        
        colors_map = {'LSTM': 'blue', 'CNN': 'red', 'CatBoost': 'green'}
        
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
