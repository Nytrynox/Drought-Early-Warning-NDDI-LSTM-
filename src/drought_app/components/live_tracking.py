"""
Live Real-time Satellite Tracking and Visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def create_live_tracking_dashboard(lat, lon, days_back=30):
    """
    Create a live tracking dashboard with real-time satellite data
    """
    
    st.markdown("### 🛰️ Live Satellite Tracking Dashboard")
    
    # Display tracking info
    col1, col2, col3 = st.columns(3)
    col1.metric("📍 Tracking Location", f"Lat: {lat:.4f}")
    col2.metric("🌍 Longitude", f"Lon: {lon:.4f}")
    col3.metric("⏱️ Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    st.divider()
    
    return True


def create_realtime_nddi_chart(df):
    """
    Create real-time NDDI chart with live updates
    """
    
    fig = go.Figure()
    
    # NDDI line
    if 'Date' in df.columns and 'NDDI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['NDDI'],
            mode='lines+markers',
            name='NDDI',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        ))
        
        # Add threshold lines
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", 
                     annotation_text="Extreme Drought")
        fig.add_hline(y=-0.2, line_dash="dash", line_color="orange",
                     annotation_text="Moderate Drought")
        fig.add_hline(y=0, line_dash="dash", line_color="blue",
                     annotation_text="Normal")
        
        fig.update_layout(
            title="Real-time NDDI Tracking",
            xaxis_title="Date",
            yaxis_title="NDDI Value",
            height=400,
            hovermode='x unified'
        )
    
    return fig


def create_ndvi_ndwi_scatter(df):
    """
    Create NDVI vs NDWI scatter plot colored by NDDI
    """
    
    if all(col in df.columns for col in ['NDVI', 'NDWI', 'NDDI']):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['NDVI'],
            y=df['NDWI'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['NDDI'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="NDDI"),
                line=dict(width=1, color='white')
            ),
            text=df.get('Date', range(len(df))),
            hovertemplate='<b>NDVI:</b> %{x:.3f}<br>' +
                         '<b>NDWI:</b> %{y:.3f}<br>' +
                         '<b>Date:</b> %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="NDVI vs NDWI Relationship (colored by NDDI)",
            xaxis_title="NDVI (Vegetation)",
            yaxis_title="NDWI (Water Content)",
            height=500
        )
        
        return fig
    
    return None


def create_drought_timeline(df):
    """
    Create drought severity timeline
    """
    
    if 'Date' in df.columns and 'NDDI' in df.columns:
        # Classify drought severity
        df_copy = df.copy()
        
        def classify_drought(nddi):
            if nddi < -0.5:
                return "Extreme Drought"
            elif nddi < -0.2:
                return "Severe Drought"
            elif nddi < 0:
                return "Moderate Drought"
            elif nddi < 0.2:
                return "Normal"
            else:
                return "Wet"
        
        df_copy['Severity'] = df_copy['NDDI'].apply(classify_drought)
        
        # Color mapping
        color_map = {
            "Extreme Drought": "#8B0000",
            "Severe Drought": "#FF4500",
            "Moderate Drought": "#FFA500",
            "Normal": "#90EE90",
            "Wet": "#4169E1"
        }
        
        df_copy['Color'] = df_copy['Severity'].map(color_map)
        
        fig = go.Figure()
        
        for severity in df_copy['Severity'].unique():
            df_severity = df_copy[df_copy['Severity'] == severity]
            fig.add_trace(go.Scatter(
                x=df_severity['Date'],
                y=df_severity['NDDI'],
                mode='markers',
                name=severity,
                marker=dict(
                    size=8,
                    color=color_map.get(severity, 'gray')
                )
            ))
        
        fig.update_layout(
            title="Drought Severity Timeline",
            xaxis_title="Date",
            yaxis_title="NDDI",
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    return None


def create_environmental_dashboard(df):
    """
    Create comprehensive environmental monitoring dashboard
    """
    
    # Check available columns
    has_temp = 'Temperature' in df.columns or 'Temperature_C' in df.columns
    has_precip = 'Precipitation' in df.columns or 'Precipitation_mm' in df.columns
    has_soil = 'Soil_Moisture' in df.columns
    
    if has_temp or has_precip or has_soil:
        from plotly.subplots import make_subplots
        
        rows = sum([has_temp, has_precip, has_soil])
        
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=[
                title for title, has in [
                    ("Temperature (°C)", has_temp),
                    ("Precipitation (mm)", has_precip),
                    ("Soil Moisture (%)", has_soil)
                ] if has
            ],
            vertical_spacing=0.1
        )
        
        row = 1
        x_data = df['Date'] if 'Date' in df.columns else range(len(df))
        
        if has_temp:
            temp_col = 'Temperature_C' if 'Temperature_C' in df.columns else 'Temperature'
            fig.add_trace(
                go.Scatter(x=x_data, y=df[temp_col], mode='lines', 
                          name='Temperature', line=dict(color='red')),
                row=row, col=1
            )
            row += 1
        
        if has_precip:
            precip_col = 'Precipitation_mm' if 'Precipitation_mm' in df.columns else 'Precipitation'
            fig.add_trace(
                go.Bar(x=x_data, y=df[precip_col], name='Precipitation',
                      marker=dict(color='blue')),
                row=row, col=1
            )
            row += 1
        
        if has_soil:
            fig.add_trace(
                go.Scatter(x=x_data, y=df['Soil_Moisture'], mode='lines',
                          name='Soil Moisture', line=dict(color='brown')),
                row=row, col=1
            )
        
        fig.update_layout(
            height=300 * rows,
            showlegend=True,
            title_text="Environmental Parameters Monitoring"
        )
        
        return fig
    
    return None
