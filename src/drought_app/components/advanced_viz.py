"""
Advanced Visualizations: Heatmaps, Time Series, Correlation Matrices
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


def create_nddi_heatmap(df, date_col='Date', value_col='NDDI'):
    """
    Create an interactive temporal heatmap for NDDI values
    """
    
    # Prepare data
    df_copy = df.copy()
    if date_col in df_copy.columns:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy['Year'] = df_copy[date_col].dt.year
        df_copy['Month'] = df_copy[date_col].dt.month
        df_copy['Day'] = df_copy[date_col].dt.day
        
        # Create pivot table for heatmap
        pivot_data = df_copy.pivot_table(
            values=value_col,
            index='Month',
            columns='Day',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot_data)],
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title=value_col),
            hoverongaps=False,
            hovertemplate='Month: %{y}<br>Day: %{x}<br>NDDI: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{value_col} Heatmap - Daily Patterns by Month',
            xaxis_title='Day of Month',
            yaxis_title='Month',
            height=500
        )
        
        return fig
    else:
        st.warning("Date column not found for temporal heatmap")
        return None


def create_spatial_heatmap(df, lat_col='Latitude', lon_col='Longitude', value_col='NDDI'):
    """
    Create a spatial density heatmap
    """
    
    fig = go.Figure(go.Densitymapbox(
        lat=df[lat_col],
        lon=df[lon_col],
        z=df[value_col],
        radius=30,
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title=value_col),
        hovertemplate='Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<br>Value: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center_lat=df[lat_col].mean(),
        mapbox_center_lon=df[lon_col].mean(),
        mapbox_zoom=8,
        height=600,
        title=f'{value_col} Spatial Heatmap'
    )
    
    return fig


def create_correlation_heatmap(df):
    """
    Create correlation matrix heatmap for all numeric features
    """
    
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title='Correlation'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=800
    )
    
    return fig


def create_multi_metric_dashboard(df):
    """
    Create a comprehensive dashboard with multiple visualizations
    """
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('NDDI Time Series', 'NDVI vs NDWI', 
                        'Temperature Trend', 'Precipitation Pattern'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Plot 1: NDDI Time Series
    if 'Date' in df.columns and 'NDDI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['NDDI'], mode='lines', name='NDDI', line=dict(color='green')),
            row=1, col=1
        )
    
    # Plot 2: NDVI vs NDWI
    if 'NDVI' in df.columns and 'NDWI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['NDVI'], y=df['NDWI'], mode='markers', name='NDVI vs NDWI',
                      marker=dict(color=df.get('NDDI', df['NDVI']), colorscale='Viridis', showscale=True)),
            row=1, col=2
        )
    
    # Plot 3: Temperature
    if 'Date' in df.columns and 'Temperature_C' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Temperature_C'], mode='lines', name='Temperature',
                      line=dict(color='red')),
            row=2, col=1
        )
    
    # Plot 4: Precipitation
    if 'Date' in df.columns and 'Precipitation_mm' in df.columns:
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Precipitation_mm'], name='Precipitation',
                  marker=dict(color='blue')),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title_text="Multi-Metric Environmental Dashboard")
    
    return fig


def create_3d_surface_plot(df, x_col='NDVI', y_col='NDWI', z_col='NDDI'):
    """
    Create 3D surface plot for relationship visualization
    """
    
    # Create grid
    xi = np.linspace(df[x_col].min(), df[x_col].max(), 50)
    yi = np.linspace(df[y_col].min(), df[y_col].max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate Z values
    from scipy.interpolate import griddata
    points = df[[x_col, y_col]].values
    values = df[z_col].values
    Zi = griddata(points, values, (Xi, Yi), method='cubic')
    
    # Create 3D surface
    fig = go.Figure(data=[go.Surface(x=Xi, y=Yi, z=Zi, colorscale='RdYlGn')])
    
    fig.update_layout(
        title=f'3D Surface: {z_col} vs {x_col} and {y_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        height=700
    )
    
    return fig


def create_time_series_decomposition(df, date_col='Date', value_col='NDDI'):
    """
    Decompose time series into trend, seasonal, and residual components
    """
    
    if date_col not in df.columns or value_col not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.sort_values(date_col)
    
    # Simple moving averages for trend
    df_copy['Trend'] = df_copy[value_col].rolling(window=30, center=True).mean()
    df_copy['Residual'] = df_copy[value_col] - df_copy['Trend']
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Original', 'Trend', 'Residual'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=df_copy[date_col], y=df_copy[value_col], mode='lines', name='Original'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_copy[date_col], y=df_copy['Trend'], mode='lines', name='Trend',
                  line=dict(color='red')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_copy[date_col], y=df_copy['Residual'], mode='lines', name='Residual',
                  line=dict(color='gray')),
        row=3, col=1
    )
    
    fig.update_layout(height=900, title_text=f"Time Series Decomposition: {value_col}")
    
    return fig


def create_drought_severity_gauge(current_nddi):
    """
    Create a gauge chart showing current drought severity
    """
    
    # Determine severity
    if current_nddi < -0.5:
        severity = "Extreme Drought"
        color = "darkred"
    elif current_nddi < -0.2:
        severity = "Severe Drought"
        color = "red"
    elif current_nddi < 0:
        severity = "Moderate Drought"
        color = "orange"
    elif current_nddi < 0.2:
        severity = "Normal"
        color = "lightgreen"
    else:
        severity = "Wet Conditions"
        color = "blue"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_nddi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Current NDDI<br><span style='font-size:0.8em;color:{color}'>{severity}</span>"},
        delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [-1, -0.5], 'color': '#8B0000'},
                {'range': [-0.5, -0.2], 'color': '#FF4500'},
                {'range': [-0.2, 0], 'color': '#FFA500'},
                {'range': [0, 0.2], 'color': '#90EE90'},
                {'range': [0.2, 1], 'color': '#4169E1'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': current_nddi
            }
        }
    ))
    
    fig.update_layout(height=400)
    
    return fig


def render_advanced_visualizations(df):
    """
    Render all advanced visualizations in Streamlit
    """
    
    st.header("📊 Advanced Visualizations")
    
    # Visualization selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Temporal Heatmap",
            "Spatial Heatmap",
            "Correlation Matrix",
            "Multi-Metric Dashboard",
            "3D Surface Plot",
            "Time Series Decomposition",
            "Drought Severity Gauge"
        ]
    )
    
    if viz_type == "Temporal Heatmap":
        st.subheader("📅 Temporal NDDI Heatmap")
        fig = create_nddi_heatmap(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Spatial Heatmap":
        st.subheader("🗺️ Spatial Density Heatmap")
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            fig = create_spatial_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Latitude and Longitude columns required for spatial heatmap")
    
    elif viz_type == "Correlation Matrix":
        st.subheader("🔗 Feature Correlation Heatmap")
        fig = create_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Multi-Metric Dashboard":
        st.subheader("📊 Comprehensive Environmental Dashboard")
        fig = create_multi_metric_dashboard(df)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "3D Surface Plot":
        st.subheader("🌐 3D Relationship Visualization")
        if all(col in df.columns for col in ['NDVI', 'NDWI', 'NDDI']):
            try:
                fig = create_3d_surface_plot(df)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Install scipy for 3D surface plots: pip install scipy")
        else:
            st.warning("NDVI, NDWI, and NDDI columns required")
    
    elif viz_type == "Time Series Decomposition":
        st.subheader("📈 Time Series Decomposition")
        fig = create_time_series_decomposition(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Date and NDDI columns required")
    
    elif viz_type == "Drought Severity Gauge":
        st.subheader("🎯 Current Drought Severity")
        if 'NDDI' in df.columns:
            current_nddi = df['NDDI'].iloc[-1]
            fig = create_drought_severity_gauge(current_nddi)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Current NDDI", f"{current_nddi:.3f}")
            col2.metric("7-Day Average", f"{df['NDDI'].tail(7).mean():.3f}")
            col3.metric("30-Day Average", f"{df['NDDI'].tail(30).mean():.3f}")
        else:
            st.warning("NDDI column required")
