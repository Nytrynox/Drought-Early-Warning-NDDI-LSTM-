"""
Interactive Map Visualization for Satellite Data
Shows NDDI overlay on geographical maps
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


def create_nddi_map(
    df: pd.DataFrame,
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    nddi_col: str = 'NDDI',
    date_col: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive map showing NDDI values
    
    Args:
        df: DataFrame with location and NDDI data
        lat_col: Latitude column name
        lon_col: Longitude column name
        nddi_col: NDDI column name
        date_col: Optional date column for time series
    
    Returns:
        Plotly figure object
    """
    # Prepare data
    plot_df = df.copy()
    
    # Add drought severity classification
    plot_df['Drought_Severity'] = plot_df[nddi_col].apply(classify_drought)
    plot_df['Color'] = plot_df['Drought_Severity'].apply(get_severity_color)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter mapbox
    fig.add_trace(go.Scattermapbox(
        lat=plot_df[lat_col],
        lon=plot_df[lon_col],
        mode='markers',
        marker=dict(
            size=12,
            color=plot_df[nddi_col],
            colorscale=[
                [0, '#8B0000'],      # Extreme drought
                [0.25, '#FF4500'],   # Severe drought
                [0.5, '#FFA500'],    # Moderate drought
                [0.75, '#90EE90'],   # Normal
                [1, '#4169E1']       # Wet
            ],
            cmin=-1,
            cmax=1,
            colorbar=dict(
                title="NDDI",
                thickness=15,
                len=0.7
            ),
            opacity=0.8
        ),
        text=plot_df.apply(
            lambda row: f"NDDI: {row[nddi_col]:.3f}<br>"
                       f"Severity: {row['Drought_Severity']}<br>"
                       f"Lat: {row[lat_col]:.4f}<br>"
                       f"Lon: {row[lon_col]:.4f}",
            axis=1
        ),
        hoverinfo='text',
        name='NDDI'
    ))
    
    # Update layout
    center_lat = plot_df[lat_col].mean()
    center_lon = plot_df[lon_col].mean()
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=5
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title="NDDI Satellite Data - Geographical Distribution"
    )
    
    return fig


def create_heatmap_grid(
    df: pd.DataFrame,
    metric: str = 'NDDI',
    grid_size: int = 20
) -> go.Figure:
    """
    Create a heatmap grid for spatial analysis
    """
    # Create a grid
    lat_bins = pd.cut(df['Latitude'], bins=grid_size)
    lon_bins = pd.cut(df['Longitude'], bins=grid_size)
    
    # Aggregate by grid cell
    grid_data = df.groupby([lat_bins, lon_bins])[metric].mean().reset_index()
    
    # Create pivot table
    pivot = grid_data.pivot_table(
        values=metric,
        index='Latitude',
        columns='Longitude',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(x) for x in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title=metric)
    ))
    
    fig.update_layout(
        title=f'{metric} Spatial Heatmap',
        xaxis_title='Longitude Bins',
        yaxis_title='Latitude Bins',
        height=500
    )
    
    return fig


def create_time_series_map_animation(
    df: pd.DataFrame,
    date_col: str = 'Date',
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    value_col: str = 'NDDI'
) -> go.Figure:
    """
    Create an animated map showing changes over time
    """
    if date_col not in df.columns:
        return create_nddi_map(df, lat_col, lon_col, value_col)
    
    # Ensure date is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Create animation
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=value_col,
        size=abs(df[value_col]) * 10 + 5,
        animation_frame=df[date_col].dt.strftime('%Y-%m-%d'),
        color_continuous_scale='RdYlGn',
        range_color=[-1, 1],
        mapbox_style='open-street-map',
        zoom=5,
        height=600,
        title=f'{value_col} Over Time'
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=value_col,
            thickness=15,
            len=0.7
        )
    )
    
    return fig


def classify_drought(nddi: float) -> str:
    """Classify drought severity from NDDI value"""
    if pd.isna(nddi):
        return "Unknown"
    elif nddi < -0.5:
        return "Extreme Drought"
    elif nddi < -0.2:
        return "Severe Drought"
    elif nddi < 0.0:
        return "Moderate Drought"
    elif nddi < 0.2:
        return "Normal"
    else:
        return "Wet Conditions"


def get_severity_color(severity: str) -> str:
    """Get color for severity level"""
    colors = {
        "Extreme Drought": "#8B0000",
        "Severe Drought": "#FF4500",
        "Moderate Drought": "#FFA500",
        "Normal": "#90EE90",
        "Wet Conditions": "#4169E1",
        "Unknown": "#808080"
    }
    return colors.get(severity, "#808080")


def render_map_view(df: pd.DataFrame):
    """
    Render the map visualization interface
    """
    st.header("🗺️ Satellite Data Map Visualization")
    
    # Map type selection
    map_type = st.selectbox(
        "Map Visualization Type",
        ["Point Map", "Heatmap Grid", "Time Animation"]
    )
    
    if map_type == "Point Map":
        st.plotly_chart(create_nddi_map(df), use_container_width=True)
    
    elif map_type == "Heatmap Grid":
        grid_size = st.slider("Grid Resolution", 10, 50, 20)
        st.plotly_chart(create_heatmap_grid(df, grid_size=grid_size), use_container_width=True)
    
    elif map_type == "Time Animation":
        if 'Date' in df.columns:
            st.plotly_chart(create_time_series_map_animation(df), use_container_width=True)
        else:
            st.warning("Date column not found. Showing point map instead.")
            st.plotly_chart(create_nddi_map(df), use_container_width=True)
    
    # Statistics summary
    st.divider()
    st.subheader("📊 Spatial Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'NDDI' in df.columns:
        col1.metric("Mean NDDI", f"{df['NDDI'].mean():.3f}")
        col2.metric("Min NDDI", f"{df['NDDI'].min():.3f}")
        col3.metric("Max NDDI", f"{df['NDDI'].max():.3f}")
        col4.metric("Std Dev", f"{df['NDDI'].std():.3f}")
        
        # Severity distribution
        df_with_severity = df.copy()
        df_with_severity['Severity'] = df_with_severity['NDDI'].apply(classify_drought)
        severity_counts = df_with_severity['Severity'].value_counts()
        
        st.subheader("Drought Severity Distribution")
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Distribution of Drought Severity Classes",
            color=severity_counts.index,
            color_discrete_map={
                "Extreme Drought": "#8B0000",
                "Severe Drought": "#FF4500",
                "Moderate Drought": "#FFA500",
                "Normal": "#90EE90",
                "Wet Conditions": "#4169E1"
            }
        )
        st.plotly_chart(fig_severity, use_container_width=True)
