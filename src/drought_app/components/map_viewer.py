"""
Interactive Map Visualization for Satellite Data
Shows NDDI overlay on geographical maps with 2D and 3D views
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from scipy.interpolate import griddata


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


def create_3d_terrain_map(df: pd.DataFrame) -> go.Figure:
    """
    Create a detailed 3D terrain-style map with NDDI as height
    """
    # Create interpolated grid for smooth 3D surface
    lat = df['Latitude'].values
    lon = df['Longitude'].values
    nddi = df['NDDI'].values
    
    # Check if we have enough spatial variation
    lat_range = lat.max() - lat.min()
    lon_range = lon.max() - lon.min()
    
    # If insufficient spatial variation (all points very close together)
    if lat_range < 0.01 and lon_range < 0.01:
        # Create simple 3D scatter plot instead of surface
        fig = go.Figure(data=[
            go.Scatter3d(
                x=lon,
                y=lat,
                z=nddi,
                mode='markers',
                marker=dict(
                    size=8,
                    color=nddi,
                    colorscale=[
                        [0, '#8B0000'],      # Extreme drought (dark red)
                        [0.25, '#FF4500'],   # Severe drought (orange-red)
                        [0.5, '#FFA500'],    # Moderate drought (orange)
                        [0.75, '#90EE90'],   # Normal (light green)
                        [1, '#006400']       # Wet (dark green)
                    ],
                    colorbar=dict(
                        title="NDDI<br>Value",
                        thickness=20,
                        len=0.7,
                        x=1.05
                    ),
                    opacity=0.8,
                    line=dict(color='white', width=1)
                ),
                text=[f'NDDI: {n:.3f}<br>Lat: {la:.4f}<br>Lon: {lo:.4f}' 
                      for n, la, lo in zip(nddi, lat, lon)],
                hoverinfo='text',
                name='NDDI Data Points'
            )
        ])
        
        fig.update_layout(
            title="3D NDDI Scatter Plot - Single Location Data",
            scene=dict(
                xaxis_title="Longitude (°E)",
                yaxis_title="Latitude (°N)",
                zaxis_title="NDDI Value",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                bgcolor='rgb(230, 230, 250)'
            ),
            height=700,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig
    
    # Create grid with padding for better interpolation
    lat_padding = max(0.05, lat_range * 0.1)
    lon_padding = max(0.05, lon_range * 0.1)
    
    lat_range_grid = np.linspace(lat.min() - lat_padding, lat.max() + lat_padding, 100)
    lon_range_grid = np.linspace(lon.min() - lon_padding, lon.max() + lon_padding, 100)
    lon_grid, lat_grid = np.meshgrid(lon_range_grid, lat_range_grid)
    
    # Interpolate NDDI values onto grid using linear method (more robust than cubic)
    points = np.column_stack((lat, lon))
    try:
        nddi_grid = griddata(points, nddi, (lat_grid, lon_grid), method='linear', fill_value=nddi.mean())
        # Fill any remaining NaN values
        if np.any(np.isnan(nddi_grid)):
            nddi_grid = np.nan_to_num(nddi_grid, nan=nddi.mean())
    except Exception as e:
        # Fallback to nearest neighbor if linear fails
        nddi_grid = griddata(points, nddi, (lat_grid, lon_grid), method='nearest')
    
    # Create 3D surface
    fig = go.Figure(data=[
        go.Surface(
            x=lon_grid,
            y=lat_grid,
            z=nddi_grid,
            colorscale=[
                [0, '#8B0000'],      # Extreme drought (dark red)
                [0.25, '#FF4500'],   # Severe drought (orange-red)
                [0.5, '#FFA500'],    # Moderate drought (orange)
                [0.75, '#90EE90'],   # Normal (light green)
                [1, '#006400']       # Wet (dark green)
            ],
            colorbar=dict(
                title="NDDI<br>Value",
                thickness=20,
                len=0.7,
                x=1.05
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True))
            ),
            hovertemplate='Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>NDDI: %{z:.3f}<extra></extra>'
        )
    ])
    
    # Add scatter points for actual data locations
    fig.add_trace(go.Scatter3d(
        x=lon,
        y=lat,
        z=nddi,
        mode='markers',
        marker=dict(
            size=4,
            color='black',
            opacity=0.6,
            line=dict(color='white', width=1)
        ),
        name='Actual Data Points',
        hovertemplate='Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>NDDI: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="3D NDDI Terrain Map - Uttar Pradesh",
        scene=dict(
            xaxis_title="Longitude (°E)",
            yaxis_title="Latitude (°N)",
            zaxis_title="NDDI Value",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            bgcolor='rgb(230, 230, 250)'
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_detailed_2d_map(df: pd.DataFrame) -> go.Figure:
    """
    Create a highly detailed 2D map with satellite imagery style
    """
    # Create base scatter mapbox
    fig = go.Figure()
    
    # Add main data points with size based on data quality
    fig.add_trace(go.Scattermapbox(
        lat=df['Latitude'],
        lon=df['Longitude'],
        mode='markers',
        marker=dict(
            size=15,
            color=df['NDDI'],
            colorscale=[
                [0, '#8B0000'],      # Extreme drought
                [0.2, '#FF4500'],    # Severe drought
                [0.4, '#FFA500'],    # Moderate drought
                [0.6, '#FFFF00'],    # Slight drought
                [0.7, '#90EE90'],    # Normal
                [0.85, '#228B22'],   # Wet
                [1, '#006400']       # Very wet
            ],
            cmin=-1,
            cmax=1,
            colorbar=dict(
                title="NDDI<br>Index",
                thickness=20,
                len=0.8,
                x=1.02,
                tickvals=[-1, -0.5, -0.2, 0, 0.2, 0.5, 1],
                ticktext=['Extreme<br>Drought', 'Severe', 'Moderate', 'Normal', 'Slight<br>Wet', 'Wet', 'Very<br>Wet']
            ),
            opacity=0.85,
            line=dict(color='white', width=2)
        ),
        text=df.apply(
            lambda row: (
                f"<b>Location</b><br>"
                f"Lat: {row['Latitude']:.4f}°N<br>"
                f"Lon: {row['Longitude']:.4f}°E<br><br>"
                f"<b>NDDI: {row['NDDI']:.3f}</b><br>"
                f"Severity: {classify_drought(row['NDDI'])}<br><br>"
                f"NDVI: {row.get('NDVI', 'N/A'):.3f if pd.notna(row.get('NDVI')) else 'N/A'}<br>"
                f"NDWI: {row.get('NDWI', 'N/A'):.3f if pd.notna(row.get('NDWI')) else 'N/A'}<br>"
                + (f"<br><b>District:</b> {row.get('District', 'N/A')}<br>" if 'District' in row and pd.notna(row.get('District')) else "")
                + (f"<b>Village:</b> {row.get('Village', 'N/A')}" if 'Village' in row and pd.notna(row.get('Village')) else "")
            ),
            axis=1
        ),
        hoverinfo='text',
        name='NDDI Data Points'
    ))
    
    # Calculate center and zoom
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    # Calculate appropriate zoom level based on data spread
    lat_range = df['Latitude'].max() - df['Latitude'].min()
    lon_range = df['Longitude'].max() - df['Longitude'].min()
    max_range = max(lat_range, lon_range)
    
    if max_range < 0.1:
        zoom = 12
    elif max_range < 0.5:
        zoom = 10
    elif max_range < 1:
        zoom = 8
    elif max_range < 3:
        zoom = 6
    else:
        zoom = 5
    
    fig.update_layout(
        mapbox=dict(
            style='satellite-streets',  # Detailed satellite view
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
        title=dict(
            text="Detailed 2D Satellite Map - NDDI Distribution",
            font=dict(size=20, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig


def render_map_view(df: pd.DataFrame):
    """
    Render the enhanced map visualization interface with 3D and detailed 2D views
    """
    st.header("🗺️ Advanced Satellite Data Map Visualization")
    
    # Map type selection with more options
    map_type = st.selectbox(
        "Map Visualization Type",
        ["Detailed 2D Satellite Map", "3D Terrain Map", "Standard Point Map", "Heatmap Grid", "Time Animation"]
    )
    
    if map_type == "Detailed 2D Satellite Map":
        st.info("🛰️ High-resolution 2D satellite map with detailed NDDI overlay and location information")
        st.plotly_chart(create_detailed_2d_map(df), use_container_width=True)
        
    elif map_type == "3D Terrain Map":
        st.info("🏔️ Interactive 3D terrain visualization where height represents NDDI value - drag to rotate!")
        st.plotly_chart(create_3d_terrain_map(df), use_container_width=True)
        
    elif map_type == "Standard Point Map":
        st.plotly_chart(create_nddi_map(df), use_container_width=True)
    
    elif map_type == "Heatmap Grid":
        grid_size = st.slider("Grid Resolution", 10, 50, 20)
        st.plotly_chart(create_heatmap_grid(df, grid_size=grid_size), use_container_width=True)
    
    elif map_type == "Time Animation":
        if 'Date' in df.columns:
            st.plotly_chart(create_time_series_map_animation(df), use_container_width=True)
        else:
            st.warning("Date column not found. Showing detailed 2D map instead.")
            st.plotly_chart(create_detailed_2d_map(df), use_container_width=True)
    
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
