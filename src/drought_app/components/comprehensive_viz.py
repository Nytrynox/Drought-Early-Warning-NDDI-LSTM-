"""
Comprehensive Visualization Module: 2D/3D Maps, Water Stress, Model Comparison
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st


def create_model_comparison_plot(results, y_test, model_names=['LSTM', 'CNN', 'CatBoost']):
    """
    Create comprehensive comparison plot for all models
    """
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Model Predictions vs Actual',
            'Prediction Errors',
            'Model Performance Metrics',
            'Residual Distribution'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "histogram"}]
        ]
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Predictions vs Actual
    fig.add_trace(
        go.Scatter(y=y_test, mode='lines', name='Actual', 
                  line=dict(color='black', width=2)),
        row=1, col=1
    )
    
    for idx, model_name in enumerate(model_names):
        if model_name in results['models']:
            predictions = results['models'][model_name]['predictions']
            fig.add_trace(
                go.Scatter(y=predictions, mode='lines', name=model_name,
                          line=dict(color=colors[idx], width=2, dash='dash')),
                row=1, col=1
            )
    
    # Plot 2: Errors
    for idx, model_name in enumerate(model_names):
        if model_name in results['models']:
            predictions = results['models'][model_name]['predictions']
            errors = y_test - predictions
            fig.add_trace(
                go.Scatter(y=errors, mode='markers', name=f'{model_name} Error',
                          marker=dict(color=colors[idx], size=4)),
                row=1, col=2
            )
    
    # Plot 3: Metrics comparison
    metrics_data = []
    for model_name in model_names:
        if model_name in results['models']:
            metrics_data.append({
                'Model': model_name,
                'RMSE': results['models'][model_name]['rmse'],
                'MAE': results['models'][model_name]['mae'],
                'R²': results['models'][model_name]['r2']
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        
        for metric in ['RMSE', 'MAE']:
            fig.add_trace(
                go.Bar(x=metrics_df['Model'], y=metrics_df[metric], name=metric),
                row=2, col=1
            )
    
    # Plot 4: Residuals
    for idx, model_name in enumerate(model_names):
        if model_name in results['models']:
            predictions = results['models'][model_name]['predictions']
            residuals = y_test - predictions
            fig.add_trace(
                go.Histogram(x=residuals, name=f'{model_name}',
                           marker=dict(color=colors[idx]), opacity=0.6),
                row=2, col=2
            )
    
    fig.update_layout(height=800, showlegend=True, title_text="Multi-Model Performance Comparison")
    
    return fig


def create_water_stress_map_2d(df, lat_col='Latitude', lon_col='Longitude'):
    """
    Create 2D water stress level map
    """
    
    # Calculate water stress index (using NDDI, NDWI, soil moisture)
    df_map = df.copy()
    
    # Define water stress levels
    def calculate_water_stress(row):
        stress_score = 0
        
        if 'NDDI' in row:
            # NDDI < -0.5 = high stress
            if row['NDDI'] < -0.5:
                stress_score += 5
            elif row['NDDI'] < -0.2:
                stress_score += 3
            elif row['NDDI'] < 0:
                stress_score += 1
        
        if 'NDWI' in row:
            # Low NDWI = high stress
            if row['NDWI'] < 0.2:
                stress_score += 3
            elif row['NDWI'] < 0.4:
                stress_score += 1
        
        if 'Soil_Moisture' in row:
            # Low soil moisture = high stress
            if row['Soil_Moisture'] < 0.3:
                stress_score += 4
            elif row['Soil_Moisture'] < 0.5:
                stress_score += 2
        
        return stress_score
    
    df_map['Water_Stress'] = df_map.apply(calculate_water_stress, axis=1)
    
    # Create map
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=df_map[lat_col],
        lon=df_map[lon_col],
        mode='markers',
        marker=dict(
            size=15,
            color=df_map['Water_Stress'],
            colorscale=[
                [0, 'darkgreen'],    # No stress
                [0.2, 'lightgreen'], # Low stress
                [0.4, 'yellow'],     # Moderate stress
                [0.6, 'orange'],     # High stress
                [0.8, 'red'],        # Very high stress
                [1, 'darkred']       # Extreme stress
            ],
            showscale=True,
            colorbar=dict(
                title="Water Stress Level",
                tickvals=[0, 3, 6, 9, 12],
                ticktext=['None', 'Low', 'Moderate', 'High', 'Extreme']
            )
        ),
        text=[f"Stress: {s}<br>NDDI: {d:.3f}" 
              for s, d in zip(df_map['Water_Stress'], df_map.get('NDDI', [0]*len(df_map)))],
        hovertemplate='<b>Location</b><br>' +
                     'Lat: %{lat:.4f}<br>' +
                     'Lon: %{lon:.4f}<br>' +
                     '%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=df_map[lat_col].mean(), lon=df_map[lon_col].mean()),
            zoom=8
        ),
        height=600,
        title="2D Water Stress Level Map"
    )
    
    return fig


def create_water_stress_map_3d(df, lat_col='Latitude', lon_col='Longitude'):
    """
    Create 3D water stress visualization
    """
    
    df_map = df.copy()
    
    # Calculate water stress
    def calculate_water_stress(row):
        stress_score = 0
        if 'NDDI' in row:
            if row['NDDI'] < -0.5:
                stress_score += 5
            elif row['NDDI'] < -0.2:
                stress_score += 3
            elif row['NDDI'] < 0:
                stress_score += 1
        if 'NDWI' in row:
            if row['NDWI'] < 0.2:
                stress_score += 3
            elif row['NDWI'] < 0.4:
                stress_score += 1
        return stress_score
    
    df_map['Water_Stress'] = df_map.apply(calculate_water_stress, axis=1)
    
    # Create 3D surface
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=df_map[lon_col],
        y=df_map[lat_col],
        z=df_map['Water_Stress'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_map['Water_Stress'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Water Stress")
        ),
        text=[f"Stress: {s}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}" 
              for s, lat, lon in zip(df_map['Water_Stress'], df_map[lat_col], df_map[lon_col])],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Water Stress Level',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=700,
        title="3D Water Stress Visualization"
    )
    
    return fig


def create_detailed_heatmap(df, value_col='NDDI'):
    """
    Create detailed heatmap with all parameters
    """
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Create correlation heatmap
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title='Detailed Feature Correlation Heatmap',
            height=600,
            xaxis_title='Features',
            yaxis_title='Features'
        )
        
        return fig
    
    return None


def create_linear_comparison_plot(results, y_test):
    """
    Create linear comparison plot for all models
    """
    
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))),
        y=y_test,
        mode='lines',
        name='Actual Values',
        line=dict(color='black', width=3)
    ))
    
    # Add predictions from each model
    colors = {'LSTM': 'blue', 'CNN': 'red', 'CatBoost': 'green', 'Random Forest': 'orange', 'SVR': 'purple'}
    
    for model_name, color in colors.items():
        if model_name in results['models']:
            predictions = results['models'][model_name]['predictions']
            fig.add_trace(go.Scatter(
                x=list(range(len(predictions))),
                y=predictions,
                mode='lines',
                name=f'{model_name} (R²={results["models"][model_name]["r2"]:.3f})',
                line=dict(color=color, width=2, dash='dot')
            ))
    
    fig.update_layout(
        title='Linear Comparison: All Models vs Actual Values',
        xaxis_title='Sample Index',
        yaxis_title='Target Value',
        height=500,
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )
    
    return fig


def create_3d_terrain_map(df, lat_col='Latitude', lon_col='Longitude', value_col='NDDI'):
    """
    Create 3D terrain-style map with error handling for insufficient spatial variation
    """
    
    if all(col in df.columns for col in [lat_col, lon_col, value_col]):
        lat = df[lat_col].values
        lon = df[lon_col].values
        values = df[value_col].values
        
        # Check spatial variation
        lat_range = lat.max() - lat.min()
        lon_range = lon.max() - lon.min()
        
        # If insufficient spatial variation, use 3D scatter instead of surface
        if lat_range < 0.01 and lon_range < 0.01:
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=lon,
                    y=lat,
                    z=values,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=values,
                        colorscale='RdYlGn',
                        colorbar=dict(title=value_col),
                        line=dict(color='white', width=1)
                    ),
                    text=[f'{value_col}: {v:.3f}<br>Lat: {la:.4f}<br>Lon: {lo:.4f}' 
                          for v, la, lo in zip(values, lat, lon)],
                    hoverinfo='text'
                )
            ])
            
            fig.update_layout(
                title=f'3D {value_col} Scatter - Single Location',
                scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title=value_col,
                    camera=dict(eye=dict(x=1.7, y=1.7, z=1.3))
                ),
                height=700
            )
            return fig
        
        # Create grid with padding
        lat_padding = max(0.05, lat_range * 0.1)
        lon_padding = max(0.05, lon_range * 0.1)
        
        lat_unique = np.linspace(lat.min() - lat_padding, lat.max() + lat_padding, 50)
        lon_unique = np.linspace(lon.min() - lon_padding, lon.max() + lon_padding, 50)
        
        lat_grid, lon_grid = np.meshgrid(lat_unique, lon_unique)
        
        # Interpolate values with error handling
        from scipy.interpolate import griddata
        points = df[[lat_col, lon_col]].values
        
        try:
            # Try linear interpolation first (more robust)
            value_grid = griddata(points, values, (lat_grid, lon_grid), method='linear', fill_value=values.mean())
            if np.any(np.isnan(value_grid)):
                value_grid = np.nan_to_num(value_grid, nan=values.mean())
        except Exception:
            # Fallback to nearest neighbor
            try:
                value_grid = griddata(points, values, (lat_grid, lon_grid), method='nearest')
            except Exception:
                # Last resort: return None to skip this visualization
                return None
        
        fig = go.Figure(data=[go.Surface(
            x=lon_grid,
            y=lat_grid,
            z=value_grid,
            colorscale='RdYlGn',
            colorbar=dict(title=value_col)
        )])
        
        fig.update_layout(
            title=f'3D Terrain Map: {value_col}',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title=value_col,
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.3))
            ),
            height=700
        )
        
        return fig
    
    return None


def create_water_stress_gauge(current_stress_level):
    """
    Create gauge chart for current water stress level
    """
    
    # Determine stress category
    if current_stress_level >= 10:
        category = "Extreme"
        color = "darkred"
    elif current_stress_level >= 7:
        category = "Very High"
        color = "red"
    elif current_stress_level >= 5:
        category = "High"
        color = "orange"
    elif current_stress_level >= 3:
        category = "Moderate"
        color = "yellow"
    else:
        category = "Low"
        color = "lightgreen"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_stress_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Water Stress Level<br><span style='font-size:0.8em;color:{color}'>{category}</span>"},
        gauge={
            'axis': {'range': [0, 15], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 3], 'color': '#90EE90'},
                {'range': [3, 5], 'color': '#FFFF99'},
                {'range': [5, 7], 'color': '#FFA500'},
                {'range': [7, 10], 'color': '#FF4500'},
                {'range': [10, 15], 'color': '#8B0000'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': current_stress_level
            }
        }
    ))
    
    fig.update_layout(height=400)
    
    return fig
