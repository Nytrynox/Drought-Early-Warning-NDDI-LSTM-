"""
Enhanced Data Export and Recording Module
Exports satellite data, predictions, and creates comprehensive CSV reports
"""

import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from io import BytesIO
import zipfile


def create_comprehensive_csv_export(
    satellite_df=None,
    predictions_df=None,
    model_metrics=None,
    include_metadata=True
):
    """
    Create a comprehensive CSV export package with all data and results
    
    Args:
        satellite_df: DataFrame with satellite data
        predictions_df: DataFrame with model predictions
        model_metrics: Dictionary with model performance metrics
        include_metadata: Include metadata sheet
    
    Returns:
        BytesIO object with ZIP file containing multiple CSVs
    """
    
    # Create a BytesIO object for the ZIP file
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # 1. Export satellite data
        if satellite_df is not None:
            csv_buffer = satellite_df.to_csv(index=False)
            zip_file.writestr('01_satellite_data.csv', csv_buffer)
        
        # 2. Export predictions
        if predictions_df is not None:
            csv_buffer = predictions_df.to_csv(index=False)
            zip_file.writestr('02_model_predictions.csv', csv_buffer)
        
        # 3. Export model metrics
        if model_metrics is not None:
            metrics_df = pd.DataFrame([
                {
                    'Model': name,
                    'RMSE': info.get('rmse', 'N/A'),
                    'MAE': info.get('mae', 'N/A'),
                    'R2_Score': info.get('r2', 'N/A')
                }
                for name, info in model_metrics.items()
            ])
            csv_buffer = metrics_df.to_csv(index=False)
            zip_file.writestr('03_model_metrics.csv', csv_buffer)
        
        # 4. Create metadata file
        if include_metadata:
            metadata = {
                'Export_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Data_Points': [len(satellite_df) if satellite_df is not None else 0],
                'Models_Trained': [len(model_metrics) if model_metrics else 0],
                'System': ['AI-Powered Drought Early Warning System'],
                'Version': ['1.0']
            }
            metadata_df = pd.DataFrame(metadata)
            csv_buffer = metadata_df.to_csv(index=False)
            zip_file.writestr('00_metadata.csv', csv_buffer)
    
    zip_buffer.seek(0)
    return zip_buffer


def record_satellite_data_to_csv(df, location_name="location", auto_save=True):
    """
    Record real-time satellite data to CSV file with timestamp
    
    Args:
        df: DataFrame with satellite data
        location_name: Name of location for filename
        auto_save: Automatically save to file
    
    Returns:
        Filename if auto_save is True, else DataFrame
    """
    
    # Add timestamp column
    df_export = df.copy()
    df_export['Recorded_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"satellite_data_{location_name}_{timestamp}.csv"
    
    if auto_save:
        df_export.to_csv(filename, index=False)
        return filename
    else:
        return df_export


def create_heatmap_data(df, lat_col='Latitude', lon_col='Longitude', value_col='NDDI'):
    """
    Prepare data for heatmap visualization
    
    Args:
        df: DataFrame with location and value data
        lat_col: Latitude column name
        lon_col: Longitude column name
        value_col: Value column for heatmap intensity
    
    Returns:
        DataFrame formatted for heatmap
    """
    
    # Create grid coordinates
    heatmap_df = df[[lat_col, lon_col, value_col]].copy()
    heatmap_df.columns = ['lat', 'lon', 'value']
    
    # Add intensity category
    heatmap_df['intensity'] = pd.cut(
        heatmap_df['value'],
        bins=[-1, -0.5, -0.2, 0, 0.2, 1],
        labels=['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Normal', 'Wet']
    )
    
    return heatmap_df


def generate_download_button(data, filename, button_text, file_type='csv'):
    """
    Create a download button for data export
    
    Args:
        data: Data to download (DataFrame, BytesIO, or string)
        filename: Name for downloaded file
        button_text: Text to display on button
        file_type: Type of file ('csv', 'zip', 'json')
    
    Returns:
        Streamlit download button
    """
    
    if file_type == 'csv':
        if isinstance(data, pd.DataFrame):
            csv = data.to_csv(index=False)
            mime = 'text/csv'
        else:
            csv = data
            mime = 'text/csv'
        
        return st.download_button(
            label=button_text,
            data=csv,
            file_name=filename,
            mime=mime
        )
    
    elif file_type == 'zip':
        return st.download_button(
            label=button_text,
            data=data,
            file_name=filename,
            mime='application/zip'
        )
    
    elif file_type == 'json':
        import json
        if isinstance(data, dict):
            json_str = json.dumps(data, indent=2)
        else:
            json_str = data.to_json(orient='records', indent=2)
        
        return st.download_button(
            label=button_text,
            data=json_str,
            file_name=filename,
            mime='application/json'
        )
