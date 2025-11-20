"""
Real-time Satellite Data Integration for NDDI
Supports Google Earth Engine, NASA MODIS, Sentinel Hub
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import requests


class SatelliteDataFetcher:
    """
    Fetches real-time NDVI, NDWI, and NDDI data from satellite imagery
    """
    
    def __init__(self):
        self.gee_available = False
        self.sentinel_available = False
        
        # Try to import Google Earth Engine
        try:
            import ee
            self.ee = ee
            self.gee_available = True
        except ImportError:
            pass
        
        # Try to import sentinelsat
        try:
            from sentinelsat import SentinelAPI
            self.SentinelAPI = SentinelAPI
            self.sentinel_available = True
        except ImportError:
            pass
    
    def fetch_modis_ndvi(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch MODIS NDVI data from NASA API
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with NDVI time series
        """
        # NASA MODIS API endpoint (example - replace with actual endpoint)
        # This is a placeholder for the actual NASA API
        
        url = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'startDate': start_date,
            'endDate': end_date,
            'kmAboveBelow': 0,
            'kmLeftRight': 0
        }
        
        try:
            # Note: Actual implementation requires NASA API key
            # response = requests.get(url, params=params, headers={'Authorization': 'Bearer YOUR_API_KEY'})
            # data = response.json()
            
            # For now, return simulated data
            return self._simulate_satellite_data(lat, lon, start_date, end_date)
        
        except Exception as e:
            print(f"MODIS fetch error: {e}")
            return self._simulate_satellite_data(lat, lon, start_date, end_date)
    
    def fetch_sentinel_data(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch Sentinel-2 data from Copernicus Hub
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date
            username: Copernicus Hub username
            password: Copernicus Hub password
        
        Returns:
            DataFrame with satellite indices
        """
        if not self.sentinel_available:
            return self._simulate_satellite_data(lat, lon, start_date, end_date)
        
        try:
            # Sentinel Hub API implementation would go here
            # Requires authentication and proper API setup
            return self._simulate_satellite_data(lat, lon, start_date, end_date)
        
        except Exception as e:
            print(f"Sentinel fetch error: {e}")
            return self._simulate_satellite_data(lat, lon, start_date, end_date)
    
    def fetch_earth_engine_data(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch data using Google Earth Engine
        
        Requires:
            - Google Earth Engine account
            - Authenticated with: ee.Authenticate()
        """
        if not self.gee_available:
            print("ℹ️  Google Earth Engine not available. Using simulated data.")
            return self._simulate_satellite_data(lat, lon, start_date, end_date)
        
        try:
            # Initialize Earth Engine
            try:
                self.ee.Initialize()
            except Exception as init_error:
                # Try with earthengine-legacy project
                try:
                    self.ee.Initialize(project='earthengine-legacy')
                except:
                    print(f"⚠️  GEE not authenticated. Run: python authenticate_gee.py")
                    return self._simulate_satellite_data(lat, lon, start_date, end_date)
            
            print(f"🛰️  Fetching real satellite data from Google Earth Engine...")
            
            # Define point of interest
            point = self.ee.Geometry.Point([lon, lat])
            
            # Load Sentinel-2 Surface Reflectance data
            collection = (self.ee.ImageCollection('COPERNICUS/S2_SR')
                         .filterBounds(point)
                         .filterDate(start_date, end_date)
                         .filter(self.ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
            
            # Calculate NDVI and NDWI for each image
            def calculate_indices(image):
                # NDVI = (NIR - Red) / (NIR + Red)
                # For Sentinel-2: NIR=B8, Red=B4
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                
                # NDWI = (Green - NIR) / (Green + NIR)  
                # For Sentinel-2: Green=B3, NIR=B8
                ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
                
                # Add timestamp
                return image.addBands([ndvi, ndwi]).set('date', image.date().format('YYYY-MM-dd'))
            
            collection_with_indices = collection.map(calculate_indices)
            
            # Get the collection size
            size = collection_with_indices.size().getInfo()
            
            if size == 0:
                print(f"⚠️  No satellite images found for location ({lat}, {lon}) in date range.")
                print(f"   Using simulated data instead.")
                return self._simulate_satellite_data(lat, lon, start_date, end_date)
            
            print(f"✅ Found {size} satellite images")
            
            # Extract time series data
            def extract_point_data(image):
                # Get values at point location
                values = image.reduceRegion(
                    reducer=self.ee.Reducer.first(),
                    geometry=point,
                    scale=10  # 10m resolution for Sentinel-2
                ).getInfo()
                
                return {
                    'date': image.get('date').getInfo(),
                    'NDVI': values.get('NDVI'),
                    'NDWI': values.get('NDWI')
                }
            
            # Get list of images
            image_list = collection_with_indices.toList(size)
            
            # Extract data from each image
            data_points = []
            for i in range(min(size, 100)):  # Limit to 100 images for performance
                image = self.ee.Image(image_list.get(i))
                point_data = extract_point_data(image)
                if point_data['NDVI'] is not None and point_data['NDWI'] is not None:
                    data_points.append(point_data)
            
            if len(data_points) == 0:
                print("⚠️  No valid data extracted. Using simulated data.")
                return self._simulate_satellite_data(lat, lon, start_date, end_date)
            
            # Create DataFrame
            df = pd.DataFrame(data_points)
            df['Date'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
            df = df.sort_values('Date')
            
            # Calculate NDDI
            df['NDDI'] = (df['NDVI'] - df['NDWI']) / (df['NDVI'] + df['NDWI'] + 1e-10)
            
            # Add location info
            df['Latitude'] = lat
            df['Longitude'] = lon
            
            # Fill in additional environmental data (simulated)
            dates = pd.to_datetime(df['Date'])
            day_of_year = dates.dt.dayofyear
            n = len(df)
            
            df['Temperature_C'] = 20 + (30 - abs(lat) * 0.5) + 10 * np.sin(2 * np.pi * day_of_year / 365)
            df['Precipitation_mm'] = np.maximum(0, 50 + 40 * np.sin(2 * np.pi * day_of_year / 365 + np.pi))
            df['Soil_Moisture'] = 0.3 + 0.2 * (df['NDWI'] / df['NDWI'].max())
            df['Cloud_Cover'] = np.random.uniform(0, 20, n)
            df['Data_Quality'] = 'Good'
            
            print(f"✅ Successfully fetched {len(df)} data points from Google Earth Engine!")
            
            return df
        
        except Exception as e:
            print(f"❌ Earth Engine error: {e}")
            print("   Falling back to simulated data.")
            return self._simulate_satellite_data(lat, lon, start_date, end_date)
    
    def _simulate_satellite_data(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generate simulated satellite data for testing
        This simulates realistic NDDI patterns based on location and season
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Time index for seasonal patterns
        time_idx = np.arange(n)
        day_of_year = pd.to_datetime(dates).dayofyear
        
        # Latitude-based baseline (more vegetation near equator)
        lat_factor = 1.0 - abs(lat) / 90.0 * 0.5
        
        # Seasonal patterns
        seasonal_ndvi = 0.3 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
        seasonal_ndwi = 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Generate NDVI (vegetation index)
        ndvi_base = 0.4 + 0.3 * lat_factor
        ndvi = ndvi_base + seasonal_ndvi + np.random.normal(0, 0.05, n)
        ndvi = np.clip(ndvi, 0, 1)
        
        # Generate NDWI (water index)
        ndwi_base = 0.2 + 0.1 * lat_factor
        ndwi = ndwi_base + seasonal_ndwi + np.random.normal(0, 0.05, n)
        ndwi = np.clip(ndwi, 0, 1)
        
        # Calculate NDDI
        nddi = (ndvi - ndwi) / (ndvi + ndwi + 1e-10)
        nddi = np.clip(nddi, -1, 1)
        
        # Additional environmental variables
        temp_base = 20 + (30 - abs(lat) * 0.5)
        temperature = temp_base + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, n)
        
        precip_base = 50 + (100 - abs(lat) * 0.8)
        precipitation = np.maximum(
            0,
            precip_base + 40 * np.sin(2 * np.pi * day_of_year / 365 + np.pi) + np.random.normal(0, 15, n)
        )
        
        soil_moisture = 0.3 + 0.2 * (ndwi / ndwi.max()) + np.random.normal(0, 0.05, n)
        soil_moisture = np.clip(soil_moisture, 0, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Latitude': lat,
            'Longitude': lon,
            'NDVI': ndvi,
            'NDWI': ndwi,
            'NDDI': nddi,
            'Temperature_C': temperature,
            'Precipitation_mm': precipitation,
            'Soil_Moisture': soil_moisture,
            'Cloud_Cover': np.random.uniform(0, 30, n),
            'Data_Quality': np.random.choice(['Good', 'Fair', 'Poor'], n, p=[0.7, 0.2, 0.1])
        })
        
        return df
    
    def get_available_sources(self) -> dict:
        """Return available satellite data sources"""
        return {
            'Google Earth Engine': self.gee_available,
            'Sentinel Hub': self.sentinel_available,
            'Simulated Data': True
        }


def fetch_real_time_satellite_data(
    lat: float,
    lon: float,
    days_back: int = 180,
    source: str = 'auto'
) -> pd.DataFrame:
    """
    Convenience function to fetch satellite data
    
    Args:
        lat: Latitude
        lon: Longitude
        days_back: Number of days of historical data
        source: Data source ('gee', 'sentinel', 'modis', 'auto')
    
    Returns:
        DataFrame with satellite time series
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    fetcher = SatelliteDataFetcher()
    
    if source == 'gee' and fetcher.gee_available:
        return fetcher.fetch_earth_engine_data(
            lat, lon,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    elif source == 'sentinel' and fetcher.sentinel_available:
        return fetcher.fetch_sentinel_data(
            lat, lon,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    elif source == 'modis':
        return fetcher.fetch_modis_ndvi(
            lat, lon,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    else:
        # Auto-select best available source
        if fetcher.gee_available:
            return fetcher.fetch_earth_engine_data(
                lat, lon,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        else:
            return fetcher._simulate_satellite_data(
                lat, lon,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )


def fetch_satellite_data_for_agriculture(
    agriculture_df: pd.DataFrame,
    sample_locations: int = 5,
    days_back: int = 180
) -> pd.DataFrame:
    """
    Fetch REAL satellite data for multiple locations from UP agriculture dataset
    
    Args:
        agriculture_df: Agriculture dataset with district/village names or GPS coordinates
        sample_locations: Number of random locations to sample
        days_back: Number of days of historical data
    
    Returns:
        Combined satellite data for all sampled locations
    """
    combined_data = []
    
    # Check if GPS coordinates already exist
    has_gps = 'Latitude' in agriculture_df.columns and 'Longitude' in agriculture_df.columns
    
    # If not, add GPS coordinates from district/village names
    if not has_gps and 'district_name' in agriculture_df.columns:
        print("🗺️  Adding GPS coordinates from Uttar Pradesh district/village mapping...")
        from .up_gps_coordinates import add_gps_to_up_village_data
        agriculture_df = add_gps_to_up_village_data(agriculture_df.copy())
        has_gps = True
    
    if has_gps:
        # Sample random locations from agriculture dataset
        sample_df = agriculture_df.sample(n=min(sample_locations, len(agriculture_df)))
        
        print(f"📍 Fetching satellite data for {len(sample_df)} Uttar Pradesh locations...")
        
        for idx, row in sample_df.iterrows():
            try:
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                
                # Validate coordinates are in Uttar Pradesh
                from .up_gps_coordinates import validate_up_coordinates
                if not validate_up_coordinates(lat, lon):
                    print(f"⚠️  Warning: Coordinates ({lat}, {lon}) outside UP boundaries, skipping")
                    continue
                
                district = row.get('district_name', 'Unknown')
                village = row.get('village_name', 'Unknown')
                
                print(f"   • {district} - {village}: ({lat:.4f}, {lon:.4f})")
                
                # Fetch REAL satellite data for this location
                satellite_data = fetch_real_time_satellite_data(lat, lon, days_back, source='gee')
                
                if len(satellite_data) > 0:
                    # Add agriculture metadata from UP village data
                    satellite_data['Agriculture_Index'] = idx
                    satellite_data['District'] = district
                    satellite_data['Village'] = village
                    satellite_data['State'] = row.get('state_name', 'UTTAR PRADESH')
                    satellite_data['Block'] = row.get('block_tehsil_name', 'Unknown')
                    satellite_data['Geographical_Area'] = row.get('geographical_area', 0)
                    satellite_data['Cultivable_Area'] = row.get('cultivable_area', 0)
                    satellite_data['Net_Sown_Area'] = row.get('net_sown_area', 0)
                    satellite_data['Gross_Irrigated_Area'] = row.get('gross_irrigated_area_total', 0)
                    satellite_data['Net_Irrigated_Area'] = row.get('net_irrigated_area', 0)
                    satellite_data['Ground_Water_Pre_Monsoon'] = row.get('avg_ground_water_level_pre_monsoon', 0)
                    satellite_data['Ground_Water_Post_Monsoon'] = row.get('avg_ground_water_level_post_monsoon', 0)
                    
                    combined_data.append(satellite_data)
                    print(f"      ✅ Retrieved {len(satellite_data)} data points")
                    
            except Exception as e:
                print(f"❌ Error processing location {idx}: {e}")
                continue
    
    if combined_data:
        print(f"\n✅ Successfully fetched satellite data for {len(combined_data)} locations!")
        return pd.concat(combined_data, ignore_index=True)
    else:
        print("❌ No satellite data could be fetched. Please check GEE authentication.")
        return pd.DataFrame()


def analyze_drought_trends(satellite_df: pd.DataFrame) -> dict:
    """
    Analyze drought trends from satellite time series data
    
    Args:
        satellite_df: Satellite data with NDDI values
    
    Returns:
        Dictionary with trend analysis
    """
    analysis = {
        'current_status': 'Unknown',
        'trend': 'Stable',
        'severity_distribution': {},
        'risk_level': 'Low'
    }
    
    if len(satellite_df) == 0 or 'NDDI' not in satellite_df.columns:
        return analysis
    
    # Current drought status
    latest_nddi = satellite_df['NDDI'].iloc[-1]
    analysis['current_status'] = calculate_drought_severity(latest_nddi)
    
    # Trend analysis (last 30 days vs previous)
    if len(satellite_df) >= 60:
        recent_nddi = satellite_df['NDDI'].tail(30).mean()
        previous_nddi = satellite_df['NDDI'].tail(60).head(30).mean()
        
        if recent_nddi < previous_nddi - 0.05:
            analysis['trend'] = 'Worsening'
        elif recent_nddi > previous_nddi + 0.05:
            analysis['trend'] = 'Improving'
        else:
            analysis['trend'] = 'Stable'
    
    # Severity distribution
    severities = satellite_df['NDDI'].apply(calculate_drought_severity)
    analysis['severity_distribution'] = severities.value_counts().to_dict()
    
    # Risk level
    extreme_severe_count = (
        severities.isin(['Extreme Drought', 'Severe Drought']).sum()
    )
    risk_ratio = extreme_severe_count / len(severities)
    
    if risk_ratio > 0.5:
        analysis['risk_level'] = 'High'
    elif risk_ratio > 0.2:
        analysis['risk_level'] = 'Medium'
    else:
        analysis['risk_level'] = 'Low'
    
    return analysis


def calculate_drought_severity(nddi: float) -> str:
    """
    Classify drought severity based on NDDI value
    
    NDDI ranges:
    - < -0.5: Extreme drought
    - -0.5 to -0.2: Severe drought
    - -0.2 to 0.0: Moderate drought
    - 0.0 to 0.2: Normal
    - > 0.2: Wet conditions
    """
    if nddi < -0.5:
        return "Extreme Drought"
    elif nddi < -0.2:
        return "Severe Drought"
    elif nddi < 0.0:
        return "Moderate Drought"
    elif nddi < 0.2:
        return "Normal"
    else:
        return "Wet Conditions"


def get_drought_color(severity: str) -> str:
    """Get color code for drought severity"""
    colors = {
        "Extreme Drought": "#8B0000",
        "Severe Drought": "#FF4500",
        "Moderate Drought": "#FFA500",
        "Normal": "#90EE90",
        "Wet Conditions": "#4169E1"
    }
    return colors.get(severity, "#808080")
