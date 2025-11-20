"""
Google Earth Engine Authentication (SSL Bypass)
This script bypasses SSL verification for authentication
"""

import ssl
import urllib.request

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import ee
    print("=" * 60)
    print("🌍 Google Earth Engine Authentication (SSL Bypass)")
    print("=" * 60)
    print()
    
    # Authenticate
    print("🔐 Opening browser for authentication...")
    print("Please sign in with your Google account.")
    print()
    
    ee.Authenticate()
    
    print()
    print("=" * 60)
    print("✅ Authentication successful!")
    print("=" * 60)
    print()
    
    print("🧪 Testing connection...")
    
    # Try to initialize
    try:
        ee.Initialize()
        print(f"✅ Successfully connected to Google Earth Engine!")
    except Exception as e:
        if 'no project found' in str(e):
            print("ℹ️  Initializing with cloud project...")
            try:
                ee.Initialize(project='earthengine-legacy')
                print(f"✅ Successfully connected to Google Earth Engine!")
            except:
                print("⚠️  Note: You may need to specify a project when using GEE")
                print("   But authentication is complete!")
    
    # Test query
    try:
        image = ee.Image('USGS/SRTMGL1_003')
        title = image.get('title').getInfo()
        print(f"   Test query result: {title}")
    except:
        pass
    
    print()
    print("🎉 Authentication complete!")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import sys
    sys.exit(1)
