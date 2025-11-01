"""
Google Earth Engine Authentication Setup Script
Run this to authenticate your GEE account
"""

import ee

print("🌍 Google Earth Engine Authentication")
print("=" * 50)
print()

try:
    # Try to initialize - check if already authenticated
    ee.Initialize()
    print("✅ Already authenticated with Google Earth Engine!")
    print()
    
    # Test with a simple query
    print("🧪 Testing GEE connection...")
    image = ee.Image('USGS/SRTMGL1_003')
    print("✅ Successfully connected to Google Earth Engine!")
    print()
    print("You can now use real satellite data in the dashboard!")
    
except Exception as e:
    print("⚠️  Not authenticated yet. Let's set that up...")
    print()
    print("Please follow these steps:")
    print()
    print("1. Visit: https://code.earthengine.google.com/")
    print("2. Sign in with your Google account")
    print("3. Copy the following code and run it in Python:")
    print()
    print("=" * 50)
    print("import ee")
    print("ee.Authenticate()")
    print("ee.Initialize(project='your-project-id')")
    print("=" * 50)
    print()
    print("Then run this script again!")
    print()
    print(f"Error details: {e}")
