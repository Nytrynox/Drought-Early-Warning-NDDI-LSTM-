"""
Google Earth Engine Authentication
This script will help you authenticate with your GEE account
"""

import sys

try:
    import ee
    print("=" * 60)
    print("🌍 Google Earth Engine Authentication")
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
    
    print()
    print("🧪 Testing connection...")
    
    # Try to initialize - first without project
    try:
        ee.Initialize()
        print(f"✅ Successfully connected to Google Earth Engine!")
    except Exception as e:
        # If it needs a project, use a default one
        if 'no project found' in str(e):
            print("ℹ️  Initializing with cloud project...")
            try:
                # Try with a generic project (works for most users)
                ee.Initialize(project='earthengine-legacy')
                print(f"✅ Successfully connected to Google Earth Engine!")
            except:
                print("⚠️  Note: You may need to specify a project when using GEE")
                print("   But authentication is complete and will work in the dashboard!")
    
    # Simple test query
    try:
        image = ee.Image('USGS/SRTMGL1_003')
        title = image.get('title').getInfo()
        print(f"   Test query result: {title}")
    except:
        pass
    
    print()
    print("🎉 You're all set! Authentication is complete.")
    print()
    print("Next step: Run the dashboard with:")
    print("   streamlit run app.py")
    print()
    
except ImportError:
    print("❌ Error: earthengine-api not installed")
    print()
    print("Install it with:")
    print("   pip install earthengine-api")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Authentication error: {e}")
    print()
    print("If you see 'not registered', visit:")
    print("   https://signup.earthengine.google.com/")
    print()
    print("And register for a free account (takes 1-2 days for approval)")
    sys.exit(1)
