import pandas as pd
import numpy as np

def make_synthetic_dataset(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    # Base NDVI/NDWI seasonal patterns
    ndvi = 0.4 + 0.2*np.sin(np.linspace(0, 6*np.pi, n)) + rng.normal(0, 0.05, n)
    ndwi = 0.2 + 0.1*np.cos(np.linspace(0, 6*np.pi, n)) + rng.normal(0, 0.03, n)
    df = pd.DataFrame({
        "date": dates,
        "NDVI": ndvi,
        "NDWI": ndwi,
    })
    return df
