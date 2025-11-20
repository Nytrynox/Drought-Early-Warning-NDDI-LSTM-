import pandas as pd
import numpy as np


def compute_nddi_column(df: pd.DataFrame, ndvi_col: str, ndwi_col: str, out_col: str = "NDDI") -> pd.DataFrame:
    ndvi = df[ndvi_col].astype(float)
    ndwi = df[ndwi_col].astype(float)
    denom = (ndvi + ndwi)
    # Avoid division by zero
    nddi = (ndvi - ndwi) / np.where(denom == 0, np.nan, denom)
    out = df.copy()
    out[out_col] = nddi
    return out
