import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

from ..core.session import get_session_state
from ..utils.nddi import compute_nddi_column


def _guess_columns(df: pd.DataFrame):
    cols_lower = {c: c.lower() for c in df.columns}
    # time
    time_candidates = [c for c,l in cols_lower.items() if any(k in l for k in ["date","time","timestamp"])]
    time_col = time_candidates[0] if time_candidates else df.columns[0]
    # ndvi/ndwi/nddi
    ndvi_candidates = [c for c,l in cols_lower.items() if "ndvi" in l]
    ndwi_candidates = [c for c,l in cols_lower.items() if "ndwi" in l]
    nddi_candidates = [c for c,l in cols_lower.items() if "nddi" in l]
    ndvi = ndvi_candidates[0] if ndvi_candidates else None
    ndwi = ndwi_candidates[0] if ndwi_candidates else None
    nddi = nddi_candidates[0] if nddi_candidates else None
    # lat/lon
    lat_candidates = [c for c,l in cols_lower.items() if any(k in l for k in ["lat","latitude"])]
    lon_candidates = [c for c,l in cols_lower.items() if any(k in l for k in ["lon","lng","longitude"]) ]
    lat = lat_candidates[0] if lat_candidates else None
    lon = lon_candidates[0] if lon_candidates else None
    return time_col, ndvi, ndwi, nddi, lat, lon


def _coerce_numeric_locale(series: pd.Series) -> pd.Series:
    """Try to coerce strings with locale-specific decimal separators to English floats.
    Examples handled:
    - "0,123" -> 0.123
    - "1 234,56" -> 1234.56
    - "1.234,56" -> 1234.56
    - "1,234.56" remains 1234.56
    """
    if series.dtype.kind in ("i", "u", "f"):
        return series
    s = series.astype(str).str.strip()
    # Remove non-breaking spaces and normal spaces used as thousand separators
    s = s.str.replace("\xa0", " ", regex=False)
    # Heuristic: if there are both '.' and ',' and comma appears after last dot, treat comma as decimal
    def _normalize(val: str) -> str:
        if val is None:
            return val
        v = val.strip()
        if not v:
            return v
        has_dot = "." in v
        has_comma = "," in v
        if has_dot and has_comma:
            # If comma is after the last dot -> comma decimal, dots thousands
            if v.rfind(",") > v.rfind("."):
                v = v.replace(".", "")  # remove thousands
                v = v.replace(",", ".")  # decimal point
                return v
            else:
                # typical English formatting already
                return v.replace(",", "")
        if has_comma and not has_dot:
            # Likely comma decimal or thousands with no decimal
            # If there is exactly one comma and after removing spaces there are <=3 digits after it -> decimal
            parts = v.split(",")
            if len(parts) == 2 and len(parts[1]) <= 3:
                return v.replace(" ", "").replace(",", ".")
            # else treat comma as thousands
            return v.replace(",", "")
        # If only dots exist, could be decimal or thousands; removing thousands-style dots like 1.234.567
        if has_dot and v.count(".") > 1:
            # assume thousands grouping, remove all but last
            head, _, tail = v.rpartition(".")
            return head.replace(".", "") + "." + tail
        return v
    s = s.apply(_normalize)
    return pd.to_numeric(s, errors="coerce")


def _downsample_df(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    return df.iloc[idx]


def _preset_for_source(source_name: Optional[str], df: pd.DataFrame):
    if not source_name:
        return None
    name = source_name.lower()
    # Preset for agriculture_dataset.csv (adjust as needed based on typical schemas)
    if "agriculture_dataset.csv" in name:
        # Exact header mapping observed in the provided CSV
        if "Temporal_Images" in df.columns:
            time_col = "Temporal_Images"
        else:
            time_col = None  # will fall back to row-order synthetic time
        ndvi_col = "NDVI" if "NDVI" in df.columns else None
        # NDWI and NDDI are not present in the file; leave None so UI can prompt or compute from user-selected proxy
        ndwi_col = None
        nddi_col = None
        return {
            "time": time_col,
            "ndvi": ndvi_col,
            "ndwi": ndwi_col,
            "nddi": nddi_col,
        }
        def _find_column(candidates_exact, candidates_contains):
            cols_lower = {c.lower(): c for c in df.columns}
            # exact matches
            for cand in candidates_exact:
                if cand.lower() in cols_lower:
                    return cols_lower[cand.lower()]
            # contains tokens
            for c in df.columns:
                lc = c.lower()
                for token in candidates_contains:
                    if token.lower() in lc:
                        return c
            return None

        time_col = _find_column(
            ["date", "timestamp", "time", "observationdate", "observation_date"],
            ["date", "time", "timestamp"],
        ) or df.columns[0]
        ndvi_col = _find_column(
            ["ndvi", "ndvi_value", "ndvi_index", "ndvi_mean"],
            ["ndvi", "vegetation index"],
        )
        ndwi_col = _find_column(
            ["ndwi", "ndwi_value", "ndwi_index", "ndwi_mean"],
            ["ndwi", "water index", "normalized difference water"],
        )
        nddi_col = _find_column(
            ["nddi", "nddi_value", "nddi_index", "nddi_mean"],
            ["nddi", "drought index"],
        )
        return {
            "time": time_col,
            "ndvi": ndvi_col,
            "ndwi": ndwi_col,
            "nddi": nddi_col,
        }
    return None


def render_preprocess_tab():
    st.subheader("Preprocessing and NDDI")
    ss = get_session_state()

    if "raw_df" not in ss:
        st.info("Please ingest data first in the Data tab.")
        return

    df = ss["raw_df"].copy()
    with st.expander("Preview raw data"):
        st.dataframe(df.head(10))
        st.caption("Column dtypes:")
        st.write(df.dtypes.astype(str).to_dict())

    # Simple mode
    st.caption("Column mapping")
    simple = st.checkbox("Simple mode (auto-detect columns)", value=True)
    cols = list(df.columns)
    preset = _preset_for_source(ss.get("source_name"), df)
    g_time, g_ndvi, g_ndwi, g_nddi, g_lat, g_lon = _guess_columns(df)
    if preset:
        g_time = preset.get("time", g_time)
        g_ndvi = preset.get("ndvi", g_ndvi)
        g_ndwi = preset.get("ndwi", g_ndwi)
        g_nddi = preset.get("nddi", g_nddi)
    if simple:
        time_col = g_time
        ndvi_col = g_ndvi
        ndwi_col = g_ndwi
        nddi_col = g_nddi
        lat_col = g_lat
        lon_col = g_lon
        st.write({"time": time_col, "ndvi": ndvi_col, "ndwi": ndwi_col, "nddi": nddi_col, "source": ss.get("source_name")})
        # If NDDI not present and we couldn't auto-detect NDVI/NDWI, ask just for those two.
        if (nddi_col is None) and (ndvi_col is None or ndwi_col is None):
            st.warning("Auto-detect couldn't find NDVI/NDWI. Please select them below.")
            ndvi_col = st.selectbox("NDVI column", options=[None] + cols)
            ndwi_col = st.selectbox("NDWI column", options=[None] + cols)
    else:
        time_idx = cols.index(g_time) if g_time in cols else 0
        ndvi_idx = ( [None] + cols ).index(g_ndvi) if g_ndvi in cols else 0
        ndwi_idx = ( [None] + cols ).index(g_ndwi) if g_ndwi in cols else 0
        nddi_idx = ( [None] + cols ).index(g_nddi) if g_nddi in cols else 0

        time_col = st.selectbox("Time column", options=cols, index=time_idx)
        ndvi_col = st.selectbox("NDVI column (optional if NDDI provided)", options=[None] + cols, index=ndvi_idx)
        ndwi_col = st.selectbox("NDWI column (optional if NDDI provided)", options=[None] + cols, index=ndwi_idx)
        nddi_col = st.selectbox("NDDI column (optional)", options=[None] + cols, index=nddi_idx)
        with st.expander("Optional geospatial columns"):
            lat_idx = ( [None] + cols ).index(g_lat) if g_lat in cols else 0
            lon_idx = ( [None] + cols ).index(g_lon) if g_lon in cols else 0
            lat_col = st.selectbox("Latitude column (optional)", options=[None] + cols, index=lat_idx)
            lon_col = st.selectbox("Longitude column (optional)", options=[None] + cols, index=lon_idx)

    target_freq = st.selectbox("Resample frequency", options=["D", "W", "MS"], index=0)
    agg_fn = st.selectbox("Aggregation", options=["mean", "median", "max", "min"], index=0)

    drought_thresh = st.slider("Drought threshold on NDDI (higher means drier)", min_value=-1.0, max_value=1.0, value=0.2, step=0.01)

    if st.button("Process", type="primary"):
        try:
            # Parse time
            if time_col not in df.columns:
                st.error(f"Time column '{time_col}' not found.")
                return
            # Parse time; if too many NaT, fallback to original order with a synthetic RangeIndex
            parsed = pd.to_datetime(df[time_col], errors="coerce")
            valid_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
            if valid_ratio < 0.5:
                st.warning("Time parsing failed for most rows; proceeding without resampling.")
                df = df.reset_index(drop=True)
                df.index.name = time_col
                use_resample = False
            else:
                df[time_col] = parsed
                df = df.dropna(subset=[time_col])
                df = df.sort_values(time_col)
                df = df.set_index(time_col)
                use_resample = True

            # Compute NDDI if needed (with NDVI-only fallback)
            if nddi_col is None or nddi_col == "None":
                if (ndvi_col is not None and ndvi_col != "None") and (ndwi_col in (None, "None")):
                    # Fallback: use NDVI as proxy for NDDI
                    if ndvi_col not in df.columns:
                        st.error("Selected NDVI column was not found in the data.")
                        return
                    df[ndvi_col] = _coerce_numeric_locale(df[ndvi_col])
                    df["NDDI"] = df[ndvi_col]
                    nddi_col_effective = "NDDI"
                    st.info("NDWI not provided; using NDVI as a proxy for NDDI for visualization and modeling.")
                else:
                    if (ndvi_col is None or ndvi_col == "None") or (ndwi_col is None or ndwi_col == "None"):
                        st.error("Provide NDVI and NDWI or an existing NDDI column.")
                        return
                    # Validate presence
                    if ndvi_col not in df.columns or ndwi_col not in df.columns:
                        st.error("Selected NDVI/NDWI columns were not found in the data.")
                        return
                    # Ensure NDVI/NDWI are numeric
                    df[ndvi_col] = _coerce_numeric_locale(df[ndvi_col])
                    df[ndwi_col] = _coerce_numeric_locale(df[ndwi_col])
                    df = compute_nddi_column(df, ndvi_col, ndwi_col, out_col="NDDI")
                    nddi_col_effective = "NDDI"
            else:
                # Validate presence
                if nddi_col not in df.columns:
                    st.error(f"Selected NDDI column '{nddi_col}' was not found in the data.")
                    return
                # Ensure NDDI is numeric
                df[nddi_col] = _coerce_numeric_locale(df[nddi_col])
                nddi_col_effective = nddi_col

            # Resample
            agg_map = {"mean": "mean", "median": "median", "max": "max", "min": "min"}
            # After coercion, detect numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                st.error("No numeric columns to process. Ensure NDVI/NDWI/NDDI are numeric.")
                return
            if use_resample:
                df_res = df[numeric_cols].resample(target_freq).agg(agg_map[agg_fn])
                if len(df_res) < 5:
                    st.warning("Resampling produced too few rows; using original (unresampled) data.")
                    df_res = df[numeric_cols].copy()
            else:
                df_res = df[numeric_cols].copy()
            if nddi_col_effective not in df_res.columns:
                st.error(
                    f"Selected NDDI column '{nddi_col_effective}' is not numeric or missing after resampling. "
                    "Pick the correct NDDI column, or select NDVI and NDWI so the app can compute NDDI."
                )
                return
            df_res = df_res.dropna(subset=[nddi_col_effective])

            # Persist selections
            ss["proc_df"] = df_res
            ss["meta"] = {
                "time_col": time_col,
                "nddi_col": nddi_col_effective,
                "lat_col": None if (lat_col in (None, "None")) else lat_col,
                "lon_col": None if (lon_col in (None, "None")) else lon_col,
                "drought_thresh": drought_thresh,
            }

            st.success(f"Processed: {len(df_res)} rows after resampling {target_freq}")
            df_plot = df_res[[nddi_col_effective]].copy()
            df_plot = _downsample_df(df_plot, max_points=2000)
            if len(df_res) > len(df_plot):
                st.caption(f"Downsampled chart to {len(df_plot)} points for performance (from {len(df_res)}).")
            st.line_chart(df_plot)
            if len(df_res) < 30:
                st.warning(
                    f"Only {len(df_res)} data points after resampling. Consider using a finer frequency "
                    f"(e.g., 'D' or 'W') or loading a longer time range for better model training."
                )
        except Exception as e:
            st.error(f"Processing failed: {e}")

    if "proc_df" in ss:
        st.success("Preprocessed data ready. Proceed to Model tab.")
