
# app.py
# Streamlit version of "Planet’s Fever Chart — Climate Data Storytelling (NASA GISTEMP)"
# Converts the original Colab notebook to an interactive Streamlit app.
# Source inspiration: user-provided notebook/script.

import io
import re
import os
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------
# Page setup
# --------------------------
st.set_page_config(page_title="Planet's Fever Chart — Streamlit", layout="wide")
st.title("🌍 Planet’s Fever Chart — Climate Data Storytelling (NASA GISTEMP)")

with st.expander("About this app"):
    st.markdown("""
This app visualizes global temperature **anomalies** (NASA GISTEMP) as a "fever chart,"
adds smoothing and threshold controls, and (optionally) produces a simple forecast using Prophet.

**Tips**
- Use the sidebar to select **data source**, **year range**, **smoothing**, and **threshold**.
- Forecasting is optional and may take a bit longer to compute on first run.
    """)

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("Controls")

DEFAULT_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

data_mode = st.sidebar.radio(
    "Data input",
    ["Download from NASA (default)", "Upload CSV"],
    index=0
)

uploaded = None
if data_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload GLB.Ts+dSST.csv", type=["csv"])

smooth_opt = st.sidebar.selectbox("Smoothing", ["None", "5-year", "10-year"], index=2)
threshold_label = st.sidebar.selectbox("Threshold line", ["None", "+1.5°C", "+2.0°C"], index=1)

show_seasons = st.sidebar.checkbox("Show seasonal anomalies (DJF/MAM/JJA/SON)", value=True)
enable_forecast = st.sidebar.checkbox("Enable Prophet forecast (experimental)", value=False)
show_co2 = st.sidebar.checkbox("Show CO₂ chart (Mauna Loa)", value=False)

# --------------------------
# Data loading
# --------------------------
@st.cache_data(show_spinner=True)
def load_gistemp_csv_from_url(url: str) -> pd.DataFrame:
    # Load raw CSV with a 1-line header to skip
    df = pd.read_csv(url, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    # Keep rows with a 4-digit Year
    df = df[df["Year"].astype(str).str.match(r"^\d{4}$", na=False)].copy()
    df["Year"] = df["Year"].astype(int)
    # Numeric coercion
    for c in df.columns:
        if c != "Year":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=True)
def load_gistemp_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Year"].astype(str).str.match(r"^\d{4}$", na=False)].copy()
    df["Year"] = df["Year"].astype(int)
    for c in df.columns:
        if c != "Year":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if uploaded is not None:
    df_raw = load_gistemp_csv_from_bytes(uploaded.getvalue())
else:
    # Attempt to load from NASA URL
    try:
        df_raw = load_gistemp_csv_from_url(DEFAULT_URL)
    except Exception as e:
        st.error("Failed to load from NASA. Please upload the CSV manually.")
        st.stop()

# --------------------------
# Prepare anomaly series
# --------------------------
def prepare_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "J-D" not in d.columns:
        st.error("CSV missing 'J-D' (annual mean) column.")
        st.stop()
    d = d.dropna(subset=["J-D"]).copy()
    d = d[d["Year"] >= 1880]
    d = d.rename(columns={"J-D": "Anomaly"})
    d = d[["Year", "Anomaly"]].reset_index(drop=True)
    # Rolling means
    d["Anomaly_5yr"] = d["Anomaly"].rolling(window=5, center=True, min_periods=1).mean()
    d["Anomaly_10yr"] = d["Anomaly"].rolling(window=10, center=True, min_periods=1).mean()
    # Absolute temperature (approximate baseline 14°C for 1951–1980)
    BASELINE_C = 14.0
    d["Absolute_Temp"] = BASELINE_C + d["Anomaly"]
    return d

df = prepare_anomaly(df_raw)

years = df["Year"].tolist()
ymin, ymax = int(min(years)), int(max(years))

st.sidebar.markdown("---")
start_year, end_year = st.sidebar.select_slider(
    "Year range",
    options=list(range(ymin, ymax + 1)),
    value=(ymin, ymax)
)

# Apply filters
df_range = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].copy()
if df_range.empty:
    st.warning("No data in the selected range.")
    st.stop()

# Determine smoothing and threshold numeric values
smooth_k = {"None": 0, "5-year": 5, "10-year": 10}[smooth_opt]
threshold_val = {"None": None, "+1.5°C": 1.5, "+2.0°C": 2.0}[threshold_label]

# --------------------------
# Fever chart
# --------------------------
def render_fever_chart(d: pd.DataFrame, smooth_k: int, threshold_val):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=d["Year"], y=d["Anomaly"], mode="lines",
        line=dict(color="firebrick", width=2),
        name="Annual anomaly"
    ))

    if smooth_k and smooth_k > 1:
        label = f"{smooth_k}-year smooth"
        sm = d["Anomaly"].rolling(window=int(smooth_k), center=True, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=d["Year"], y=sm, mode="lines",
            line=dict(color="black", width=3),
            name=label
        ))

    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    if threshold_val is not None:
        fig.add_hline(y=float(threshold_val), line_dash="dot", line_color="blue",
                      annotation_text=f"+{threshold_val:.1f}°C target")

    # annotate last point
    last_row = d.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_row["Year"]], y=[last_row["Anomaly"]],
        mode="markers+text", marker=dict(size=8),
        text=[f"{last_row['Anomaly']:.2f}°C"], textposition="top center",
        name="Latest year"
    ))

    fig.update_layout(
        title="Global Temperature Anomaly (°C) — The Planet’s Fever Chart",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C vs 1951–1980)",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", x=0, y=1.1)
    )
    return fig

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Fever Chart")
    st.plotly_chart(render_fever_chart(df_range, smooth_k, threshold_val), use_container_width=True)

with c2:
    st.markdown("**Latest values**")
    st.metric(label=f"Last year in range ({int(df_range['Year'].iloc[-1])})",
              value=f"{float(df_range['Anomaly'].iloc[-1]):.2f} °C")
    st.metric(label="Range mean (°C)",
              value=f"{float(df_range['Anomaly'].mean()):.2f}")
    st.metric(label="Range trend (last − first, °C)",
              value=f"{float(df_range['Anomaly'].iloc[-1] - df_range['Anomaly'].iloc[0]):.2f}")

# --------------------------
# Seasonal anomalies (optional)
# --------------------------
if show_seasons:
    # We need the seasonal columns from the original CSV (DJF/MAM/JJA/SON)
    seas = df_raw.copy()
    for c in ["DJF", "MAM", "JJA", "SON"]:
        seas[c] = pd.to_numeric(seas.get(c), errors="coerce")
    seas = seas[(seas["Year"] >= start_year) & (seas["Year"] <= end_year)]

    fig = go.Figure()
    for label in ["DJF", "MAM", "JJA", "SON"]:
        if label in seas.columns:
            fig.add_trace(go.Scatter(x=seas["Year"], y=seas[label], mode="lines", name=label))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Seasonal Anomalies (°C)",
        xaxis_title="Year", yaxis_title="Temperature Anomaly (°C)",
        template="plotly_white", height=420
    )
    st.subheader("Seasonal Anomalies")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Forecast (optional)
# --------------------------
if enable_forecast:
    st.subheader("Forecast to 2100 (Prophet, experimental)")
    try:
        from prophet import Prophet  # type: ignore
        # Prepare data for Prophet
        prophet_df = df.copy()[["Year", "Anomaly"]]
        prophet_df = prophet_df.dropna().copy()
        prophet_df["ds"] = pd.to_datetime(prophet_df["Year"].astype(int), format="%Y")
        prophet_df["y"] = prophet_df["Anomaly"]
        prophet_df = prophet_df[["ds", "y"]].sort_values("ds")

        m = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode="additive",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.68
        )
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=max(0, 2100 - prophet_df["ds"].dt.year.max()), freq="YS")
        fcst = m.predict(future)
        fcst["Year"] = fcst["ds"].dt.year

        # Identify 1.5°C exceed year (after last observed year)
        last_obs_year = int(prophet_df["ds"].dt.year.max())
        exceed = fcst[(fcst["yhat"] >= 1.5) & (fcst["Year"] > last_obs_year)]
        exceed_year = int(exceed.iloc[0]["Year"]) if not exceed.empty else None

        # Plot
        figf = go.Figure()
        # Observed
        figf.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="markers",
                                  marker=dict(size=4), name="Observed"))
        # Forecast
        figf.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], mode="lines",
                                  line=dict(width=2), name="Forecast"))
        # Future highlight
        mask_future = fcst["Year"] > last_obs_year
        figf.add_trace(go.Scatter(x=fcst.loc[mask_future, "ds"],
                                  y=fcst.loc[mask_future, "yhat"],
                                  mode="lines",
                                  line=dict(width=3),
                                  name="Forecast (future)"))
        # Uncertainty band (future only)
        figf.add_traces([
            go.Scatter(
                x=pd.concat([fcst.loc[mask_future, "ds"], fcst.loc[mask_future, "ds"][::-1]]),
                y=pd.concat([fcst.loc[mask_future, "yhat_upper"],
                             fcst.loc[mask_future, "yhat_lower"][::-1]]),
                fill="toself",
                line=dict(color="rgba(253, 146, 85, 0.2)"),
                fillcolor="rgba(253, 146, 85, 0.25)",
                name="±1 SD (68% CI)",
                showlegend=True
            )
        ])
        figf.add_hline(y=0, line_dash="dash", line_color="gray")
        figf.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="+1.5°C")

        if exceed_year:
            figf.add_vline(x=pd.Timestamp(year=exceed_year, month=1, day=1),
                           line_dash="dot", line_color="red",
                           annotation_text=f"1.5°C exceeded ~{exceed_year}")

        figf.update_layout(
            title="Global Temperature Anomaly Forecast (to 2100)",
            xaxis_title="Year", yaxis_title="Anomaly (°C)",
            template="plotly_white", height=520
        )
        st.plotly_chart(figf, use_container_width=True)

    except Exception as e:
        st.info("Prophet is not installed or failed to run. Install with `pip install prophet` and try again.")
        st.exception(e)

# --------------------------
# CO₂ chart (optional)
# --------------------------
if show_co2:
    st.subheader("Atmospheric CO₂ — Mauna Loa (annual mean)")
    CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"
    try:
        df_co2 = pd.read_csv(CO2_URL, comment="#", header=None, names=["year","mean","uncertainty"])
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=df_co2["year"], y=df_co2["mean"], mode="lines+markers",
                                  name="CO₂ (ppm)"))
        figc.update_layout(
            title="Atmospheric CO₂ Concentrations at Mauna Loa (1959–present)",
            xaxis_title="Year", yaxis_title="CO₂ (ppm)",
            template="plotly_white", height=420
        )
        st.plotly_chart(figc, use_container_width=True)

        start_co2 = float(df_co2["mean"].iloc[0])
        end_co2 = float(df_co2["mean"].iloc[-1])
        inc = end_co2 - start_co2
        pct = inc / start_co2 * 100
        st.caption(f"Increase: {inc:.2f} ppm ({pct:.1f}%) — {int(df_co2['year'].iloc[0])} to {int(df_co2['year'].iloc[-1])}")

    except Exception as e:
        st.warning("Failed to load CO₂ data. Check your internet connection.")
        st.exception(e)

st.markdown("---")
st.markdown("Data: NASA GISTEMP; CO₂ from NOAA Mauna Loa. App converted from a Colab notebook to Streamlit.")
