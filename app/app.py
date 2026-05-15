import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crime Prediction Heatmap",
    layout="wide"
)

st.title("Crime Prediction Heatmap")
st.markdown(
    "Interactive visualization of predicted crime hotspots"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    """
    Load and preprocess crime dataset.
    """
    df = pd.read_csv("../datasets/processed_data.csv")

    # Convert datetime column
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    from tensorflow.keras.initializers import Orthogonal

    model = tf.keras.models.load_model(
        "../models/model.keras",
        compile=False,
        custom_objects={"Orthogonal": Orthogonal}
    )

    return model


# Load resources
df = load_data()
model = load_model()

# =========================================================
# GRID SETUP
# =========================================================

# Grid dimensions
H = df["lat_bin"].max() + 1
W = df["lon_bin"].max() + 1

# Map each grid cell to average coordinates
grid_mapping = (
    df.groupby(["lat_bin", "lon_bin"])[["Latitude", "Longitude"]]
      .mean()
      .reset_index()
)

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("Prediction Settings")

# Dataset date boundaries
min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()

# ---------------------------------------------------------
# LOOKBACK WINDOW
# ---------------------------------------------------------

lookback = st.sidebar.slider(
    "Lookback Window (days)",
    min_value=7,
    max_value=60,
    value=24
)

# ---------------------------------------------------------
# COMPUTE EARLIEST VALID DATE
# ---------------------------------------------------------
#
# The model needs at least `lookback` days of historical
# data before the selected prediction start date.
#
# Example:
# If lookback = 24 days,
# and dataset starts at 2012-01-01,
# then earliest valid prediction date is:
#
# 2012-01-01 + 24 days
#
# This prevents users from selecting dates that
# don't have enough historical context.
# ---------------------------------------------------------

earliest_valid_date = (
    pd.to_datetime(min_date) +
    pd.Timedelta(days=lookback)
).date()

# ---------------------------------------------------------
# DATE SELECTION
# ---------------------------------------------------------
#
# We restrict the minimum selectable date
# so users cannot choose invalid dates.
# ---------------------------------------------------------

start_date = st.sidebar.date_input(
    "Prediction Start Date",
    value=max_date,
    min_value=earliest_valid_date,
    max_value=max_date
)

# ---------------------------------------------------------
# NUMBER OF FUTURE DAYS TO PREDICT
# ---------------------------------------------------------

n_days = st.sidebar.slider(
    "Days to Predict",
    min_value=1,
    max_value=180,
    value=30
)

# ---------------------------------------------------------
# MINIMUM PREDICTION THRESHOLD
# ---------------------------------------------------------
#
# Removes low-confidence predictions from visualization.
# ---------------------------------------------------------

threshold = st.sidebar.slider(
    "Minimum Prediction Value",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01
)

# =========================================================
# EXTRA VALIDATION SAFEGUARD
# =========================================================
#
# Even though we restrict date selection,
# we keep this validation as a backup safety check.
# =========================================================

selected_date = pd.to_datetime(start_date)

required_start = (
    selected_date -
    pd.Timedelta(days=lookback)
)

available_start = df["datetime"].min()

enough_history = required_start >= available_start

if not enough_history:

    st.error(
        f"""
        Not enough historical data available.

        Please select a date after:
        {earliest_valid_date}
        """
    )

    st.stop()

# =========================================================
# PREPARE INPUT FRAMES
# =========================================================

def prepare_frames_for_prediction(
    start_date,
    lookback=24
):
    """
    Prepare historical binary occupancy grids.

    Returns:
        frames: list of HxW arrays
        time_periods: corresponding dates
    """

    start_date = pd.to_datetime(start_date)

    # Start of historical window
    lookback_start = (
        start_date -
        pd.Timedelta(days=lookback)
    )

    # Keep only required historical period
    df_range = df[
        df["datetime"] >= lookback_start
    ].copy()

    # Group by day
    df_range["day"] = (
        df_range["datetime"].dt.floor("D")
    )

    # Keep unique active crime cells
    crime_cells = df_range[
        ["day", "lat_bin", "lon_bin"]
    ].drop_duplicates()

    # Sort all days
    time_periods = sorted(
        crime_cells["day"].unique()
    )

    # Map date -> tensor index
    period_to_idx = {
        p: i for i, p in enumerate(time_periods)
    }

    # Create empty tensor
    tensor = np.zeros(
        (len(time_periods), H, W)
    )

    # Fill tensor with active cells
    for _, row in crime_cells.iterrows():

        t = period_to_idx[row["day"]]
        x = int(row["lat_bin"])
        y = int(row["lon_bin"])

        tensor[t, x, y] = 1

    # Convert tensor into list of frames
    frames = [
        tensor[i]
        for i in range(len(time_periods))
    ]

    return frames, time_periods

# =========================================================
# FUTURE PREDICTION
# =========================================================

def predict_future(
    start_date,
    n_days,
    lookback=24
):
    """
    Predict future crime maps.

    Returns:
        predictions: list of HxW arrays
        pred_dates: corresponding dates
    """

    # Prepare historical frames
    frames, frame_dates = (
        prepare_frames_for_prediction(
            start_date,
            lookback
        )
    )

    # Ensure enough history exists
    if len(frames) < lookback:

        raise ValueError(
            f"Not enough historical data "
            f"for lookback={lookback}"
        )

    # Shape:
    # (1, time, H, W, channels)
    X = np.stack(frames[-lookback:])[
        np.newaxis, ..., np.newaxis
    ]

    predictions = []
    pred_dates = []

    current_date = pd.to_datetime(start_date)

    # Streamlit progress bar
    progress_bar = st.progress(0)

    # Predict sequentially
    for i in range(n_days):

        # Predict next frame
        pred = model.predict(
            X,
            verbose=0
        )

        pred_grid = pred[0, :, :, 0]

        predictions.append(pred_grid)
        pred_dates.append(current_date)

        # Move to next day
        current_date += pd.Timedelta(days=1)

        # Append prediction into rolling window
        next_frame = pred[
            :, np.newaxis, :, :, :
        ]

        X = np.concatenate(
            [
                X[:, 1:, :, :, :],
                next_frame
            ],
            axis=1
        )

        # Update progress bar
        progress_bar.progress(
            (i + 1) / n_days
        )

    return predictions, pred_dates

# =========================================================
# MAP GRID PREDICTIONS TO COORDINATES
# =========================================================

def map_to_coordinates(grid_pred):
    """
    Convert prediction grid into geographic points.
    """

    points = []

    for _, row in grid_mapping.iterrows():

        i = int(row["lat_bin"])
        j = int(row["lon_bin"])

        points.append({
            "lat": row["Latitude"],
            "lon": row["Longitude"],
            "value": grid_pred[i, j]
        })

    return points

# =========================================================
# CONVERT PREDICTIONS TO DATAFRAME
# =========================================================

def predictions_to_df(
    predictions,
    pred_dates
):
    """
    Convert all predictions into dataframe
    suitable for Plotly visualization.
    """

    rows = []

    for date, frame in zip(
        pred_dates,
        predictions
    ):

        points = map_to_coordinates(frame)

        for p in points:

            # Ignore weak predictions
            if p["value"] >= threshold:

                rows.append({
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "value": float(p["value"]),
                    "time": str(date.date())
                })

    return pd.DataFrame(rows)

# =========================================================
# RUN PREDICTIONS
# =========================================================

if st.button("Run Prediction"):

    with st.spinner(
        "Generating future predictions..."
    ):

        predictions, pred_dates = (
            predict_future(
                start_date,
                n_days,
                lookback
            )
        )

        df_plot = predictions_to_df(
            predictions,
            pred_dates
        )

    st.success("Prediction completed!")

    # =====================================================
    # CREATE ANIMATED HEATMAP
    # =====================================================

    fig = px.density_mapbox(
        df_plot,
        lat="lat",
        lon="lon",
        z="value",
        radius=8,
        animation_frame="time",
        mapbox_style="carto-positron",
        zoom=10,
        center={
            "lat": df_plot["lat"].mean(),
            "lon": df_plot["lon"].mean()
        },
        height=800,
        color_continuous_scale= "Inferno"
    )

    # Remove extra margins
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        )
    )

    # Display map
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # SHOW RAW DATA
    # =====================================================

    with st.expander("Show Prediction Data"):
        st.dataframe(df_plot)