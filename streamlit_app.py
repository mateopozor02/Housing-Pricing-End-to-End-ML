"""
Streamlit UI for Housing Price Prediction - Holdout Explorer

A web interface for exploring model predictions on the holdout dataset.
Loads feature-engineered holdout data from AWS S3 and makes predictions via the FastAPI endpoint.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import boto3
import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Config
# ============================================================================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "housing-pricing-regression-data")
REGION = os.getenv("AWS_REGION", "us-east-2")

s3 = boto3.client("s3", region_name=REGION)

# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Utility Functions
# ============================================================================
def load_from_s3(key: str, local_path: str) -> str:
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        st.info(f"📥 Downloading {key} from S3…")
        try:
            s3.download_file(S3_BUCKET, key, str(local_path))
        except Exception as e:
            st.error(f"Failed to download {key} from S3: {e}")
            raise
    return str(local_path)


def batch_predictions(
    payload: list[dict],
    batch_size: int = 100,
    timeout: int = 120,
    max_workers: int = 5,
) -> list:
    """Make predictions in parallel batches to improve speed."""
    if not payload:
        return []

    total_batches = (len(payload) + batch_size - 1) // batch_size
    all_predictions = [None] * len(payload)  # Pre-allocate to maintain order

    progress_bar = st.progress(0)
    status_text = st.empty()
    lock = Lock()
    completed_count = 0

    def process_batch(batch_info: tuple) -> tuple:
        """Process a single batch and return results with position info."""
        batch_idx, batch_start, batch = batch_info
        try:
            resp = requests.post(API_URL, json=batch, timeout=timeout)
            resp.raise_for_status()
            batch_preds = resp.json().get("predictions", [])
            return batch_idx, batch_start, batch_preds, None
        except Exception as e:
            return batch_idx, batch_start, None, e

    # Prepare batch list
    batches = []
    for i, batch_start in enumerate(range(0, len(payload), batch_size)):
        batch_end = min(batch_start + batch_size, len(payload))
        batch = payload[batch_start:batch_end]
        batches.append((i + 1, batch_start, batch))

    # Process batches with threading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch): i for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            try:
                batch_idx, batch_start, batch_preds, error = future.result()
                if error:
                    st.error(f"Batch {batch_idx} failed: {error}")
                    raise error

                # Store predictions in correct positions
                for j, pred in enumerate(batch_preds):
                    all_predictions[batch_start + j] = pred

                with lock:
                    completed_count += 1
                    status_text.text(
                        f"Processing batches... {completed_count}/{total_batches} completed"
                    )
                    progress_bar.progress(completed_count / total_batches)

            except Exception as e:
                st.error(f"Error: {e}")
                raise

    status_text.empty()
    progress_bar.empty()

    return all_predictions


# ============================================================================
# Data Paths (cached locally from S3)
# ============================================================================
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "data/processed/feature_engineered_holdout_data.csv",
    "data/processed/feature_engineered_holdout_data.csv",
)
HOLDOUT_META_PATH = load_from_s3(
    "data/processed/cleaning_holdout_data.csv",
    "data/processed/cleaning_holdout_data.csv",
)


# ============================================================================
# Data Loading
# ============================================================================
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process holdout data from local cache."""
    try:
        # Load feature-engineered data (includes 'price' column for actuals)
        fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)

        # Load metadata (date, city_full, etc.)
        meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[
            ["date", "city_full"]
        ]

        logger.info(f"Loaded {len(fe)} feature-engineered records")
        logger.info(f"Loaded {len(meta)} metadata records")

        # Align lengths if they differ
        if len(fe) != len(meta):
            st.warning(
                "⚠️ Engineered and meta holdout lengths differ. Aligning by index."
            )
            min_len = min(len(fe), len(meta))
            fe = fe.iloc[:min_len].copy()
            meta = meta.iloc[:min_len].copy()

        # Create display dataframe with metadata
        disp = pd.DataFrame(index=fe.index)
        disp["date"] = meta["date"]
        disp["region"] = meta["city_full"]
        disp["year"] = disp["date"].dt.year
        disp["month"] = disp["date"].dt.month
        disp["actual_price"] = fe["price"] if "price" in fe.columns else None

        logger.info(f"Processing complete. Years: {sorted(disp['year'].unique())}")

        return fe, disp

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        st.error(f"Failed to load holdout data: {e}")
        raise


# Load data
fe_df, disp_df = load_data()

# ============================================================================
# Main UI
# ============================================================================
st.title("🏠 Housing Price Prediction — Holdout Explorer")
st.markdown("Explore model predictions on the holdout dataset with actual prices.")

# Display data summary
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Total Records", f"{len(disp_df):,}")
with col_info2:
    st.metric("Regions", disp_df["region"].nunique())
with col_info3:
    date_range = (
        f"{disp_df['date'].min().date().year} to {disp_df['date'].max().date().year}"
    )
    st.metric("Date Range", date_range)

st.markdown("---")

# Filters
st.subheader("Filters")
col1, col2, col3 = st.columns(3)

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

# Prediction button
if st.button("Show Predictions 🚀", width="stretch"):
    # Apply filters
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= disp_df["region"] == region

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(
            f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}** | Records: **{len(idx)}**"
        )

        # Prepare payload
        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            # Call API
            with st.spinner("Making predictions..."):
                resp = requests.post(API_URL, json=payload, timeout=120)
                resp.raise_for_status()
                out = resp.json()

            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            # Build results dataframe
            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(
                    float
                )

            # ====================================================================
            # Results Table
            # ====================================================================
            st.subheader("Predictions vs Actuals")
            view_display = view[["date", "region", "actual_price", "prediction"]].copy()
            view_display["actual_price"] = view_display["actual_price"].apply(
                lambda x: f"${x:,.0f}"
            )
            view_display["prediction"] = view_display["prediction"].apply(
                lambda x: f"${x:,.0f}"
            )
            view_display = view_display.reset_index(drop=True)
            view_display.columns = ["Date", "Region", "Actual Price", "Predicted Price"]

            st.dataframe(view_display, width="stretch")

            # ====================================================================
            # Performance Metrics
            # ====================================================================
            st.subheader("Performance Metrics")
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = (
                (view["prediction"] - view["actual_price"]).abs() / view["actual_price"]
            ).mean() * 100
            r2 = 1 - (
                ((view["prediction"] - view["actual_price"]) ** 2).sum()
                / ((view["actual_price"] - view["actual_price"].mean()) ** 2).sum()
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("MAE", f"${mae:,.0f}")
            with c2:
                st.metric("RMSE", f"${rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")
            with c4:
                st.metric("R² Score", f"{r2:.4f}")

            # ====================================================================
            # Yearly Trend Chart
            # ====================================================================
            st.subheader("Yearly Trend Analysis")

            if region == "All":
                yearly_data = disp_df[disp_df["year"] == year].copy()
                idx_all = yearly_data.index
                payload_all = fe_df.loc[idx_all].to_dict(orient="records")

                preds_all = batch_predictions(
                    payload_all, batch_size=1200, max_workers=8
                )

                yearly_data["prediction"] = pd.Series(
                    preds_all, index=yearly_data.index
                ).astype(float)
            else:
                yearly_data = disp_df[
                    (disp_df["year"] == year) & (disp_df["region"] == region)
                ].copy()
                idx_region = yearly_data.index
                payload_region = fe_df.loc[idx_region].to_dict(orient="records")

                preds_region = batch_predictions(
                    payload_region, batch_size=1200, max_workers=8
                )

                yearly_data["prediction"] = pd.Series(
                    preds_region, index=yearly_data.index
                ).astype(float)

            # Aggregate by month
            monthly_avg = (
                yearly_data.groupby("month")[["actual_price", "prediction"]]
                .mean()
                .reset_index()
            )

            # Create line chart
            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price ($)", "month": "Month", "variable": "Type"},
                title=f"Yearly Trend — {year}{'' if region == 'All' else f' — {region}'}",
            )

            # Rename legend
            fig.for_each_trace(
                lambda t: t.update(
                    name="Actual" if t.name == "actual_price" else "Predicted"
                )
            )

            # Add highlight for selected month
            fig.add_vrect(
                x0=month - 0.5,
                x1=month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, width="stretch")

            # ====================================================================
            # Price Distribution
            # ====================================================================
            st.subheader("Price Distribution")

            fig_dist = px.scatter(
                view,
                x="actual_price",
                y="prediction",
                hover_data=["date", "region"],
                labels={
                    "actual_price": "Actual Price ($)",
                    "prediction": "Predicted Price ($)",
                },
                title="Actual vs Predicted Prices",
            )

            # Add perfect prediction line
            min_price = min(view["actual_price"].min(), view["prediction"].min())
            max_price = max(view["actual_price"].max(), view["prediction"].max())
            fig_dist.add_shape(
                type="line",
                x0=min_price,
                y0=min_price,
                x1=max_price,
                y1=max_price,
                line=dict(dash="dash", color="red"),
                name="Perfect Prediction",
            )

            st.plotly_chart(fig_dist, width="stretch")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to API server. Make sure the FastAPI server is running at "
                f"{API_URL}"
            )
        except requests.exceptions.Timeout:
            st.error("❌ API request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ API call failed: {e}")
            logger.exception(e)
            st.exception(e)

else:
    st.info("👉 Choose filters and click **Show Predictions** to compute.")

# ============================================================================
# Sidebar Info
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown(
        f"""
    ### Housing Price Predictor
    
    **API**: {API_URL}
    
    **Holdout Dataset**:
    - Records: {len(disp_df):,}
    - Date Range: {disp_df['date'].min().date()} to {disp_df['date'].max().date()}
    - Regions: {disp_df['region'].nunique()}
    
    **Model**: LightGBM Regressor
    
    **Metrics Calculated**:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - Avg % Error: Average Percentage Error
    - R²: Coefficient of Determination
    """
    )
    st.markdown("---")
    st.markdown("🏠 **Housing Price Prediction** • Powered by ML")
