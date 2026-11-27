"""
Smart Electricity Consumption Analytics and Demand Forecasting Dashboard.
Built with Streamlit and Plotly.
Styled with a Midnight Obsidian & Electric Indigo-Violet SaaS Design System.
Supports default Low Carbon London telemetry plus user-uploaded custom datasets.
"""

import os
import sys
import json
from pathlib import Path

# Add project root directory to sys.path so imports work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.metrics import (
    render_metric_card,
    render_header_banner,
    render_top_navbar,
    render_feature_strip,
)
from dashboard.components.charts import (
    create_hourly_profile_chart,
    create_daily_trend_chart,
    create_weekday_weekend_chart,
    create_monthly_chart,
    create_actual_vs_predicted_chart,
    create_future_forecast_chart,
    create_feature_importance_chart,
    create_anomaly_scatter_chart,
    create_anomaly_hourly_dist_chart,
    create_household_profile_chart,
)
from src.utils.config import load_config, get_project_root
from src.analytics.custom_pipeline import (
    inspect_uploaded_file,
    run_custom_dataset_pipeline,
    CUSTOM_SUMMARY_SUBDIR,
)

# Page configuration
st.set_page_config(
    page_title="SECA - Smart Electricity Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Midnight Obsidian & Electric Indigo-Violet SaaS Theme
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #06060d !important;
            color: #f4f4f8 !important;
        }

        /* Ambient Indigo-Violet Top & Edge Glow */
        .stApp {
            background-image:
                radial-gradient(ellipse 80% 38% at 50% -6%, rgba(79, 70, 229, 0.28), transparent 72%),
                radial-gradient(ellipse 40% 30% at 95% 20%, rgba(109, 40, 217, 0.12), transparent 70%),
                radial-gradient(ellipse 40% 30% at 5% 60%, rgba(59, 40, 204, 0.10), transparent 70%) !important;
            background-attachment: fixed !important;
        }

        .main .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1420px;
        }

        /* Hide default Streamlit multipage nav and slim header */
        div[data-testid="stSidebarNav"] { display: none; }
        header[data-testid="stHeader"] {
            background: rgba(6, 6, 13, 0.65) !important;
            backdrop-filter: blur(10px);
        }

        /* Sidebar Surface */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090a15 0%, #06060d 100%) !important;
            border-right: 1px solid rgba(139, 92, 246, 0.16) !important;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            color: #a5b4fc !important;
            font-weight: 600 !important;
            font-size: 12px !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(15, 16, 33, 0.65);
            border: 1px solid rgba(139, 92, 246, 0.14);
            border-radius: 10px;
            padding: 8px 12px;
            margin-bottom: 5px;
            transition: all 0.2s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            border-color: rgba(139, 92, 246, 0.45);
            background: rgba(59, 40, 204, 0.20);
        }

        /* Top Capsule Navbar */
        .seca-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            background: rgba(11, 12, 24, 0.88);
            border: 1px solid rgba(139, 92, 246, 0.22);
            border-radius: 9999px;
            padding: 10px 18px;
            margin-bottom: 20px;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
        }
        .seca-topbar-left, .seca-topbar-right {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .seca-brand-mark {
            font-weight: 800;
            font-size: 16px;
            letter-spacing: 0.08em;
            color: #ffffff;
            padding: 2px 8px;
        }
        .seca-topbar-divider {
            width: 1px;
            height: 18px;
            background: rgba(139, 92, 246, 0.25);
        }
        .seca-nav-pill {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #94a3b8;
            padding: 5px 11px;
            border-radius: 9999px;
        }
        .seca-nav-pill.active {
            color: #ffffff;
            background: rgba(99, 102, 241, 0.22);
            border: 1px solid rgba(139, 92, 246, 0.45);
        }
        .seca-status-chip {
            font-size: 11px;
            font-weight: 600;
            color: #c4b5fd;
            background: rgba(17, 19, 38, 0.9);
            border: 1px solid rgba(139, 92, 246, 0.22);
            padding: 5px 12px;
            border-radius: 9999px;
        }
        .seca-cta-pill {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: #ffffff;
            background: linear-gradient(90deg, #3b28cc 0%, #6366f1 100%);
            padding: 6px 14px;
            border-radius: 9999px;
            box-shadow: 0 0 18px rgba(99, 102, 241, 0.45);
        }

        /* Hero Banner Card */
        .seca-hero-card {
            background: radial-gradient(circle at 82% 20%, rgba(109, 40, 217, 0.55) 0%, rgba(39, 28, 128, 0.65) 40%, rgba(11, 12, 24, 0.96) 100%);
            border: 1px solid rgba(139, 92, 246, 0.35);
            border-radius: 18px;
            padding: 26px 30px;
            margin-bottom: 22px;
            box-shadow: 0 14px 40px rgba(34, 24, 118, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        }
        .seca-hero-inner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }
        .seca-hero-eyebrow {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #a5b4fc;
            margin-bottom: 6px;
        }
        .seca-hero-title {
            color: #ffffff !important;
            margin: 0 !important;
            font-size: 28px !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }
        .seca-hero-subtitle {
            color: #cbd5e1;
            margin: 8px 0 0 0;
            font-size: 14px;
            font-weight: 400;
        }
        .seca-hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(10, 11, 24, 0.65);
            border: 1px solid rgba(167, 139, 250, 0.45);
            color: #e0e7ff;
            padding: 7px 14px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .seca-hero-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #a78bfa;
            box-shadow: 0 0 8px #a78bfa;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(16, 17, 34, 0.92) 0%, rgba(10, 11, 22, 0.96) 100%) !important;
            padding: 18px 18px !important;
            border-radius: 16px !important;
            border: 1px solid rgba(139, 92, 246, 0.20) !important;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06);
            transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(167, 139, 250, 0.55) !important;
            box-shadow: 0 8px 28px rgba(79, 70, 229, 0.22);
            transform: translateY(-2px);
        }
        div[data-testid="stMetric"] label {
            color: #94a3b8 !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 24px !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            color: #a78bfa !important;
            font-weight: 600 !important;
        }

        /* Feature Pill Chips */
        .seca-chip {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(12, 13, 25, 0.9);
            border: 1px solid rgba(139, 92, 246, 0.18);
            border-radius: 12px;
            padding: 11px 14px;
            margin-bottom: 16px;
            color: #cbd5e1;
            font-size: 12px;
            font-weight: 600;
        }
        .seca-chip-active {
            background: linear-gradient(135deg, rgba(59, 40, 204, 0.85) 0%, rgba(99, 102, 241, 0.65) 100%);
            border-color: rgba(167, 139, 250, 0.65);
            color: #ffffff;
            box-shadow: 0 6px 24px rgba(79, 70, 229, 0.35);
        }
        .seca-chip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: rgba(139, 92, 246, 0.22);
            color: #c4b5fd;
            font-size: 11px;
        }
        .seca-chip-active .seca-chip-icon {
            background: rgba(255, 255, 255, 0.20);
            color: #ffffff;
        }

        /* Chart & File Uploader Containers */
        div[data-testid="stPlotlyChart"] {
            background: linear-gradient(180deg, rgba(13, 14, 28, 0.88) 0%, rgba(9, 10, 20, 0.95) 100%);
            border: 1px solid rgba(139, 92, 246, 0.18);
            border-radius: 16px;
            padding: 12px 14px 6px 14px;
            box-shadow: 0 8px 26px rgba(0, 0, 0, 0.35);
        }
        section[data-testid="stFileUploaderDropzone"] {
            background: rgba(15, 16, 33, 0.78) !important;
            border: 1.5px dashed rgba(139, 92, 246, 0.45) !important;
            border-radius: 14px !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: #94a3b8 !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            border-radius: 10px 10px 0 0 !important;
            padding: 10px 18px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff !important;
            background: rgba(99, 102, 241, 0.16) !important;
            border-bottom: 2px solid #8b5cf6 !important;
        }

        /* Buttons & Download Buttons */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, #3b28cc 0%, #6366f1 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(167, 139, 250, 0.45) !important;
            border-radius: 9999px !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            padding: 0.5rem 1.4rem !important;
            box-shadow: 0 4px 18px rgba(79, 70, 229, 0.35) !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            box-shadow: 0 6px 24px rgba(139, 92, 246, 0.55) !important;
            filter: brightness(1.08);
        }

        /* Info / Alert Boxes */
        div[data-testid="stAlert"] {
            background: rgba(18, 19, 41, 0.85) !important;
            border: 1px solid rgba(139, 92, 246, 0.30) !important;
            border-radius: 14px !important;
            color: #e2e8f0 !important;
        }

        /* Dividers & Subheaders */
        hr {
            border-color: rgba(139, 92, 246, 0.15) !important;
        }
        h2, h3 {
            color: #f8fafc !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_dashboard_data(sample_mode: bool = False, source_mode: str = None):
    """
    Cached data loader reading precomputed analytical summaries.
    Guarantees sub-second dashboard updates without rerunning Spark operations.
    Supports 'full', 'sample', and 'custom' sources.
    """
    root = get_project_root()
    effective_mode = source_mode or ("sample" if sample_mode else "full")

    if effective_mode == "custom":
        base_dir = root / CUSTOM_SUMMARY_SUBDIR
        if not (base_dir / "overview_metrics.json").exists():
            return None
    elif effective_mode == "sample":
        base_dir = root / "dataset/processed/sample_analytics_summary"
        if not base_dir.exists():
            return None
    else:
        base_dir = root / "dataset/processed/analytics_summary"
        if not base_dir.exists():
            base_dir = root / "dataset/processed/sample_analytics_summary"
            if not base_dir.exists():
                return None

    data = {}
    try:
        with open(base_dir / "overview_metrics.json", "r") as f:
            data["overview"] = json.load(f)

        data["hourly"] = pd.read_csv(base_dir / "hourly_consumption.csv")
        data["daily"] = pd.read_csv(base_dir / "daily_consumption.csv")
        data["monthly"] = pd.read_csv(base_dir / "monthly_consumption.csv")
        data["weekday_weekend"] = pd.read_csv(base_dir / "weekday_vs_weekend.csv")
        data["peaks"] = pd.read_csv(base_dir / "peak_demand_periods.csv")
        data["hh_summary"] = pd.read_csv(base_dir / "household_summary.csv")
        data["hh_hourly"] = pd.read_csv(base_dir / "household_hourly_profile.csv")

        fc_eval_file = base_dir / "forecasting_evaluation.json"
        if fc_eval_file.exists():
            with open(fc_eval_file, "r") as f:
                data["forecast_eval"] = json.load(f)
        else:
            data["forecast_eval"] = None

        fc_comp_file = base_dir / "forecast_comparison.csv"
        data["forecast_comp"] = pd.read_csv(fc_comp_file) if fc_comp_file.exists() else None

        fc_future_file = base_dir / "future_forecast.csv"
        data["forecast_future"] = pd.read_csv(fc_future_file) if fc_future_file.exists() else None

        feat_imp_file = base_dir / "feature_importance.csv"
        data["feature_importance"] = pd.read_csv(feat_imp_file) if feat_imp_file.exists() else None

        anom_sum_file = base_dir / "anomaly_summary.json"
        if anom_sum_file.exists():
            with open(anom_sum_file, "r") as f:
                data["anomaly_summary"] = json.load(f)
        else:
            data["anomaly_summary"] = None

        anom_rec_file = base_dir / "detected_anomalies.csv"
        data["anomalies"] = pd.read_csv(anom_rec_file) if anom_rec_file.exists() else None

        anom_dist_file = base_dir / "anomaly_hourly_distribution.csv"
        data["anomaly_dist"] = pd.read_csv(anom_dist_file) if anom_dist_file.exists() else None

        custom_meta_file = base_dir / "custom_metadata.json"
        if custom_meta_file.exists():
            with open(custom_meta_file, "r") as f:
                data["custom_meta"] = json.load(f)
        else:
            data["custom_meta"] = None

    except Exception as e:
        st.error(f"Error loading analytical data: {e}")
        return None

    return data


def render_custom_upload_panel(has_existing_custom: bool):
    """Renders the custom dataset uploader, column mapper, and pipeline runner."""
    root = get_project_root()
    upload_dir = root / "dataset/custom_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    with st.expander(
        "📤 Upload & Analyze Your Own Dataset (CSV / Parquet)",
        expanded=not has_existing_custom
    ):
        st.markdown(
            "Upload any smart-meter or electricity consumption `.csv` or `.parquet` file. "
            "PySpark will automatically clean, aggregate, forecast demand, and detect anomalies on your data."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV or Parquet file",
            type=["csv", "parquet"],
            key="custom_dataset_uploader"
        )

        if uploaded_file is not None:
            safe_name = Path(uploaded_file.name).name
            dest_path = upload_dir / safe_name

            # Persist uploaded file cleanly if new or changed
            if (
                st.session_state.get("last_uploaded_name") != safe_name
                or st.session_state.get("last_uploaded_size") != uploaded_file.size
            ):
                with open(dest_path, "wb") as out_f:
                    out_f.write(uploaded_file.getbuffer())
                st.session_state["last_uploaded_name"] = safe_name
                st.session_state["last_uploaded_size"] = uploaded_file.size
                st.session_state["custom_inspection"] = inspect_uploaded_file(dest_path)

            insp = st.session_state.get("custom_inspection") or inspect_uploaded_file(dest_path)
            cols = insp["columns"]
            sugg = insp["suggested_mapping"]

            st.caption(
                f"Detected **{insp['approx_rows']:,} rows** · **{len(cols)} columns** · "
                f"**{insp['file_size_mb']} MB**"
            )
            st.dataframe(insp["preview"], use_container_width=True)

            st.markdown("##### Map Dataset Columns")
            m1, m2, m3, m4, m5 = st.columns([2, 2, 2, 2, 1.5])

            ts_idx = cols.index(sugg["timestamp"]) if sugg["timestamp"] in cols else 0
            kwh_idx = cols.index(sugg["consumption"]) if sugg["consumption"] in cols else min(len(cols) - 1, 1)
            opt_cols = ["(None / Single Meter)"] + cols
            hh_idx = opt_cols.index(sugg["household_id"]) if sugg["household_id"] in opt_cols else 0
            tar_idx = opt_cols.index(sugg["tariff_type"]) if sugg["tariff_type"] in opt_cols else 0

            with m1:
                ts_col = st.selectbox("Timestamp Column *", cols, index=ts_idx)
            with m2:
                kwh_col = st.selectbox("Consumption (kWh) Column *", cols, index=kwh_idx)
            with m3:
                hh_choice = st.selectbox("Household / Meter ID", opt_cols, index=hh_idx)
            with m4:
                tariff_choice = st.selectbox("Tariff Type", opt_cols, index=tar_idx)
            with m5:
                max_kwh = st.number_input("Max Valid kWh", min_value=1.0, value=10000.0, step=100.0)

            if st.button("⚡ Run Full Analytics Pipeline", use_container_width=True):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                def _update_progress(msg: str, pct: int):
                    progress_bar.progress(min(max(pct, 0), 100))
                    status_placeholder.info(f"**[{pct}%]** {msg}")

                try:
                    run_custom_dataset_pipeline(
                        file_path=dest_path,
                        timestamp_col=ts_col,
                        consumption_col=kwh_col,
                        household_col=None if hh_choice == "(None / Single Meter)" else hh_choice,
                        tariff_col=None if tariff_choice == "(None / Single Meter)" else tariff_choice,
                        max_kwh=float(max_kwh),
                        progress_callback=_update_progress,
                    )
                    load_dashboard_data.clear()
                    st.success("✅ Full analytics complete! Explore all 6 sections using the sidebar.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")


def main():
    load_config()

    # Sidebar Navigation
    st.sidebar.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(59, 40, 204, 0.35), rgba(15, 16, 33, 0.9));
                    border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 14px;
                    text-align: center; padding: 16px 12px; margin-bottom: 18px;">
            <h2 style="color: #ffffff; margin: 0; font-size: 20px; font-weight: 800; letter-spacing: 0.06em;">⚡ SECA</h2>
            <p style="color: #a5b4fc; font-size: 11px; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.08em;">
                Smart Energy Analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    data_source_mode = st.sidebar.radio(
        "Dataset Scope",
        [
            "Full Dataset (5M Records)",
            "Sample Dataset (36K Records)",
            "Upload Custom Dataset",
        ],
        index=0
    )

    if data_source_mode == "Upload Custom Dataset":
        source_mode = "custom"
    elif data_source_mode == "Sample Dataset (36K Records)":
        source_mode = "sample"
    else:
        source_mode = "full"

    st.sidebar.markdown("---")
    navigation_page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Overview",
            "📈 Consumption Analytics",
            "🔮 Demand Forecasting",
            "🚨 Anomaly Detection",
            "🏠 Household Analysis",
            "🔍 Data Explorer & Export"
        ]
    )

    st.sidebar.markdown("---")
    if source_mode == "custom":
        st.sidebar.info(
            "**Dataset:** User Uploaded Data  \n"
            "**Engine:** Apache Spark 4.3.0 + Parquet  \n"
            "**Pipeline:** Automated Ingestion, SQL, ML & Anomaly Detection"
        )
    else:
        st.sidebar.info(
            "**Dataset:** Low Carbon London  \n"
            "**Source:** London Datastore / UKPN  \n"
            "**Engine:** Apache Spark 4.3.0 + Parquet  \n"
            "**Cache:** Precomputed Vector Store"
        )

    # Load preprocessed analytical data
    data = load_dashboard_data(source_mode=source_mode)

    # Top Capsule Navigation Bar
    custom_meta = data.get("custom_meta") if data else None
    if source_mode == "custom":
        nav_label = f"Custom: {custom_meta['filename']}" if custom_meta else "Awaiting Upload"
        source_tag = "Custom Upload"
    elif source_mode == "sample":
        nav_label = "Sample 36K"
        source_tag = "London SmartMeter"
    else:
        nav_label = "Full 5M Telemetry"
        source_tag = "London SmartMeter"

    render_top_navbar(navigation_page, nav_label, source_tag=source_tag)

    # Show upload panel whenever "Upload Custom Dataset" is active
    if source_mode == "custom":
        render_custom_upload_panel(has_existing_custom=(data is not None))
        if data is None:
            st.info("👆 Upload a `.csv` or `.parquet` file above and click **Run Full Analytics Pipeline** to generate insights.")
            return

    if data is None:
        st.warning(
            "⚠️ Preprocessed analytical artifacts not found. "
            "Please run: `python -m src.analytics.consumption_analytics` and `python -m src.forecasting.demand_forecaster`"
        )
        return

    overview = data.get("overview", {})
    is_custom = (source_mode == "custom" and custom_meta is not None)
    banner_badge = f"Custom Dataset · {custom_meta['filename']}" if is_custom else "London Datastore Telemetry"
    banner_right = "USER TELEMETRY" if is_custom else "UK POWER NETWORKS"

    # =========================================================================
    # SECTION A: OVERVIEW
    # =========================================================================
    if navigation_page == "📊 Overview":
        render_header_banner(
            title="ELEVATING GRID INTELLIGENCE",
            subtitle=(
                f"Full Big Data analytics, demand forecasting, and anomaly diagnostics for `{custom_meta['filename']}`."
                if is_custom else
                "Seamless Big Data energy analytics, demand forecasting, and anomaly diagnostics across London smart meters."
            ),
            badge=banner_badge,
            right_tag=banner_right,
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_metric_card("Total Consumption", f"{overview.get('total_consumption_kwh', 0):,.1f} kWh", "Grid Load")
        with col2:
            render_metric_card("Average Reading", f"{overview.get('avg_consumption_kwh', 0):.4f} kWh", "Per Interval")
        with col3:
            peak_info = overview.get("peak_reading", {})
            render_metric_card("Peak Reading", f"{peak_info.get('kwh', 0):.2f} kWh", f"ID: {peak_info.get('household_id')}")
        with col4:
            render_metric_card("Households", f"{overview.get('unique_households', 0):,}", "Active Meters")
        with col5:
            render_metric_card("Processed Readings", f"{overview.get('total_readings', 0):,}", "Cleaned Telemetry")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        render_feature_strip([
            ("PySpark Ingestion", False),
            ("Parquet Storage", False),
            ("Diurnal Load Matrix", True),
            ("STLF Forecasting", False),
            ("Anomaly Diagnostics", False),
        ])

        c1, c2 = st.columns([3, 2])
        with c1:
            if "hourly" in data:
                fig_hourly = create_hourly_profile_chart(data["hourly"])
                st.plotly_chart(fig_hourly, use_container_width=True)

        with c2:
            st.subheader("Telemetry Scope & System Metadata")
            if is_custom:
                qm = custom_meta.get("quality_metrics", {})
                cm = custom_meta.get("column_mapping", {})
                st.markdown(
                    f"""
                    - **Uploaded File:** `{custom_meta.get('filename')}` ({custom_meta.get('file_size_mb')} MB)
                    - **Observation Timeline:** `{overview.get('date_start', 'N/A')}` to `{overview.get('date_end', 'N/A')}`
                    - **Data Retention Rate:** **{qm.get('retention_rate_pct', 100)}%** ({qm.get('final_clean_records', 0):,} / {qm.get('initial_records', 0):,} rows)
                    - **Mapped Timestamp:** `{cm.get('timestamp')}`
                    - **Mapped Consumption:** `{cm.get('consumption')}`
                    - **Mapped Household ID:** `{cm.get('household_id')}`
                    - **Processing Engine:** Apache Spark with Snappy Parquet
                    """
                )
                for w in custom_meta.get("warnings", []):
                    st.warning(w)
            else:
                st.markdown(
                    f"""
                    - **Observation Timeline:** `{overview.get('date_start', 'N/A')}` to `{overview.get('date_end', 'N/A')}`
                    - **Meter Reading Interval:** 30 Minutes (Half-Hourly)
                    - **Target Geography:** Greater London, United Kingdom (Low Carbon London Trial)
                    - **Baseline Flat-rate Tariff:** 14.228p / kWh (Std)
                    - **Dynamic Tariff (dToU):** Low (3.99p), Normal (11.76p), High (67.20p)
                    - **Out-of-Core Processing Engine:** Apache Spark with Snappy Parquet
                    """
                )
                st.markdown("---")
                st.info(
                    "**Regional Context:** This data represents historical London household electricity readings collected by UK Power Networks (Nov 2011 – Feb 2014). It is strictly presented as London smart meter telemetry."
                )

    # =========================================================================
    # SECTION B: CONSUMPTION ANALYTICS
    # =========================================================================
    elif navigation_page == "📈 Consumption Analytics":
        render_header_banner(
            title="CONSUMPTION PATTERNS & LOAD MATRIX",
            subtitle="Diurnal, seasonal, weekday/weekend, and peak demand aggregations powered by Spark SQL.",
            badge=banner_badge,
            right_tag=banner_right,
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "🕒 Diurnal (Hourly)", "📅 Historical Daily & Monthly", "⚖️ Weekday vs Weekend", "⚡ Peak Demand Events"
        ])

        with tab1:
            st.plotly_chart(create_hourly_profile_chart(data["hourly"]), use_container_width=True)
            st.info(
                "**Diurnal Load Shape Findings:** Displays mean hourly electricity demand alongside ±1 standard deviation bands across the 24-hour cycle."
            )

        with tab2:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(create_daily_trend_chart(data["daily"]), use_container_width=True)
            with c2:
                st.plotly_chart(create_monthly_chart(data["monthly"]), use_container_width=True)

        with tab3:
            st.plotly_chart(create_weekday_weekend_chart(data["weekday_weekend"]), use_container_width=True)
            st.info(
                "**Comparative Observations:** Compares working-day load ramps against weekend consumption curves."
            )

        with tab4:
            st.subheader("Top Peak Demand Intervals Across Grid")
            st.dataframe(data["peaks"], use_container_width=True)

    # =========================================================================
    # SECTION C: DEMAND FORECASTING
    # =========================================================================
    elif navigation_page == "🔮 Demand Forecasting":
        render_header_banner(
            title="SHORT-TERM LOAD FORECASTING",
            subtitle="Linear Regression baseline vs Random Forest comparison with chronological validation.",
            badge=banner_badge,
            right_tag=banner_right,
        )

        fc_eval = data.get("forecast_eval")
        if not fc_eval or data.get("forecast_comp") is None:
            st.warning(
                "Forecasting metrics are unavailable for this dataset (at least 12 chronological hourly intervals are required to train lag models)."
            )
        else:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            lr_metrics = fc_eval.get("linear_regression", {})
            rf_metrics = fc_eval.get("random_forest", {})

            with c1:
                render_metric_card("LR Baseline MAE", f"{lr_metrics.get('mae', 0):.3f} kWh")
            with c2:
                render_metric_card("LR Baseline RMSE", f"{lr_metrics.get('rmse', 0):.3f} kWh")
            with c3:
                render_metric_card("LR Baseline R²", f"{lr_metrics.get('r2_score', 0):.4f}")
            with c4:
                render_metric_card("RF Model MAE", f"{rf_metrics.get('mae', 0):.3f} kWh", delta=f"{lr_metrics.get('mae', 0) - rf_metrics.get('mae', 0):.3f} vs LR")
            with c5:
                render_metric_card("RF Model RMSE", f"{rf_metrics.get('rmse', 0):.3f} kWh")
            with c6:
                render_metric_card("RF Model R²", f"{rf_metrics.get('r2_score', 0):.4f}")

            st.markdown("---")

            model_choice = st.radio(
                "Select Forecasting Model to Visualize:",
                ["Random Forest Regressor (Nonlinear)", "Linear Regression (Baseline)"],
                horizontal=True
            )
            model_col = "pred_random_forest" if "Random Forest" in model_choice else "pred_linear_regression"

            st.plotly_chart(
                create_actual_vs_predicted_chart(data["forecast_comp"], model_col, model_choice),
                use_container_width=True
            )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            col_a, col_b = st.columns([3, 2])
            with col_a:
                if data["forecast_future"] is not None:
                    st.plotly_chart(create_future_forecast_chart(data["forecast_future"]), use_container_width=True)
            with col_b:
                if data["feature_importance"] is not None:
                    st.plotly_chart(create_feature_importance_chart(data["feature_importance"]), use_container_width=True)

    # =========================================================================
    # SECTION D: ANOMALY DETECTION
    # =========================================================================
    elif navigation_page == "🚨 Anomaly Detection":
        render_header_banner(
            title="SMART METER ANOMALY DETECTION",
            subtitle="Dual-engine detection utilizing Statistical Z-Scores and Unsupervised Isolation Forest.",
            badge=banner_badge,
            right_tag=banner_right,
        )

        anom_sum = data.get("anomaly_summary", {})
        if anom_sum:
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                render_metric_card("Total Anomalies", f"{anom_sum.get('total_anomalies', 0):,}", f"{anom_sum.get('anomaly_rate_pct', 0)}% of total")
            with c2:
                render_metric_card("High Surges", f"{anom_sum.get('high_consumption_anomalies', 0):,}", "Abnormal Spikes")
            with c3:
                render_metric_card("Low Drops", f"{anom_sum.get('low_consumption_anomalies', 0):,}", "Unusual Baselines")
            with c4:
                render_metric_card("Z-Score Flags", f"{anom_sum.get('statistical_zscore_count', 0):,}", "Statistical")
            with c5:
                render_metric_card("Isolation Forest", f"{anom_sum.get('isolation_forest_count', 0):,}", "Machine Learning")

        st.markdown("---")

        if data["anomalies"] is not None and not data["anomalies"].empty:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(create_anomaly_scatter_chart(data["anomalies"].head(1000)), use_container_width=True)
            with c2:
                if data["anomaly_dist"] is not None:
                    st.plotly_chart(create_anomaly_hourly_dist_chart(data["anomaly_dist"]), use_container_width=True)

            st.subheader("Filterable Anomaly Event Log")
            selected_cat = st.selectbox("Filter by Category:", ["All", "HIGH_CONSUMPTION", "LOW_CONSUMPTION"])
            filtered_anom = data["anomalies"] if selected_cat == "All" else data["anomalies"][data["anomalies"]["anomaly_category"] == selected_cat]
            st.dataframe(filtered_anom.head(200), use_container_width=True)
        else:
            st.info("No anomalous records were flagged in this dataset under the configured thresholds.")

        if anom_sum:
            st.info(
                f"ℹ️ **Operational Disclaimer:** {anom_sum.get('interpretation_disclaimer', '')}"
            )

    # =========================================================================
    # SECTION E: HOUSEHOLD ANALYSIS
    # =========================================================================
    elif navigation_page == "🏠 Household Analysis":
        render_header_banner(
            title="HOUSEHOLD CONSUMPTION PROFILING",
            subtitle="Individual consumer trends, tariff response, and benchmark comparisons.",
            badge=banner_badge,
            right_tag=banner_right,
        )

        hh_summary = data.get("hh_summary")
        if hh_summary is not None and not hh_summary.empty:
            households = list(hh_summary["household_id"].unique())
            selected_hh = st.selectbox("Select Household ID:", households, index=0)

            hh_row = hh_summary[hh_summary["household_id"] == selected_hh].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Total Consumption", f"{hh_row['total_kwh']:,.1f} kWh")
            with c2:
                render_metric_card("Average Reading", f"{hh_row['avg_kwh']:.4f} kWh", "Per Interval")
            with c3:
                render_metric_card("Peak Reading", f"{hh_row['max_kwh']:.2f} kWh")
            with c4:
                render_metric_card("Tariff Classification", str(hh_row['tariff_type']))

            st.markdown("---")

            if data["hh_hourly"] is not None:
                hh_hourly = data["hh_hourly"][data["hh_hourly"]["household_id"] == selected_hh]
                if not hh_hourly.empty:
                    fig_hh = create_household_profile_chart(hh_hourly, selected_hh)
                    st.plotly_chart(fig_hh, use_container_width=True)

            st.subheader("Top Consumers Comparison Table")
            st.dataframe(hh_summary.head(50), use_container_width=True)

    # =========================================================================
    # SECTION F: DATA EXPLORER & EXPORT
    # =========================================================================
    elif navigation_page == "🔍 Data Explorer & Export":
        render_header_banner(
            title="DATA EXPLORER & ARTIFACT EXPORT",
            subtitle="Preview preprocessed analytical datasets and export for academic reporting.",
            badge=banner_badge,
            right_tag=banner_right,
        )

        dataset_options = {
            "Hourly Consumption Profile": data.get("hourly"),
            "Daily Consumption Trend": data.get("daily"),
            "Monthly Summary": data.get("monthly"),
            "Weekday vs Weekend": data.get("weekday_weekend"),
            "Peak Demand Events": data.get("peaks"),
            "Household Summary": data.get("hh_summary"),
            "Forecasting Actual vs Predicted": data.get("forecast_comp"),
            "24-Hour Future Forecast": data.get("forecast_future"),
            "Detected Anomalies": data.get("anomalies")
        }

        selected_table_name = st.selectbox("Select Analytical Dataset:", list(dataset_options.keys()))
        selected_df = dataset_options[selected_table_name]

        if selected_df is not None:
            st.write(f"**Shape:** {selected_df.shape[0]:,} rows × {selected_df.shape[1]} columns")
            st.dataframe(selected_df.head(500), use_container_width=True)

            csv_bytes = selected_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {selected_table_name} (CSV)",
                data=csv_bytes,
                file_name=f"{selected_table_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"`{selected_table_name}` is not available for this dataset.")


if __name__ == "__main__":
    main()
