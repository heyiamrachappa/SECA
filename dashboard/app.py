"""
Smart Electricity Consumption Analytics and Demand Forecasting Dashboard.
Built with Streamlit and Plotly.
Provides an interactive, high-performance interface for Big Data Smart Grid telemetry.
"""

import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.metrics import render_metric_card, render_header_banner
from dashboard.components.charts import (
    create_hourly_profile_chart,
    create_daily_trend_chart,
    create_weekday_weekend_chart,
    create_monthly_chart,
    create_actual_vs_predicted_chart,
    create_future_forecast_chart,
    create_feature_importance_chart,
    create_anomaly_scatter_chart,
    create_anomaly_hourly_dist_chart
)
from src.utils.config import load_config, get_project_root

# Page configuration
st.set_page_config(
    page_title="SECA - Smart Electricity Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern energy analytics styling
st.markdown(
    """
    <style>
        .main { background-color: #0b111e; }
        .stMetric {
            background: #141f32;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #1f2e48;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .stMetric label { color: #8fa0be !important; font-size: 13px !important; }
        .stMetric div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 24px !important; font-weight: 700; }
        div[data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_dashboard_data(sample_mode: bool = False):
    """
    Cached data loader reading precomputed analytical summaries.
    Guarantees sub-second dashboard updates without rerunning Spark operations.
    """
    root = get_project_root()
    base_dir = root / ("dataset/processed/sample_analytics_summary" if sample_mode else "dataset/processed/analytics_summary")

    # Fallback to sample if full summary not generated yet
    if not base_dir.exists():
        base_dir = root / "dataset/processed/sample_analytics_summary"
        if not base_dir.exists():
            return None

    data = {}
    try:
        # 1. Overview
        with open(base_dir / "overview_metrics.json", "r") as f:
            data["overview"] = json.load(f)

        # 2. Hourly Profile
        data["hourly"] = pd.read_csv(base_dir / "hourly_consumption.csv")

        # 3. Daily Trend
        data["daily"] = pd.read_csv(base_dir / "daily_consumption.csv")

        # 4. Monthly Trend
        data["monthly"] = pd.read_csv(base_dir / "monthly_consumption.csv")

        # 5. Weekday vs Weekend
        data["weekday_weekend"] = pd.read_csv(base_dir / "weekday_vs_weekend.csv")

        # 6. Peak Demand Periods
        data["peaks"] = pd.read_csv(base_dir / "peak_demand_periods.csv")

        # 7. Household Summary & Profile
        data["hh_summary"] = pd.read_csv(base_dir / "household_summary.csv")
        data["hh_hourly"] = pd.read_csv(base_dir / "household_hourly_profile.csv")

        # 8. Forecasting
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

        # 9. Anomaly Detection
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

    except Exception as e:
        st.error(f"Error loading analytical data: {e}")
        return None

    return data


def main():
    config = load_config()

    # Sidebar Navigation
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h2 style="color: #00d2ff; margin: 0; font-size: 22px;">⚡ SECA Platform</h2>
            <p style="color: #8fa0be; font-size: 12px; margin: 0;">Big Data Electricity Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    data_source_mode = st.sidebar.radio(
        "📁 Dataset Scope:",
        ["Full Dataset (5M Records)", "Sample Dataset (36K Records)"],
        index=0
    )
    is_sample_mode = (data_source_mode == "Sample Dataset (36K Records)")

    st.sidebar.markdown("---")
    navigation_page = st.sidebar.radio(
        "Navigation Menu:",
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
    st.sidebar.info(
        f"**Dataset:** Low Carbon London  \n"
        f"**Source:** London Datastore / UK Power Networks  \n"
        f"**Engine:** Apache Spark 4.3.0 + Parquet  \n"
        f"**Mode:** In-Memory Vectorized Cache"
    )

    # Load preprocessed analytical data
    data = load_dashboard_data(sample_mode=is_sample_mode)
    if data is None:
        st.warning(
            "⚠️ Preprocessed analytical artifacts not found. "
            "Please run: `python -m src.analytics.consumption_analytics` and `python -m src.forecasting.demand_forecaster`"
        )
        return

    overview = data.get("overview", {})

    # =========================================================================
    # SECTION A: OVERVIEW
    # =========================================================================
    if navigation_page == "📊 Overview":
        render_header_banner(
            title="Smart Electricity Analytics Overview",
            subtitle="Executive summary of distributed smart-meter telemetry and grid demand patterns.",
            badge="London Datastore Telemetry"
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_metric_card("Total Consumption", f"{overview.get('total_consumption_kwh', 0):,.1f} kWh", "Grid Load")
        with col2:
            render_metric_card("Average Reading", f"{overview.get('avg_consumption_kwh', 0):.4f} kWh", "Per Half-Hour")
        with col3:
            peak_info = overview.get("peak_reading", {})
            render_metric_card("Peak Reading", f"{peak_info.get('kwh', 0):.2f} kWh", f"ID: {peak_info.get('household_id')}")
        with col4:
            render_metric_card("Households", f"{overview.get('unique_households', 0):,}", "Active Meters")
        with col5:
            render_metric_card("Processed Readings", f"{overview.get('total_readings', 0):,}", "Cleaned Telemetry")

        st.markdown("---")

        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Diurnal Load Profile Overview")
            if "hourly" in data:
                fig_hourly = create_hourly_profile_chart(data["hourly"])
                st.plotly_chart(fig_hourly, use_container_width=True)

        with c2:
            st.subheader("Telemetry Scope & System Metadata")
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
            st.markdown(
                """
                > **Notice on Regional Context:**  
                > This data represents historical London household electricity readings collected by UK Power Networks. It is strictly presented as London smart meter data and not current Indian or local grid data.
                """
            )

    # =========================================================================
    # SECTION B: CONSUMPTION ANALYTICS
    # =========================================================================
    elif navigation_page == "📈 Consumption Analytics":
        render_header_banner(
            title="Electricity Consumption Patterns & Load Analysis",
            subtitle="Diurnal, seasonal, weekday/weekend, and peak demand aggregations powered by Spark SQL."
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "🕒 Diurnal (Hourly)", "📅 Historical Daily & Monthly", "⚖️ Weekday vs Weekend", "⚡ Peak Demand Events"
        ])

        with tab1:
            st.subheader("24-Hour Diurnal Load Profile")
            st.plotly_chart(create_hourly_profile_chart(data["hourly"]), use_container_width=True)
            st.markdown(
                """
                **Diurnal Load Shape Findings:**  
                Electricity consumption reaches its minimum baseline during early morning hours (03:00 - 05:00), 
                exhibits a morning ramp between 07:00 and 09:00, and attains its diurnal peak in the evening (18:30 - 21:30) 
                coinciding with residential lighting, heating, and kitchen appliances.
                """
            )

        with tab2:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader("Daily Electricity Demand Trend")
                st.plotly_chart(create_daily_trend_chart(data["daily"]), use_container_width=True)
            with c2:
                st.subheader("Monthly Seasonality (kWh)")
                st.plotly_chart(create_monthly_chart(data["monthly"]), use_container_width=True)

        with tab3:
            st.subheader("Weekday vs Weekend Load Comparison")
            st.plotly_chart(create_weekday_weekend_chart(data["weekday_weekend"]), use_container_width=True)
            st.markdown(
                """
                **Comparative Observations:**  
                Weekend profiles exhibit a delayed morning wake-up ramp (flattened between 07:00 and 10:00) 
                and higher daytime sustained occupancy compared to working weekdays.
                """
            )

        with tab4:
            st.subheader("Top Peak Demand Intervals Across Grid")
            st.dataframe(data["peaks"], use_container_width=True)

    # =========================================================================
    # SECTION C: DEMAND FORECASTING
    # =========================================================================
    elif navigation_page == "🔮 Demand Forecasting":
        render_header_banner(
            title="Short-Term Load Forecasting (STLF)",
            subtitle="Linear Regression Baseline vs Random Forest Comparison with chronological evaluation."
        )

        fc_eval = data.get("forecast_eval")
        if fc_eval:
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

        if data["forecast_comp"] is not None:
            # Let user filter test display range
            df_comp = data["forecast_comp"]
            st.plotly_chart(
                create_actual_vs_predicted_chart(df_comp, model_col, model_choice),
                use_container_width=True
            )

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.subheader("Future 24-Hour Electricity Demand Forecast")
            if data["forecast_future"] is not None:
                st.plotly_chart(create_future_forecast_chart(data["forecast_future"]), use_container_width=True)
        with col_b:
            st.subheader("Top Predictive Feature Importances")
            if data["feature_importance"] is not None:
                st.plotly_chart(create_feature_importance_chart(data["feature_importance"]), use_container_width=True)

    # =========================================================================
    # SECTION D: ANOMALY DETECTION
    # =========================================================================
    elif navigation_page == "🚨 Anomaly Detection":
        render_header_banner(
            title="Smart Meter Anomaly Detection Engine",
            subtitle="Dual-engine detection utilizing Statistical Z-Scores and Unsupervised Isolation Forest."
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

        if data["anomalies"] is not None:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader("Anomalous Consumption vs Expected Baseline")
                st.plotly_chart(create_anomaly_scatter_chart(data["anomalies"].head(1000)), use_container_width=True)
            with c2:
                st.subheader("Diurnal Distribution of Anomalies")
                if data["anomaly_dist"] is not None:
                    st.plotly_chart(create_anomaly_hourly_dist_chart(data["anomaly_dist"]), use_container_width=True)

            st.subheader("Filterable Anomaly Event Log")
            selected_cat = st.selectbox("Filter by Category:", ["All", "HIGH_CONSUMPTION", "LOW_CONSUMPTION"])
            filtered_anom = data["anomalies"] if selected_cat == "All" else data["anomalies"][data["anomalies"]["anomaly_category"] == selected_cat]
            st.dataframe(filtered_anom.head(200), use_container_width=True)

        st.info(
            f"ℹ️ **Operational Disclaimer:** {anom_sum.get('interpretation_disclaimer', '')}"
        )

    # =========================================================================
    # SECTION E: HOUSEHOLD ANALYSIS
    # =========================================================================
    elif navigation_page == "🏠 Household Analysis":
        render_header_banner(
            title="Household-Level Consumption Profiling",
            subtitle="Individual consumer trends, tariff response, and benchmark comparisons."
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
                render_metric_card("Average Reading", f"{hh_row['avg_kwh']:.4f} kWh", "Per Half-Hour")
            with c3:
                render_metric_card("Peak Reading", f"{hh_row['max_kwh']:.2f} kWh")
            with c4:
                render_metric_card("Tariff Classification", str(hh_row['tariff_type']))

            st.markdown("---")

            # Individual diurnal curve
            if data["hh_hourly"] is not None:
                hh_hourly = data["hh_hourly"][data["hh_hourly"]["household_id"] == selected_hh]
                if not hh_hourly.empty:
                    fig_hh = px.line(
                        hh_hourly,
                        x="hour", y="avg_kwh",
                        markers=True,
                        template="plotly_dark",
                        title=f"<b>24-Hour Diurnal Load Profile: Household {selected_hh}</b>"
                    )
                    fig_hh.update_layout(xaxis=dict(dtick=1))
                    st.plotly_chart(fig_hh, use_container_width=True)

            st.subheader("Top Consumers Comparison Table")
            st.dataframe(hh_summary.head(50), use_container_width=True)

    # =========================================================================
    # SECTION F: DATA EXPLORER & EXPORT
    # =========================================================================
    elif navigation_page == "🔍 Data Explorer & Export":
        render_header_banner(
            title="Data Explorer & Artifact Export",
            subtitle="Preview preprocessed analytical datasets and export for academic reporting."
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


if __name__ == "__main__":
    main()
