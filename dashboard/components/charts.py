"""
Plotly Chart Builders for the Smart Electricity Analytics Dashboard.
Provides high-performance, interactive, responsive visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

CHART_THEME = "plotly_dark"
COLOR_PRIMARY = "#00d2ff"
COLOR_SECONDARY = "#ffaa00"
COLOR_ALERT = "#ff4b4b"
COLOR_SUCCESS = "#00e676"
BG_COLOR = "#0e1726"


def create_hourly_profile_chart(df: pd.DataFrame) -> go.Figure:
    """Renders 24-hour diurnal profile with mean line and confidence/spread bands."""
    fig = go.Figure()

    # Upper/Lower bound fill (std deviation band)
    if "std_kwh" in df.columns:
        upper = df["avg_kwh"] + df["std_kwh"]
        lower = np.maximum(0, df["avg_kwh"] - df["std_kwh"])

        fig.add_trace(go.Scatter(
            x=df["hour"], y=upper,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df["hour"], y=lower,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 210, 255, 0.15)',
            name='± 1 Std Dev', hoverinfo='skip'
        ))

    # Mean consumption line
    fig.add_trace(go.Scatter(
        x=df["hour"], y=df["avg_kwh"],
        mode='lines+markers',
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=6, color="#ffffff"),
        name='Avg Consumption (kWh)',
        hovertemplate="Hour %{x}:00<br>Average: <b>%{y:.4f} kWh</b><extra></extra>"
    ))

    fig.update_layout(
        template=CHART_THEME,
        title="<b>24-Hour Diurnal Electricity Consumption Profile</b>",
        xaxis=dict(title="Hour of Day (0 - 23)", dtick=1, gridcolor="#22304a"),
        yaxis=dict(title="Electricity Consumption (kWh / half-hour)", gridcolor="#22304a"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_daily_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Renders daily aggregated consumption over time with range slider."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df["date"]),
        y=df["total_kwh"],
        mode='lines',
        line=dict(color=COLOR_PRIMARY, width=2),
        name='Daily Total (kWh)',
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Total: <b>%{y:,.1f} kWh</b><extra></extra>"
    ))

    # 7-day moving average
    ma_7 = df["total_kwh"].rolling(7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df["date"]),
        y=ma_7,
        mode='lines',
        line=dict(color=COLOR_SECONDARY, width=2.5, dash='dash'),
        name='7-Day Moving Avg',
        hovertemplate="Date: %{x|%Y-%m-%d}<br>7-Day MA: <b>%{y:,.1f} kWh</b><extra></extra>"
    ))

    fig.update_layout(
        template=CHART_THEME,
        title="<b>Historical Daily Electricity Demand Trend</b>",
        xaxis=dict(title="Timeline", rangeslider=dict(visible=True), gridcolor="#22304a"),
        yaxis=dict(title="Total Grid Load (kWh)", gridcolor="#22304a"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_weekday_weekend_chart(df: pd.DataFrame) -> go.Figure:
    """Compares Weekday vs Weekend hourly load curves."""
    fig = px.line(
        df,
        x="hour",
        y="avg_kwh",
        color="day_type",
        color_discrete_map={"Weekday": COLOR_PRIMARY, "Weekend": COLOR_SECONDARY},
        markers=True,
        template=CHART_THEME,
        title="<b>Weekday vs Weekend Load Curve Comparison</b>"
    )
    fig.update_layout(
        xaxis=dict(title="Hour of Day (0 - 23)", dtick=1, gridcolor="#22304a"),
        yaxis=dict(title="Average Consumption (kWh / half-hour)", gridcolor="#22304a"),
        legend=dict(title="Day Category", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_monthly_chart(df: pd.DataFrame) -> go.Figure:
    """Renders monthly consumption bar chart."""
    df_copy = df.copy()
    df_copy["period"] = df_copy["year"].astype(str) + "-" + df_copy["month"].astype(str).str.zfill(2)

    fig = px.bar(
        df_copy,
        x="period",
        y="total_kwh",
        color="total_kwh",
        color_continuous_scale="Viridis",
        template=CHART_THEME,
        title="<b>Monthly Total Electricity Consumption (Seasonal Trends)</b>"
    )
    fig.update_layout(
        xaxis=dict(title="Month (YYYY-MM)", gridcolor="#22304a"),
        yaxis=dict(title="Total Electricity (kWh)", gridcolor="#22304a")
    )
    return fig


def create_actual_vs_predicted_chart(df: pd.DataFrame, model_col: str, model_name: str) -> go.Figure:
    """Renders actual vs predicted time-series with error display."""
    fig = go.Figure()

    timestamps = pd.to_datetime(df["timestamp"])

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df["actual_kwh"],
        mode='lines',
        line=dict(color="#8892b0", width=1.8),
        name='Actual Load (kWh)',
        hovertemplate="%{x}<br>Actual: <b>%{y:.2f} kWh</b><extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df[model_col],
        mode='lines',
        line=dict(color=COLOR_PRIMARY if "random" in model_col else COLOR_SECONDARY, width=2),
        name=f'Predicted ({model_name})',
        hovertemplate="%{x}<br>Pred: <b>%{y:.2f} kWh</b><extra></extra>"
    ))

    fig.update_layout(
        template=CHART_THEME,
        title=f"<b>Actual vs Predicted Demand: {model_name} (Out-of-Time Test Set)</b>",
        xaxis=dict(title="Test Set Chronological Timeline", gridcolor="#22304a"),
        yaxis=dict(title="Hourly Grid Load (kWh)", gridcolor="#22304a"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_future_forecast_chart(df: pd.DataFrame) -> go.Figure:
    """Renders future 24-hour demand forecast with confidence intervals."""
    fig = go.Figure()
    timestamps = pd.to_datetime(df["timestamp"])

    # 80% Confidence Interval Band
    fig.add_trace(go.Scatter(
        x=timestamps, y=df["upper_bound_80"],
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=df["lower_bound_80"],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0, 230, 118, 0.15)',
        name='80% Prediction Interval', hoverinfo='skip'
    ))

    # Random Forest Forecast
    fig.add_trace(go.Scatter(
        x=timestamps, y=df["forecast_random_forest"],
        mode='lines+markers',
        line=dict(color=COLOR_SUCCESS, width=3),
        marker=dict(size=6),
        name='Random Forest Forecast',
        hovertemplate="%{x}<br>Forecast: <b>%{y:.2f} kWh</b><extra></extra>"
    ))

    # Linear Regression Forecast
    fig.add_trace(go.Scatter(
        x=timestamps, y=df["forecast_linear_regression"],
        mode='lines',
        line=dict(color=COLOR_SECONDARY, width=2, dash='dot'),
        name='Linear Regression Baseline',
        hovertemplate="%{x}<br>LR Baseline: <b>%{y:.2f} kWh</b><extra></extra>"
    ))

    fig.update_layout(
        template=CHART_THEME,
        title="<b>24-Hour Ahead Out-of-Sample Electricity Demand Forecast</b>",
        xaxis=dict(title="Future Forecast Horizon (Next 24 Hours)", gridcolor="#22304a"),
        yaxis=dict(title="Projected Hourly Electricity Load (kWh)", gridcolor="#22304a"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_feature_importance_chart(df: pd.DataFrame) -> go.Figure:
    """Renders horizontal bar chart of top ML feature importances."""
    top_10 = df.head(10).sort_values("importance", ascending=True)
    fig = px.bar(
        top_10,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Tealgrn",
        template=CHART_THEME,
        title="<b>Top 10 Feature Importances (Random Forest Model)</b>"
    )
    fig.update_layout(
        xaxis=dict(title="Relative Importance Score", gridcolor="#22304a"),
        yaxis=dict(title="Engineered Feature")
    )
    return fig


def create_anomaly_scatter_chart(df: pd.DataFrame) -> go.Figure:
    """Visualizes detected anomalies against normal readings."""
    fig = px.scatter(
        df,
        x=pd.to_datetime(df["timestamp"]),
        y="consumption_kwh",
        color="anomaly_category",
        color_discrete_map={
            "NORMAL": "#4a5568",
            "HIGH_CONSUMPTION": COLOR_ALERT,
            "LOW_CONSUMPTION": COLOR_SECONDARY
        },
        size="anomaly_score",
        size_max=12,
        opacity=0.85,
        template=CHART_THEME,
        title="<b>Detected Consumption Anomalies vs Historical Baseline</b>"
    )
    fig.update_layout(
        xaxis=dict(title="Timestamp", gridcolor="#22304a"),
        yaxis=dict(title="Reading Consumption (kWh)", gridcolor="#22304a"),
        legend=dict(title="Classification", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_anomaly_hourly_dist_chart(df: pd.DataFrame) -> go.Figure:
    """Renders hourly distribution of anomalies."""
    fig = px.bar(
        df,
        x="hour",
        y=["HIGH_CONSUMPTION", "LOW_CONSUMPTION"] if "HIGH_CONSUMPTION" in df.columns else df.columns[1:],
        template=CHART_THEME,
        title="<b>Diurnal Anomaly Distribution (Count by Hour of Day)</b>",
        color_discrete_map={"HIGH_CONSUMPTION": COLOR_ALERT, "LOW_CONSUMPTION": COLOR_SECONDARY}
    )
    fig.update_layout(
        xaxis=dict(title="Hour of Day (0 - 23)", dtick=1, gridcolor="#22304a"),
        yaxis=dict(title="Anomaly Event Count", gridcolor="#22304a"),
        barmode="stack",
        legend=dict(title="Anomaly Type", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
