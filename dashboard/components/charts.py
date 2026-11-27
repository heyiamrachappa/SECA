"""
Plotly Chart Builders for the Smart Electricity Analytics Dashboard.
Styled in Midnight Obsidian & Electric Indigo-Violet with smooth spline curves,
translucent violet area gradients, and clean minimal grids.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

CHART_THEME = "plotly_dark"
FONT_FAMILY = "Plus Jakarta Sans, Inter, -apple-system, BlinkMacSystemFont, sans-serif"

# Midnight Obsidian & Electric Indigo-Violet Palette
COLOR_PRIMARY = "#8b5cf6"        # Electric Violet
COLOR_PRIMARY_LIGHT = "#a78bfa"  # Soft Lavender Glow
COLOR_INDIGO = "#6366f1"         # Royal Indigo
COLOR_SECONDARY = "#38bdf8"      # Cool Periwinkle / Sky
COLOR_ALERT = "#fb7185"          # Luminous Rose (High Anomaly)
COLOR_WARNING = "#fbbf24"        # Warm Amber (Low Anomaly / Baseline)
COLOR_SUCCESS = "#34d399"        # Emerald Mint (Forecast)
BG_COLOR = "rgba(11, 12, 24, 0)"
GRID_COLOR = "rgba(139, 92, 246, 0.11)"
TEXT_MUTED = "#94a3b8"


def _apply_saas_layout(fig: go.Figure, title: str, x_title: str = "", y_title: str = "", **extra) -> go.Figure:
    """Applies unified Midnight Obsidian & Electric Violet styling to any Plotly figure."""
    layout_args = dict(
        template=CHART_THEME,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=FONT_FAMILY, size=16, color="#f8fafc"),
            x=0.01,
            xanchor="left",
        ),
        font=dict(family=FONT_FAMILY, size=12, color="#cbd5e1"),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=45, r=25, t=58, b=45),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#121329",
            bordercolor="rgba(139, 92, 246, 0.55)",
            font=dict(family=FONT_FAMILY, size=12, color="#f8fafc")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#cbd5e1")
        ),
    )
    layout_args.update(extra)
    fig.update_layout(**layout_args)
    fig.update_xaxes(
        title_text=x_title,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color="#cbd5e1", size=12),
    )
    fig.update_yaxes(
        title_text=y_title,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color="#cbd5e1", size=12),
    )
    return fig


def create_hourly_profile_chart(df: pd.DataFrame) -> go.Figure:
    """Renders 24-hour diurnal profile with glowing violet spline and area gradient."""
    fig = go.Figure()

    if "std_kwh" in df.columns:
        upper = df["avg_kwh"] + df["std_kwh"]
        lower = np.maximum(0, df["avg_kwh"] - df["std_kwh"])

        fig.add_trace(go.Scatter(
            x=df["hour"], y=upper,
            mode="lines",
            line=dict(width=0, shape="spline"),
            showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=df["hour"], y=lower,
            mode="lines",
            line=dict(width=0, shape="spline"),
            fill="tonexty",
            fillcolor="rgba(99, 102, 241, 0.14)",
            name="± 1 Std Dev", hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter(
        x=df["hour"], y=df["avg_kwh"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(139, 92, 246, 0.18)",
        line=dict(color=COLOR_PRIMARY_LIGHT, width=3, shape="spline"),
        marker=dict(size=7, color="#ffffff", line=dict(color=COLOR_PRIMARY, width=2)),
        name="Avg Consumption (kWh)",
        hovertemplate="Hour %{x}:00<br>Average: <b>%{y:.4f} kWh</b><extra></extra>"
    ))

    _apply_saas_layout(
        fig,
        title="24-Hour Diurnal Electricity Consumption Profile",
        x_title="Hour of Day (0 - 23)",
        y_title="Consumption (kWh / half-hour)"
    )
    fig.update_xaxes(dtick=1)
    return fig


def create_daily_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Renders daily aggregated consumption over time with violet glow and range slider."""
    fig = go.Figure()

    dates = pd.to_datetime(df["date"])
    fig.add_trace(go.Scatter(
        x=dates,
        y=df["total_kwh"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.15)",
        line=dict(color=COLOR_PRIMARY, width=2, shape="spline"),
        name="Daily Total (kWh)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Total: <b>%{y:,.1f} kWh</b><extra></extra>"
    ))

    ma_7 = df["total_kwh"].rolling(7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=dates,
        y=ma_7,
        mode="lines",
        line=dict(color=COLOR_SECONDARY, width=2.4, dash="dot", shape="spline"),
        name="7-Day Moving Avg",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>7-Day MA: <b>%{y:,.1f} kWh</b><extra></extra>"
    ))

    _apply_saas_layout(
        fig,
        title="Historical Daily Electricity Demand Trend",
        x_title="Timeline",
        y_title="Total Grid Load (kWh)"
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor="rgba(18, 19, 41, 0.7)", bordercolor="rgba(139, 92, 246, 0.25)")
    )
    return fig


def create_weekday_weekend_chart(df: pd.DataFrame) -> go.Figure:
    """Compares Weekday vs Weekend hourly load curves."""
    fig = px.line(
        df,
        x="hour",
        y="avg_kwh",
        color="day_type",
        color_discrete_map={"Weekday": COLOR_PRIMARY_LIGHT, "Weekend": COLOR_SECONDARY},
        markers=True,
        template=CHART_THEME,
    )
    fig.update_traces(line=dict(width=3, shape="spline"), marker=dict(size=6))
    _apply_saas_layout(
        fig,
        title="Weekday vs Weekend Load Curve Comparison",
        x_title="Hour of Day (0 - 23)",
        y_title="Average Consumption (kWh / half-hour)"
    )
    fig.update_xaxes(dtick=1)
    return fig


def create_monthly_chart(df: pd.DataFrame) -> go.Figure:
    """Renders monthly consumption bar chart with indigo-violet gradient scale."""
    df_copy = df.copy()
    df_copy["period"] = df_copy["year"].astype(str) + "-" + df_copy["month"].astype(str).str.zfill(2)

    fig = px.bar(
        df_copy,
        x="period",
        y="total_kwh",
        color="total_kwh",
        color_continuous_scale=["#1e1b6b", "#4f46e5", "#8b5cf6", "#c4b5fd"],
        template=CHART_THEME,
    )
    fig.update_traces(marker_line_width=0)
    _apply_saas_layout(
        fig,
        title="Monthly Total Electricity Consumption",
        x_title="Month (YYYY-MM)",
        y_title="Total Electricity (kWh)",
        coloraxis_showscale=False
    )
    return fig


def create_actual_vs_predicted_chart(df: pd.DataFrame, model_col: str, model_name: str) -> go.Figure:
    """Renders actual vs predicted time-series with clean glowing contrast."""
    fig = go.Figure()
    timestamps = pd.to_datetime(df["timestamp"])

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df["actual_kwh"],
        mode="lines",
        line=dict(color="#64748b", width=1.8),
        name="Actual Load (kWh)",
        hovertemplate="%{x}<br>Actual: <b>%{y:.2f} kWh</b><extra></extra>"
    ))

    pred_color = COLOR_PRIMARY_LIGHT if "random" in model_col else COLOR_SECONDARY
    fill_rgba = "rgba(139, 92, 246, 0.14)" if "random" in model_col else "rgba(56, 189, 248, 0.12)"

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df[model_col],
        mode="lines",
        fill="tozeroy",
        fillcolor=fill_rgba,
        line=dict(color=pred_color, width=2.2),
        name=f"Predicted ({model_name})",
        hovertemplate="%{x}<br>Pred: <b>%{y:.2f} kWh</b><extra></extra>"
    ))

    _apply_saas_layout(
        fig,
        title=f"Actual vs Predicted Demand: {model_name}",
        x_title="Test Set Chronological Timeline",
        y_title="Hourly Grid Load (kWh)"
    )
    return fig


def create_future_forecast_chart(df: pd.DataFrame) -> go.Figure:
    """Renders future 24-hour demand forecast with confidence intervals."""
    fig = go.Figure()
    timestamps = pd.to_datetime(df["timestamp"])

    fig.add_trace(go.Scatter(
        x=timestamps, y=df["upper_bound_80"],
        mode="lines", line=dict(width=0, shape="spline"),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=df["lower_bound_80"],
        mode="lines", line=dict(width=0, shape="spline"),
        fill="tonexty", fillcolor="rgba(139, 92, 246, 0.18)",
        name="80% Prediction Interval", hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=df["forecast_random_forest"],
        mode="lines+markers",
        line=dict(color=COLOR_PRIMARY_LIGHT, width=3, shape="spline"),
        marker=dict(size=6, color="#ffffff", line=dict(color=COLOR_PRIMARY, width=2)),
        name="Random Forest Forecast",
        hovertemplate="%{x}<br>Forecast: <b>%{y:.2f} kWh</b><extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=df["forecast_linear_regression"],
        mode="lines",
        line=dict(color=COLOR_SECONDARY, width=2, dash="dot", shape="spline"),
        name="Linear Regression Baseline",
        hovertemplate="%{x}<br>LR Baseline: <b>%{y:.2f} kWh</b><extra></extra>"
    ))

    _apply_saas_layout(
        fig,
        title="24-Hour Ahead Out-of-Sample Demand Forecast",
        x_title="Future Forecast Horizon (Next 24 Hours)",
        y_title="Projected Hourly Load (kWh)"
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
        color_continuous_scale=["#312e81", "#6366f1", "#a78bfa"],
        template=CHART_THEME,
    )
    fig.update_traces(marker_line_width=0)
    _apply_saas_layout(
        fig,
        title="Top 10 Feature Importances (Random Forest)",
        x_title="Relative Importance Score",
        y_title="Engineered Feature",
        coloraxis_showscale=False,
        hovermode="y unified"
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
            "NORMAL": "#475569",
            "HIGH_CONSUMPTION": COLOR_ALERT,
            "LOW_CONSUMPTION": COLOR_WARNING
        },
        size="anomaly_score",
        size_max=12,
        opacity=0.88,
        template=CHART_THEME,
    )
    _apply_saas_layout(
        fig,
        title="Detected Consumption Anomalies vs Historical Baseline",
        x_title="Timestamp",
        y_title="Reading Consumption (kWh)",
        hovermode="closest"
    )
    return fig


def create_anomaly_hourly_dist_chart(df: pd.DataFrame) -> go.Figure:
    """Renders hourly distribution of anomalies."""
    fig = px.bar(
        df,
        x="hour",
        y=["HIGH_CONSUMPTION", "LOW_CONSUMPTION"] if "HIGH_CONSUMPTION" in df.columns else df.columns[1:],
        template=CHART_THEME,
        color_discrete_map={"HIGH_CONSUMPTION": COLOR_ALERT, "LOW_CONSUMPTION": COLOR_WARNING}
    )
    fig.update_traces(marker_line_width=0)
    _apply_saas_layout(
        fig,
        title="Diurnal Anomaly Distribution (By Hour of Day)",
        x_title="Hour of Day (0 - 23)",
        y_title="Anomaly Event Count",
        barmode="stack"
    )
    fig.update_xaxes(dtick=1)
    return fig


def create_household_profile_chart(df: pd.DataFrame, household_id: str) -> go.Figure:
    """Renders an individual household's 24-hour diurnal load curve in the Electric Violet theme."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["hour"],
        y=df["avg_kwh"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(139, 92, 246, 0.22)",
        line=dict(color=COLOR_PRIMARY_LIGHT, width=3, shape="spline"),
        marker=dict(size=7, color="#ffffff", line=dict(color=COLOR_PRIMARY, width=2)),
        name=f"Household {household_id}",
        hovertemplate="Hour %{x}:00<br>Avg: <b>%{y:.4f} kWh</b><extra></extra>"
    ))
    _apply_saas_layout(
        fig,
        title=f"24-Hour Diurnal Load Profile: Household {household_id}",
        x_title="Hour of Day (0 - 23)",
        y_title="Average Consumption (kWh / half-hour)"
    )
    fig.update_xaxes(dtick=1)
    return fig
