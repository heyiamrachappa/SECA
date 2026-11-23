"""
Unit tests for Phase 8: Interactive Dashboard and Visualization Components.
Tests data loading from cache, chart generation, and data formatting.
"""

import pytest
import plotly.graph_objects as go
from dashboard.app import load_dashboard_data
from dashboard.components.charts import (
    create_hourly_profile_chart,
    create_daily_trend_chart,
    create_weekday_weekend_chart,
    create_future_forecast_chart
)


def test_load_dashboard_data():
    """Verify that cached data loader successfully retrieves analytical tables."""
    data = load_dashboard_data(sample_mode=True)
    assert data is not None
    assert "overview" in data
    assert "hourly" in data
    assert "daily" in data
    assert "forecast_comp" in data
    assert "anomalies" in data

    assert data["overview"]["total_readings"] > 0
    assert len(data["hourly"]) == 24


def test_chart_builders():
    """Verify that Plotly chart builders return valid Figure objects."""
    data = load_dashboard_data(sample_mode=True)

    fig_hourly = create_hourly_profile_chart(data["hourly"])
    assert isinstance(fig_hourly, go.Figure)

    fig_daily = create_daily_trend_chart(data["daily"])
    assert isinstance(fig_daily, go.Figure)

    fig_wday = create_weekday_weekend_chart(data["weekday_weekend"])
    assert isinstance(fig_wday, go.Figure)

    if data["forecast_future"] is not None:
        fig_future = create_future_forecast_chart(data["forecast_future"])
        assert isinstance(fig_future, go.Figure)
