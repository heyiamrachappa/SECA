"""
Unit tests for Phase 6: Machine Learning Demand Forecasting.
Tests chronological splitting, lag and rolling feature engineering, model training, and metrics calculation.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.utils.config import get_project_root
from src.forecasting.demand_forecaster import DemandForecaster


def test_feature_engineering_chronology():
    """Verify that feature engineering produces expected lag columns and rolling features."""
    forecaster = DemandForecaster()
    df_ts = forecaster.prepare_hourly_series(sample_mode=True)
    df_feat = forecaster.engineer_features(df_ts)

    # Check that lag columns exist
    assert "lag_1h" in df_feat.columns
    assert "lag_24h" in df_feat.columns
    assert "rolling_mean_3h" in df_feat.columns
    assert "sin_hour" in df_feat.columns
    assert len(df_feat) > 0


def test_chronological_split_no_leakage():
    """Verify that train timestamps are strictly before test timestamps."""
    forecaster = DemandForecaster()
    df_ts = forecaster.prepare_hourly_series(sample_mode=True)
    df_feat = forecaster.engineer_features(df_ts)
    X_train, X_test, y_train, y_test, ts_train, ts_test = forecaster.train_test_split_chronological(df_feat)

    assert ts_train.max() < ts_test.min(), "Data leakage detected: train timestamps overlap test timestamps!"
    assert len(X_train) > len(X_test)


def test_model_training_and_evaluation():
    """Verify that both models train and produce valid evaluation metrics."""
    forecaster = DemandForecaster()
    df_ts = forecaster.prepare_hourly_series(sample_mode=True)
    df_feat = forecaster.engineer_features(df_ts)
    X_train, X_test, y_train, y_test, ts_train, ts_test = forecaster.train_test_split_chronological(df_feat)

    forecaster.train_models(X_train, y_train)
    metrics = forecaster.evaluate_models(X_test, y_test)

    assert "linear_regression" in metrics
    assert "random_forest" in metrics
    assert metrics["linear_regression"]["r2_score"] is not None
    assert metrics["random_forest"]["mae"] > 0
