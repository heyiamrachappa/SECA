"""
Machine Learning Demand Forecasting Module for Smart Electricity Analytics.
Implements Linear Regression (baseline) and Random Forest Regressor models,
with chronological validation, lag features, rolling statistics, and multi-step forecasting.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("DemandForecaster")


class DemandForecaster:
    """
    Time-series Machine Learning engine for Short-Term Load Forecasting (STLF).
    Engineers causal lag features and rolling stats with strict chronological splits
    to prevent target and future data leakage.
    """

    def __init__(self):
        self.config = load_config()
        self.fc_cfg = self.config.get("forecasting", {})
        self.test_ratio = float(self.fc_cfg.get("test_split_ratio", 0.20))
        self.horizon_hours = int(self.fc_cfg.get("forecast_horizon_hours", 24))
        self.lag_hours = [int(l) for l in self.fc_cfg.get("lag_hours", [1, 2, 3, 24, 48, 168])]
        self.rolling_windows = [int(w) for w in self.fc_cfg.get("rolling_windows_hours", [3, 6, 24])]
        self.random_seed = int(self.fc_cfg.get("random_seed", 42))

        self.lr_model: Optional[LinearRegression] = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = []

    def prepare_hourly_series(
        self,
        df_hourly_path: Optional[Path] = None,
        sample_mode: bool = False,
        custom_parquet: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Loads cleaned telemetry from Parquet and builds a continuous hourly load series.
        """
        root = get_project_root()
        if custom_parquet is not None:
            parquet_file = Path(custom_parquet) if os.path.isabs(str(custom_parquet)) else root / custom_parquet
        else:
            proc_dir = root / self.config["paths"]["processed_data_dir"]
            parquet_file = proc_dir / ("sample_cleaned_energy_data.parquet" if sample_mode else "cleaned_energy_data.parquet")

        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet data not found at {parquet_file}. Run preprocessing first.")

        logger.info(f"Loading parquet dataset for time-series feature engineering: {parquet_file}")
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_file), columns=["timestamp", "consumption_kwh"])
        pdf = table.to_pandas()

        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.sort_values("timestamp")

        # Aggregate total and average load by 1-hour interval
        hourly_series = (
            pdf.set_index("timestamp")
            .resample("1h")
            .agg({"consumption_kwh": ["sum", "mean", "count"]})
        )
        hourly_series.columns = ["total_kwh", "avg_kwh", "readings_count"]
        hourly_series = hourly_series.ffill().fillna(0.0)
        hourly_series = hourly_series.reset_index()

        logger.info(f"Constructed hourly time-series: {len(hourly_series):,} chronological hourly steps.")
        return hourly_series

    def engineer_features(self, df_ts: pd.DataFrame) -> pd.DataFrame:
        """
        Builds chronological lag features and rolling statistics.
        Ensures rolling windows are shifted by 1 to prevent target data leakage.
        """
        df = df_ts.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical temporal encodings (smooth periodic continuity)
        df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

        # Target variable: hourly total kWh
        target_series = df["total_kwh"]

        # Adapt lags & rolling windows when series is shorter than 168h (e.g., short custom uploads)
        active_lags = [lag for lag in self.lag_hours if lag < max(2, len(df) // 3)]
        if not active_lags and len(df) >= 4:
            active_lags = [1]

        for lag in active_lags:
            df[f"lag_{lag}h"] = target_series.shift(lag)

        # Rolling statistics (strictly on shifted target)
        shifted_target = target_series.shift(1)
        active_windows = [w for w in self.rolling_windows if w < len(df)]
        for w in active_windows:
            df[f"rolling_mean_{w}h"] = shifted_target.rolling(window=w, min_periods=1).mean()
            df[f"rolling_std_{w}h"] = shifted_target.rolling(window=w, min_periods=1).std().fillna(0.0)

        # Drop NaN rows resulting from max lag
        max_lag = max(active_lags) if active_lags else 0
        df_clean = df.iloc[max_lag:].reset_index(drop=True)
        logger.info(f"Engineered {df_clean.shape[1]} features across {len(df_clean):,} valid rows.")
        return df_clean

    def train_test_split_chronological(
        self,
        df_feat: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Splits dataset chronologically into train and test sets to guarantee
        that future information is never leaked into training folds.
        """
        drop_cols = ["timestamp", "total_kwh", "avg_kwh", "readings_count"]
        feature_cols = [c for c in df_feat.columns if c not in drop_cols]
        self.feature_names = feature_cols

        X = df_feat[feature_cols]
        y = df_feat["total_kwh"]
        timestamps = df_feat["timestamp"]

        n_total = len(df_feat)
        n_test = min(max(self.horizon_hours, int(n_total * self.test_ratio)), max(1, n_total // 3))
        n_train = n_total - n_test

        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
        ts_train, ts_test = timestamps.iloc[:n_train], timestamps.iloc[n_train:]

        logger.info(
            f"Chronological split: {len(X_train)} train steps ({ts_train.iloc[0]} to {ts_train.iloc[-1]}) | "
            f"{len(X_test)} test steps ({ts_test.iloc[0]} to {ts_test.iloc[-1]})"
        )
        return X_train, X_test, y_train, y_test, ts_train, ts_test

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains both Linear Regression baseline and Random Forest Regressor."""
        logger.info(f"Training Linear Regression baseline on {len(X_train)} samples...")
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train, y_train)

        logger.info(f"Training Random Forest Regressor (n_estimators=100, max_depth=12)...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=self.random_seed,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")

    def evaluate_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates MAE, RMSE, and R² for both models on unseen out-of-time test data.
        """
        if self.lr_model is None or self.rf_model is None:
            raise ValueError("Models must be trained before evaluation.")

        pred_lr = self.lr_model.predict(X_test)
        pred_rf = self.rf_model.predict(X_test)

        # Enforce non-negativity constraint on electricity demand
        pred_lr = np.clip(pred_lr, a_min=0.0, a_max=None)
        pred_rf = np.clip(pred_rf, a_min=0.0, a_max=None)

        def get_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100.0
            return {
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "r2_score": round(float(r2), 4),
                "mape_pct": round(float(mape), 2)
            }

        metrics = {
            "linear_regression": get_metrics(y_test, pred_lr),
            "random_forest": get_metrics(y_test, pred_rf)
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  Linear Regression: MAE={metrics['linear_regression']['mae']}, RMSE={metrics['linear_regression']['rmse']}, R2={metrics['linear_regression']['r2_score']}")
        logger.info(f"  Random Forest:     MAE={metrics['random_forest']['mae']}, RMSE={metrics['random_forest']['rmse']}, R2={metrics['random_forest']['r2_score']}")
        return metrics

    def save_artifacts(
        self,
        metrics: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        ts_test: pd.Series,
        output_dir: str = "dataset/processed/analytics_summary",
        save_models: bool = True
    ) -> None:
        """Saves trained models, evaluation metrics, and comparison datasets."""
        root = get_project_root()
        models_dir = root / self.config["paths"]["models_dir"]

        target_summary_dir = root / output_dir if not os.path.isabs(str(output_dir)) else Path(output_dir)
        target_summary_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Serialized Models (only when requested so custom runs don't overwrite defaults)
        if save_models:
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.lr_model, models_dir / "linear_regression.joblib")
            joblib.dump(self.rf_model, models_dir / "random_forest.joblib")
            joblib.dump(self.feature_names, models_dir / "feature_names.joblib")
            with open(models_dir / "forecasting_evaluation.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        # 2. Save Metrics
        with open(target_summary_dir / "forecasting_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # 3. Save Actual vs Predicted Comparison Dataset
        pred_lr = np.clip(self.lr_model.predict(X_test), 0.0, None)
        pred_rf = np.clip(self.rf_model.predict(X_test), 0.0, None)

        comparison_df = pd.DataFrame({
            "timestamp": ts_test.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_kwh": y_test.values,
            "pred_linear_regression": pred_lr,
            "pred_random_forest": pred_rf,
            "error_lr": np.abs(y_test.values - pred_lr),
            "error_rf": np.abs(y_test.values - pred_rf)
        })
        comparison_df.to_csv(target_summary_dir / "forecast_comparison.csv", index=False)

        # 4. Feature Importance for Random Forest
        importances = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.rf_model.feature_importances_
        }).sort_values("importance", ascending=False)
        importances.to_csv(target_summary_dir / "feature_importance.csv", index=False)

        # 5. Future Out-of-Sample Forecast (Next 24 Hours)
        last_row = X_test.iloc[-1:].copy()
        last_ts = pd.to_datetime(ts_test.iloc[-1])
        future_steps = []
        current_features = last_row.copy()

        for h in range(1, self.horizon_hours + 1):
            future_ts = last_ts + pd.DateOffset(hours=h)
            # Update temporal features
            current_features["hour"] = future_ts.hour
            current_features["day_of_week"] = future_ts.dayofweek
            current_features["month"] = future_ts.month
            current_features["is_weekend"] = int(future_ts.dayofweek in [5, 6])
            current_features["sin_hour"] = np.sin(2 * np.pi * future_ts.hour / 24.0)
            current_features["cos_hour"] = np.cos(2 * np.pi * future_ts.hour / 24.0)
            current_features["sin_dow"] = np.sin(2 * np.pi * future_ts.dayofweek / 7.0)
            current_features["cos_dow"] = np.cos(2 * np.pi * future_ts.dayofweek / 7.0)
            current_features["sin_month"] = np.sin(2 * np.pi * future_ts.month / 12.0)
            current_features["cos_month"] = np.cos(2 * np.pi * future_ts.month / 12.0)

            # Predict next step
            lr_pred = max(0.0, float(self.lr_model.predict(current_features)[0]))
            rf_pred = max(0.0, float(self.rf_model.predict(current_features)[0]))

            future_steps.append({
                "timestamp": future_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "forecast_hour_ahead": h,
                "forecast_linear_regression": round(lr_pred, 3),
                "forecast_random_forest": round(rf_pred, 3),
                "lower_bound_80": round(rf_pred * 0.88, 3),
                "upper_bound_80": round(rf_pred * 1.12, 3)
            })

            # Roll lag_1h forward
            if "lag_1h" in current_features:
                current_features["lag_1h"] = rf_pred

        future_df = pd.DataFrame(future_steps)
        future_df.to_csv(target_summary_dir / "future_forecast.csv", index=False)

        logger.info("Saved all forecasting artifacts, model binaries, and comparison CSVs.")


def run_forecasting(
    sample_mode: bool = False,
    custom_parquet: Optional[Path] = None,
    output_dir: Optional[str] = None,
    save_models: bool = True
) -> Dict[str, Any]:
    """Helper runner function to train, evaluate, and save models."""
    forecaster = DemandForecaster()
    df_ts = forecaster.prepare_hourly_series(sample_mode=sample_mode, custom_parquet=custom_parquet)
    if len(df_ts) < 12:
        raise ValueError(
            f"Dataset contains only {len(df_ts)} hourly steps; at least 12 hourly intervals are required for demand forecasting."
        )
    df_feat = forecaster.engineer_features(df_ts)
    X_train, X_test, y_train, y_test, ts_train, ts_test = forecaster.train_test_split_chronological(df_feat)

    forecaster.train_models(X_train, y_train)
    metrics = forecaster.evaluate_models(X_test, y_test)

    summary_dir = output_dir or ("dataset/processed/sample_analytics_summary" if sample_mode else "dataset/processed/analytics_summary")
    forecaster.save_artifacts(metrics, X_test, y_test, ts_test, output_dir=summary_dir, save_models=save_models)

    print("\n=== [SECA] DEMAND FORECASTING EVALUATION ===")
    print("Linear Regression (Baseline):")
    print(f"  MAE:  {metrics['linear_regression']['mae']:.4f} kWh")
    print(f"  RMSE: {metrics['linear_regression']['rmse']:.4f} kWh")
    print(f"  R²:   {metrics['linear_regression']['r2_score']:.4f}")
    print("Random Forest Regressor (Nonlinear Comparison):")
    print(f"  MAE:  {metrics['random_forest']['mae']:.4f} kWh")
    print(f"  RMSE: {metrics['random_forest']['rmse']:.4f} kWh")
    print(f"  R²:   {metrics['random_forest']['r2_score']:.4f}")
    print(f"Artifacts exported to: {summary_dir} and models/\n")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Machine Learning Demand Forecasting Module")
    parser.add_argument("--sample-mode", action="store_true", help="Train on sample dataset")
    args = parser.parse_args()

    run_forecasting(sample_mode=args.sample_mode)
