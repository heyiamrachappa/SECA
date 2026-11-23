# Data Quality and Preprocessing Report

**Generated Timestamp:** Column<'current_timestamp()'>  
**Dataset Name:** Low Carbon London SmartMeter Energy Consumption Dataset  

---

## 1. Quality & Cleansing Metrics

| Metric | Value |
| :--- | :--- |
| **Initial Telemetry Records** | 36,000 |
| **Cleaned & Verified Records** | 36,000 |
| **Filtered / Deduplicated Records** | 0 |
| **Data Retention Rate** | **100.0%** |
| **Consumption Validation Range** | [0.0 kWh, 25.0 kWh] |

---

## 2. Transformations Applied

1. **Missing Value Resolution:** String `'Null'` and blank entries converted to numeric `0.0`.
2. **Physical Boundary Enforcement:** Half-hourly consumption constrained to `0.0` - `25.0` kWh.
3. **Timestamp Standardization:** Converted to native Spark `TimestampType` (UTC).
4. **Duplicate Purging:** Strict deduplication on `(household_id, timestamp)`.
5. **Engineered Features:** Added `hour`, `day`, `month`, `year`, `week_of_year`, `day_of_week`, `day_name`, and `is_weekend`.
6. **Storage Optimization:** Exported to Snappy-compressed Apache Parquet format for fast vectorized downstream query execution.
