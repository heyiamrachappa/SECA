# Low Carbon London SmartMeter Dataset Inspection Report

**Generated Date:** 2026-11-23 23:20:23  
**Data Source Directory:** `/home/spidey/PROJECTS/SECA/dataset/sample`  
**Total CSV Block Files:** 1  
**Total Telemetry Records:** 36,000  
**Total Disk Footprint:** 1.68 MB  

---

## 1. Schema & Column Headers Analysis

Inspection of `london_smartmeter_sample.csv`:
- **Raw Headers:** `['LCLid', 'stdorToU', 'DateTime', 'KWH/hh (per half hour) ']`
- **Standardized Headers:** `['LCLid', 'stdorToU', 'DateTime', 'KWH/hh (per half hour)']`
- **Schema Validation Status:** PASSED (Conforms to Low Carbon London spec)

### Column Data Dictionary
1. `LCLid` (String): Unique identifier assigned to each household during the trial.
2. `stdorToU` (String): Tariff type (`Std` for Flat-rate 14.228p/kWh, `ToU` for Dynamic Time of Use).
3. `DateTime` (String): Half-hourly reading timestamp formatted as `YYYY-MM-DD HH:MM:SS.0000000`.
4. `KWH/hh (per half hour) ` (String/Float): Electricity consumed during the 30-minute interval in kilowatt-hours. Contains string `'Null'` for missing meter telemetry.

---

## 2. Partition Block Summary

| File Name | File Size (MB) | Record Count | Sample Households | Nulls in Sample |
| :--- | :--- | :--- | :--- | :--- |
| `london_smartmeter_sample.csv` | 1.68 | 36,000 | 7 | 19 |

---

## 3. Data Quality Findings & Ingestion Directives

1. **Trailing Whitespace in Headers:** Header `KWH/hh (per half hour) ` contains trailing whitespace. PySpark ingestion must sanitize and trim column names.
2. **Missing Value Representation:** Meter disconnections and transmission drops are represented as literal string `"Null"`. Preprocessing must convert these to numeric `NaN` / `0.0`.
3. **Timestamp Precision:** Timestamps include microsecond suffixes (`.0000000`). Preprocessing must parse them cleanly into UTC timestamps.
