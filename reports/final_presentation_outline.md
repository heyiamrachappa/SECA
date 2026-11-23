# Final Presentation Outline: Smart Electricity Consumption Analytics and Demand Forecasting

**Presentation Title:** Smart Electricity Consumption Analytics & Demand Forecasting Using Big Data  
**Student Name / Project Team:** Big Data Analytics Capstone  
**Target Audience:** University Evaluators, Academic Examiners, and Industry Reviewers  
**Duration:** 15–20 Minutes (including live demonstration and Q&A)  

---

## Slide 1: Title & Introduction
- **Project Title:** Smart Electricity Consumption Analytics and Demand Forecasting Using Big Data
- **Domain:** Big Data Engineering & Machine Learning
- **Dataset:** Low Carbon London SmartMeter Energy Consumption Dataset (UK Power Networks / Greater London Authority)
- **Primary Objective:** Build a distributed, end-to-end data processing, analytics, machine learning forecasting, and interactive monitoring platform for smart electrical grids.

---

## Slide 2: Problem Statement & Industrial Motivation
- **Smart Grid Telemetry Challenges:**
  - 5,567 London households generating half-hourly consumption telemetry (~167 million readings).
  - High velocity and volume cause fatal memory exhaustion in traditional single-node tools (Pandas).
  - Data quality defects: meter transmission drops (`'Null'` strings), erratic sensor spikes, duplicate timestamps.
- **Project Goal:** Design an out-of-core distributed pipeline using Apache Spark, columnar Parquet storage, Scikit-learn ML models, and a responsive Streamlit dashboard.

---

## Slide 3: System Architecture & Tech Stack
- **Distributed Engine:** Apache Spark 4.3.0 & Spark SQL (Catalyst Optimizer, out-of-core partition streaming).
- **Runtime:** Eclipse Temurin OpenJDK 17 LTS on 16 CPU cores.
- **Storage:** Snappy-compressed Apache Parquet (85% storage space compression with pushdown predicates).
- **Machine Learning:** Scikit-learn (Linear Regression baseline, Random Forest Regressor, Isolation Forest).
- **Interface:** Streamlit & Plotly (GPU-accelerated interactive web charts).

---

## Slide 4: Data Engineering & Preprocessing Pipeline
- **Schema Discovery:** Normalization of raw headers (`LCLid`, `stdorToU`, `DateTime`, `KWH/hh (per half hour) `).
- **Cleansing & Validation:**
  - Replaced literal `'Null'` and whitespace tokens with numeric `0.0`.
  - Filtered unphysical single-home half-hour readings ($> 25.0$ kWh).
  - Strict deduplication on `(household_id, timestamp)`.
- **Temporal Feature Engineering:** Extracted `hour`, `day`, `month`, `year`, `week_of_year`, `day_of_week`, and `is_weekend`.
- **Storage Impact:** Reduced raw CSV footprint from 242 MB down to **36 MB Parquet** (~85% compression).

---

## Slide 5: Core Consumption Analytics (11 Key Queries)
- **Total Processed Telemetry:** 4,996,561 readings across 152 households.
- **Total Energy Consumed:** 1,088,958.59 kWh (Mean: 0.2179 kWh/half-hour).
- **Diurnal Load Dynamics:**
  - Base nighttime valley: 03:00 - 05:00 (~0.134 kWh).
  - Morning activation ramp: 07:00 - 09:00 (~0.220 kWh).
  - Evening peak demand: 18:30 - 21:00 (~0.385 kWh).
- **Weekday vs. Weekend Behaviors:** Delayed morning wake-up ramp on weekends; higher sustained midday load.

---

## Slide 6: Machine Learning — Demand Forecasting Methodology
- **Objective:** Short-Term Load Forecasting (STLF) for future grid dispatch planning.
- **Validation Discipline:**
  - **Chronological Split:** 80% train (15,504 hourly steps) vs. 20% test (3,876 hourly steps).
  - **Zero Leakage:** Lags ($t-1, t-2, t-3, t-24, t-48, t-168$) and rolling stats strictly computed on past values.
  - Periodic sine/cosine cyclic encodings for hour, day of week, and month.

---

## Slide 7: Forecasting Experimental Results & Comparison
| Metric | Linear Regression (Baseline) | Random Forest Regressor (Nonlinear) | Performance Gain |
| :--- | :--- | :--- | :--- |
| **MAE** | 3.8917 kWh | **3.6306 kWh** | **6.7% Error Reduction** |
| **RMSE** | 4.9891 kWh | **4.7264 kWh** | **5.3% Error Reduction** |
| **$R^2$ Score** | 0.9289 | **0.9362** | **Higher explained variance** |
- **Conclusion:** Random Forest effectively captures complex non-linear evening peak load shifts compared to the linear baseline.
- **Out-of-Sample Horizon:** 24-hour ahead future projection with 80% confidence interval bands.

---

## Slide 8: Anomaly Detection Engine
- **Dual Detection Strategy:**
  1. *Statistical Z-Score:* Diurnal baseline normalization ($\theta = 3.0$ standard deviations per household per hour).
  2. *Unsupervised ML:* Scikit-learn Isolation Forest ($2\%$ contamination rate).
- **Classification Categories:**
  - `HIGH_CONSUMPTION`: Surge events, potential HVAC/equipment faults.
  - `LOW_CONSUMPTION`: Unoccupied properties, localized feeder outages, packet drops.
- **Consensus:** 1,019 events dual-confirmed by both algorithms out of 100,000 analyzed records.
- **Academic Disclaimer:** Anomalies are statistical signals for field inspection, not automatic proofs of theft.

---

## Slide 9: Interactive Streamlit Dashboard Architecture
- **6 Integrated Views:**
  1. *Overview:* Executive KPI metric cards, scope metadata, diurnal load preview.
  2. *Consumption Analytics:* Hourly, daily trend with range slider, monthly seasonality, weekday vs weekend.
  3. *Demand Forecasting:* Actual vs predicted curves, model selector, 24-hour future forecast, feature importance.
  4. *Anomaly Detection:* Scatter plot vs baselines, diurnal distribution, filterable anomaly log table.
  5. *Household Analysis:* Individual meter drilldown, diurnal load curve, top consumer ranking.
  6. *Data Explorer:* Filterable table browser, one-click CSV artifact export.
- **Performance Secret:** In-memory caching (`@st.cache_data`) over precomputed analytical summaries ensures $< 100$ ms UI interactions without blocking on Spark.

---

## Slide 10: Conclusion & Key Learnings
- Successfully engineered an industrial-grade Big Data pipeline processing millions of smart meter records.
- Apache Spark + Parquet demonstrated scalable out-of-core performance with zero memory saturation.
- Rigorous time-series forecasting demonstrated that Random Forest outperforms linear models ($R^2 = 0.9362$) without data leakage.
- Clean separation of concerns: Ingestion ➔ Preprocessing ➔ Spark SQL ➔ ML ➔ Anomaly Detection ➔ Streamlit Dashboard.
- **Live Demonstration:** `python main.py --phase dashboard`
