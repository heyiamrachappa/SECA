# Smart Electricity Consumption Analytics and Demand Forecasting Using Big Data

A semester-end Big Data Analytics project delivering an end-to-end distributed data processing, pattern analysis, machine learning forecasting, and interactive analytics dashboard for smart-meter energy consumption data.

---

## 1. Project Overview & Problem Statement

Modern smart electricity grids generate high-frequency, massive-volume time-series telemetry. Analyzing this data at scale requires distributed computing frameworks capable of handling hundreds of millions of records without encountering memory bottlenecks.

This project utilizes the **Low Carbon London SmartMeter Energy Consumption Dataset** (historical half-hourly electricity readings from 5,567 London households between November 2011 and February 2014, published by UK Power Networks via the London Datastore) to:
1. Ingest multi-file CSV telemetry using distributed partitions.
2. Cleanse and validate records using Apache Spark and Spark SQL.
3. Compute household-level, diurnal, seasonal, and peak-demand metrics.
4. Train predictive models (Linear Regression baseline and Random Forest) to forecast future consumption.
5. Detect statistical and behavioral consumption anomalies using Z-score filtering and Isolation Forest.
6. Surface all insights through an interactive Streamlit and Plotly analytical dashboard.

> **Academic Note on Dataset Scope:**  
> This dataset represents historical London household electricity readings. It is strictly maintained as historical London data and is never misrepresented as current or Indian grid data.

---

## 2. Technology Stack & Architectural Decisions

| Layer | Technology | Decision Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.14 / 3.10+ | Industry standard for data engineering, ML, and dashboarding. |
| **Distributed Engine** | Apache Spark (PySpark 4.3.0) | Out-of-core distributed processing engine; avoids single-node memory saturation. |
| **Query Engine** | Spark SQL & Catalyst Optimizer | High-throughput aggregations with predicate/column pushdown. |
| **JVM Runtime** | Eclipse Temurin OpenJDK 17 LTS | High-performance 64-bit Java Virtual Machine required by Spark. |
| **Storage Format** | Apache Parquet (Snappy compressed) | Columnar storage with schema preservation, high compression ratio, and pushdown filters. |
| **Machine Learning** | Scikit-learn (Linear Regression, Random Forest, Isolation Forest) | Robust, auditable algorithms for tabular time-series regression and unsupervised anomaly detection. |
| **In-Memory Analytics**| Pandas 2.3.x, NumPy, PyArrow | Optimized vectorization and zero-copy Apache Arrow serialization between JVM and Python workers. |
| **Visualization & UI** | Streamlit 1.67+, Plotly | Interactive, responsive web UI with GPU-accelerated web charts. |
| **Testing** | Pytest | Automated verification of data pipelines, schemas, and model metrics. |

---

## 3. Hardware & Environment Audit

The environment audit conducted on the development machine revealed:
- **Operating System:** Ubuntu 26.04.1 LTS (Linux kernel 7.0.0-43-generic x86_64)
- **CPU:** 13th Gen Intel(R) Core(TM) i5-13450HX (16 logical cores)
- **RAM:** 14 GiB total (~8.7 GiB available), with 4 GiB swap
- **Storage:** 466 GB NVMe SSD with 358 GB free space
- **Python:** 3.14.4
- **JVM:** Eclipse Temurin OpenJDK 17.0.21.1+2 (installed in user space `~/.jdk/jdk-17`)

### Optimization Strategy for Local Processing
With 16 CPU cores and 14 GiB RAM:
1. **Local Master:** Spark runs as `local[*]`, utilizing all 16 cores for multi-threaded executor tasks.
2. **Memory Limit:** JVM driver memory is set to `4g` to leave ample headroom (~4-5 GB) for OS, Streamlit, and Python ML tasks without paging to swap.
3. **Partitioning:** Default shuffle partitions tuned to `16` to match core concurrency and prevent excessive small-task scheduling overhead.

---

## 4. Project Directory Structure

```text
SECA/
├── config/
│   └── config.yaml             # Centralized pipeline and model configurations
├── dashboard/
│   ├── app.py                  # Streamlit dashboard entrypoint
│   ├── pages/                  # Multipage dashboard views
│   └── components/             # Reusable UI cards and chart components
├── dataset/
│   ├── raw/                    # Raw CSV files (git-ignored)
│   ├── processed/              # Cleaned Parquet partitions (git-ignored)
│   └── sample/                 # Small representative sample for quick testing
├── logs/                       # Application and Spark execution logs
├── models/                     # Serialized ML artifacts (.joblib / .pkl)
├── notebooks/                  # Exploratory and prototyping notebooks
├── reports/                    # Generated analytical summaries and figures
├── src/
│   ├── __init__.py
│   ├── ingestion/              # PySpark multi-file ingestion pipeline
│   ├── preprocessing/          # Data cleaning, timestamp parsing, feature extraction
│   ├── analytics/              # Spark SQL aggregations (diurnal, weekly, peak demand)
│   ├── forecasting/            # Time-series ML pipeline (Linear Regression & RF)
│   ├── anomaly_detection/      # Statistical & Isolation Forest anomaly detectors
│   └── utils/
│       ├── config.py           # Configuration loader
│       ├── logger.py           # Structured logging
│       └── spark_session.py    # SparkSession factory with auto JVM discovery
├── tests/
│   ├── __init__.py
│   └── test_environment.py     # Environment and dependencies test suite
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Pinned Python package dependencies
├── main.py                     # CLI orchestrator entrypoint
└── README.md                   # Complete project documentation
```

---


