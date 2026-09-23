# DataGuard AI 🛡️

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas" alt="Pandas" />
</p>

<p align="center">
  <strong>AI-powered data reliability and AI-readiness analysis for messy real-world datasets.</strong>
</p>

DataGuard AI is a data quality and AI-readiness platform that profiles CSV and Excel datasets, finds the root causes of bad data, estimates business impact, and turns those findings into a prioritized remediation plan. It helps teams answer one crucial question before trusting analytics or ML models:

> Is this data reliable enough to drive decisions?

---

## Why this project exists

Most teams don’t have a data quality problem. They have a trust problem.

By the time a report, dashboard, or AI model is built, the real issue is usually hidden in the data:

- missing values that skew segments and KPIs
- duplicate records that inflate revenue or customer counts
- outliers that distort forecasts and alerts
- schema drift that breaks pipelines and dashboards
- stale or inconsistent data that undermines trust in decisions

DataGuard AI gives you an operational view of data health, not just a column-by-column scan.

---

## What it does

DataGuard AI analyzes a dataset across six core quality dimensions and then explains why issues exist, what they cost, and what should be fixed first.

| Stage | What happens |
|---|---|
| Profiling | Measures completeness, consistency, uniqueness, validity, schema, and freshness per column |
| Root Cause Analysis | Detects missingness patterns, outlier clusters, date/category anomalies, and correlated quality failures |
| Business Impact | Estimates financial exposure, operational risk, and downstream KPI/model risk |
| Recommendations | Produces a prioritized remediation plan based on impact and effort |
| Dashboard | Presents results in an interactive UI with drill-down, filters, and scorecards |

### Quality dimensions covered

- Completeness
- Consistency
- Uniqueness
- Validity
- Schema integrity
- Freshness

### AI-readiness scoring

The project calculates an overall AI-readiness score from 0–100, along with a grade from A to F, helping teams quickly assess whether a dataset is trustworthy enough for analytics or model training.

---

## How it works

```text
CSV / Excel Dataset
        ↓
Data Profiling Engine
        ↓
Quality Scoring
        ↓
Root Cause Analysis
        ↓
Impact Estimation
        ↓
Priority Recommendations
        ↓
Interactive Dashboard / API
```

The pipeline combines rule-based profiling with analytical detection to surface:

- hidden data quality issues
- unusual clusters and outliers
- missingness correlations across columns
- likely business consequences of dirty records
- a clear action plan for improvement

---

## ✨ Key features

### 1. Data quality profiling

Automatically inspects each column for:

- missing values and null patterns
- duplicates and near-duplicates
- outlier detection using statistical thresholds
- dtype mismatches and schema drift
- freshness problems and stale records

### 2. AI root-cause analysis

Goes beyond flags and explains patterns such as:

- missing region values correlated with missing product data
- outliers concentrated in a specific date window
- recurring anomalies by category or segment
- format irregularities causing downstream failures

### 3. Business impact quantification

Translates poor-quality data into business language:

- revenue exposure
- affected customer or transaction counts
- downstream reporting risk
- model / decision integrity risk

### 4. Remediation planning

Not just “what is wrong,” but “what matters most and what should be fixed first.”

Recommendations are ranked using a practical impact × effort lens so teams can act quickly.

### 5. Interactive UI

The project ships with a dashboard to explore:

- overall quality score and grade
- column-level metrics
- issue drill-downs
- recommendations by severity and effort
- raw data preview
- sample dataset testing

---

## 🚀 Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the app

```bash
python server.py
```

Then open:

```text
http://localhost:8000
```

### 3) Optional: use the Streamlit version

```bash
streamlit run ui/app.py
```

Then open:

```text
http://localhost:8501
```

---

## 📊 Sample datasets included

The repository includes realistic sample datasets designed to stress-test profiling and root-cause analysis.

### Customer Transactions

`sample_data/customer_transactions.csv`

Contains structured quality issues such as:

- missing email addresses
- correlated missing region and product values
- revenue outliers
- duplicate rows
- mixed-type order references

### Customer Profiles

`sample_data/customer_profiles_enriched.csv`

Synthetic customer-level aggregates used for:

- segmentation analysis
- LTV and churn monitoring
- VIP and return-rate testing
- aggregate-level data quality validation

---

## 🧠 Example outcomes

The system can surface insights like:

- “Revenue outliers are concentrated in a 2-day ingestion window.”
- “Missing region values are highly correlated with missing product descriptions.”
- “3.4% of transactions are duplicates and are inflating customer counts.”
- “Email completeness is below threshold and affects downstream personalization workflows.”

This turns raw data profiling into actionable operational intelligence.

---

## 🏗️ Project structure

```text
DATAGUARD-AI/
├── ai/
│   ├── __init__.py
│   ├── impact.py
│   ├── llm.py
│   ├── narrative.py
│   ├── recommendations.py
│   └── root_cause.py
├── core/
│   ├── __init__.py
│   ├── profiler.py
│   └── scorer.py
├── static/
│   └── ...
├── ui/
│   └── app.py
├── sample_data/
│   ├── customer_profiles_enriched.csv
│   ├── customer_transactions.csv
│   └── ...
├── __init__.py
├── generate_extra_datasets.py
├── generate_retail_inventory.py
├── generate_sample_data.py
├── pyproject.toml
├── requirements.txt
├── run.sh
├── server.py
├── README.md
└── Modelfile
```

---

## 🧩 Tech stack

- Python 3.10+
- Pandas
- NumPy
- SciPy
- OpenPyXL
- Streamlit
- FastAPI
- Ollama / LLM integration

---

## 🎯 Use cases

DataGuard AI is useful for:

- data quality reviews before dashboard publishing
- AI readiness checks before powering ML workflows
- ETL pipeline validation
- CRM, finance, HR, operations, and e-commerce data review
- stakeholder communication around data issues and remediation

---

## 📌 Why teams like it

Unlike basic data profile reports, DataGuard AI answers the questions that matter to decision-makers:

- What is wrong?
- Why is it happening?
- How much is it costing us?
- What should we fix first?

That makes it a strong fit for data teams, analytics teams, product teams, and AI/ML practitioners working with imperfect operational data.

---

## 🔮 Roadmap

Planned improvements include:

- more advanced anomaly detection models
- richer dataset comparisons over time
- exportable PDF/HTML reports
- integration with warehouse and BI ecosystems
- deeper LLM-generated summaries for business stakeholders

---

## Contributing

Contributions are welcome. If you want to improve the profiling engine, add new detection rules, extend the dashboard, or improve the documentation, feel free to open a pull request.

---

## A quick note

This project was built to bring data quality analysis into a more decision-friendly, business-aware workflow. It helps teams move from “the data looks odd” to “here is exactly what is wrong, why it matters, and what to do next.”

If you want to turn this into a polished public GitHub project, this README is now structured more like a product landing page than a basic project write-up.

---

<p align="center">
  <strong>DataGuard AI</strong> — trust your data before it costs you the decision.
</p>
