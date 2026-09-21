# DataGuard AI 🛡️

**AI-Powered Data Reliability & AI-Readiness Analyst**

DataGuard AI profiles any CSV or Excel dataset across six quality dimensions, investigates *why* issues exist using AI root-cause analysis, quantifies business impact, and delivers prioritised remediation recommendations — so you know whether your data can actually be trusted before you build a dashboard, ML model, or AI agent.

---

## What it does

| Stage | What happens |
|---|---|
| **Profiling** | Completeness, consistency, uniqueness, validity, schema & freshness scored per column |
| **AI Root-Cause** | Detects outlier clusters by date window, source category & correlated missingness |
| **Business Impact** | Estimates monetary exposure and downstream dashboard/model risk |
| **Recommendations** | Prioritised action plan ordered by impact × effort |
| **Dashboard** | Interactive Streamlit UI with gauge, column explorer, issue drill-down, and recommendations filter |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the sample dataset (optional)
python generate_sample_data.py

# 3. Launch the dashboard
streamlit run ui/app.py
```

Then open http://localhost:8501 and either upload your own CSV/Excel or click **Load sample dataset**.

---

## Architecture

```
CSV / Excel
    ↓
core/profiler.py     ← 7 quality dimensions, per-column stats, IQR outlier detection
    ↓
core/scorer.py       ← Weighted AI-Readiness score (0–100) + grade A–F
    ↓
ai/root_cause.py     ← Missingness correlation, outlier clustering by date & category
    ↓
ai/impact.py         ← Monetary exposure estimation, downstream risk classification
    ↓
ai/recommendations.py ← Prioritised action plan (effort × impact matrix)
    ↓
ui/app.py            ← Streamlit dashboard (6 tabs)
```

---

## Sample Dataset

`sample_data/customer_transactions.csv` (2,070 rows) contains deliberate quality problems:

- **6.4%** missing email addresses
- **3.9%** missing region values (correlated with missing product)  
- **6.0%** revenue outliers (₹500K–₹2.5M vs normal ₹500–₹50K range)
- **3.4%** duplicate rows
- **8%** mixed-type `order_ref` column (integers + strings)

### New: Customer Profiles Dataset

`sample_data/customer_profiles_enriched.csv` — synthetic customer-level aggregates designed to complement the transaction and product datasets:

- **Schema:** `customer_id, email, region, total_spent_inr, num_orders, avg_order_value, first_order_date, last_order_date, favorite_product, return_rate, is_vip`
- **Purpose:** Supports segmentation, lifetime value analysis, churn and VIP detection, and recommendation system testing without altering source transactions.
- **Quality cues injected:** some missing `email` values, zero `total_spent_inr` for inactive customers, and a small share of high-spend VIPs to test outlier detection and profiling behavior.

Use this file to test downstream aggregation analyses and to validate how the profiler handles aggregate-level datasets.

---

## Project Structure

```
dataguard_ai/
├── core/
│   ├── profiler.py          # Data profiling engine
│   └── scorer.py            # AI-Readiness scorer
├── ai/
│   ├── root_cause.py        # Root-cause analysis
│   ├── impact.py            # Business impact estimation
│   └── recommendations.py   # Remediation recommendations
├── ui/
│   └── app.py               # Streamlit dashboard
├── sample_data/
│   └── customer_transactions.csv
├── generate_sample_data.py
├── requirements.txt
└── README.md
```

---

## Interview Talking Points

> *"I realised that businesses don't just have an analytics problem. Before trusting an insight, they need to know whether the underlying data is reliable. So I built an AI-assisted data reliability system that profiles datasets, detects anomalies and quality issues, identifies potential root causes, quantifies business impact and recommends remediation — directly addressing what Gartner calls 'AI-ready data' as a critical 2026 capability."*
