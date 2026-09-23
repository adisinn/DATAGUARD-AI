<div align="center">

# 🛡️ DataGuard AI

### Turn messy datasets into trustworthy decisions.

**AI-assisted data reliability, root-cause analysis, and AI-readiness scoring for CSV and Excel data.**

<p>
  <a href="https://github.com/adisinn/DATAGUARD-AI"><img src="https://img.shields.io/badge/status-active-22c55e?style=flat-square" alt="Project status" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/FastAPI-backend-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Ollama-local%20AI-black?style=flat-square" alt="Ollama" />
</p>

<p>
  <a href="#-quick-start">Quick start</a> ·
  <a href="#-what-it-does">Features</a> ·
  <a href="#-technical-architecture">Architecture</a> ·
  <a href="#-sample-datasets">Samples</a>
</p>

</div>

---

## The problem

A dashboard can be beautiful, an ML model can be accurate, and an executive report can be persuasive — while the underlying data is quietly wrong.

DataGuard AI helps teams answer four questions before trusting a dataset:

1. **What is wrong?**
2. **Why is it happening?**
3. **What could it affect?**
4. **What should we fix first?**

Upload a CSV or Excel file and receive a practical data health report with quality scores, issue evidence, estimated business impact, and a prioritized remediation plan.

> **Data quality is not just a technical concern. It is a decision-risk concern.**

---

## ✨ What it does

| Capability | Outcome |
|---|---|
| **Deep profiling** | Measures completeness, validity, consistency, uniqueness, schema integrity, and freshness |
| **AI-readiness scoring** | Produces a weighted score from **0–100** and a grade from **A–F** |
| **Root-cause investigation** | Finds correlated missingness, date-window clusters, category concentration, and format anomalies |
| **Business impact analysis** | Estimates affected records, monetary exposure, and downstream dashboard/model risk |
| **Remediation planning** | Ranks fixes by impact × effort and highlights quick wins |
| **Interactive exploration** | Lets users inspect columns, issues, evidence, raw data, and recommendations in one workspace |
| **Local AI analyst** | Uses Ollama for narrative reports and dataset-aware chat without sending data to a hosted LLM |

---

## 🎬 Product experience

The application is designed as a decision workflow rather than a static profiling report:

```text
Upload data
    ↓
Profile every column
    ↓
Score data health and AI-readiness
    ↓
Investigate root causes
    ↓
Quantify business impact
    ↓
Prioritize remediation
    ↓
Ask the AI analyst what to do next
```

The dashboard includes seven views:

- **Score & Dimensions** — overall AI-readiness score, grade, verdict, and dimension breakdown
- **Column Intelligence** — missingness, uniqueness, outliers, formats, distributions, and numeric statistics
- **Issues & Root Cause** — severity-ranked problems with supporting evidence
- **Business Impact** — affected records, monetary exposure, and downstream risks
- **Remediation Plan** — prioritized actions with effort and impact filters
- **Raw Data** — preview and export analyzed data
- **AI Analyst** — executive narrative generation and conversational investigation

---

## 🚀 Quick start

### Option A — Streamlit dashboard

```bash
# Clone the repository
git clone https://github.com/adisinn/DATAGUARD-AI.git
cd DATAGUARD-AI

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate      # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run ui/app.py
```

Open **http://localhost:8501** in your browser.

Upload a `.csv`, `.xlsx`, or `.xls` file — or select one of the included sample datasets.

### Option B — FastAPI application

The repository also includes `server.py`, which exposes the analysis engine through an HTTP API and serves the static frontend.

Install the API runtime if needed:

```bash
pip install fastapi uvicorn python-multipart
python server.py
```

Open **http://localhost:8000**.

> **Note:** the Streamlit dashboard is the simplest way to explore the project. The FastAPI server is useful when integrating the analysis engine into another frontend or workflow.

### Optional — enable the local AI analyst

DataGuard AI works with deterministic analysis even when an LLM is unavailable. To enable local narrative generation and chat:

```bash
# Install Ollama from https://ollama.com
ollama serve
ollama pull llama3.2
```

Refresh the application and the AI Analyst will become available. Your uploaded data remains local to your machine.

---

## 🔍 Quality dimensions

DataGuard AI evaluates six dimensions of data health:

| Dimension | What it checks |
|---|---|
| **Completeness** | Missing cells, null-heavy columns, and incomplete records |
| **Validity** | Statistical outliers and values outside expected ranges |
| **Uniqueness** | Duplicate rows and low-cardinality identity fields |
| **Consistency** | Format variation, mixed representations, and string anomalies |
| **Schema** | Structural issues, empty columns, and unexpected data shapes |
| **Freshness** | Data age and potential freshness-SLA violations |

The weighted AI-readiness score makes the report easy to communicate while preserving the detailed evidence needed for engineering and analytics work.

---

## 🧠 Root-cause analysis

Most profiling tools stop at “this column has missing values.” DataGuard AI attempts to explain the pattern behind the issue.

Examples of evidence it can surface include:

- missing values concentrated in the same records across multiple columns
- outliers grouped inside a narrow ingestion window
- anomalies concentrated in one source category or business segment
- mixed-type identifiers caused by inconsistent upstream serialization
- stale data that may violate downstream freshness expectations

The goal is not to declare a definitive cause when the data cannot prove one. The goal is to provide useful evidence and plausible causes that guide investigation.

---

## 💼 Business impact

Technical data issues become easier to prioritize when they are connected to business consequences.

The impact layer can summarize:

- affected rows and affected percentage
- estimated monetary exposure when value columns are available
- downstream dashboard and KPI risk
- potential model-training and decision-quality risk
- high-, medium-, and low-risk issue counts

This helps teams move from:

> “There are outliers in the revenue column.”

to:

> “The outliers affect a measurable portion of the dataset, may distort revenue reporting, and should be investigated before the next reporting cycle.”

---

## ✅ Remediation planning

Every detected issue can feed into a prioritized action plan. Recommendations include:

- the proposed action
- the affected quality dimension
- expected impact
- estimated effort
- issue reference and supporting detail

The dashboard also highlights **quick wins**: high-impact changes that should be relatively easy to implement.

---

## 🏗️ Technical architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    CSV / Excel Input                     │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                 core/profiler.py                        │
│  Column profiles · missingness · duplicates · outliers   │
│  schema checks · distributions · freshness               │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                  core/scorer.py                         │
│       Weighted quality dimensions → 0–100 score          │
│                         → A–F grade                      │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                 ai/root_cause.py                        │
│  Correlated missingness · temporal clusters · categories │
│                   evidence generation                   │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    ai/impact.py                         │
│    Affected records · monetary exposure · risk levels   │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│               ai/recommendations.py                     │
│          Impact × effort prioritization and plan         │
└───────────────────────┬──────────────────┬───────────────┘
                        │                  │
                        ▼                  ▼
              Streamlit dashboard     FastAPI backend
                 ui/app.py              server.py
                        │                  │
                        └────────┬─────────┘
                                 ▼
                         Optional local AI
                              Ollama
```

### Core analysis contract

The analysis pipeline follows a simple composition model:

```python
from dataguard_ai.core import run_profile
from dataguard_ai.ai import run_analysis

profile, quality_score = run_profile(dataframe)
analysis = run_analysis(dataframe, profile, quality_score)
```

The resulting objects contain the profile, quality score, root-cause report, business-impact report, and recommendation report used by both application surfaces.

### Detection approach

The project combines deterministic checks and analytical heuristics:

- Pandas-based column profiling
- duplicate and missingness calculations
- IQR-based numerical outlier detection
- format anomaly checks for string columns
- date-window and category-cluster analysis
- weighted quality scoring
- rule-based impact estimation
- optional Ollama-generated narrative and chat responses

---

## 📁 Project structure

```text
DATAGUARD-AI/
├── ai/
│   ├── __init__.py
│   ├── impact.py              # Business impact estimation
│   ├── llm.py                 # Ollama status and chat integration
│   ├── narrative.py           # Executive narrative generation
│   ├── recommendations.py     # Prioritized remediation plan
│   └── root_cause.py          # Issue detection and evidence
├── core/
│   ├── __init__.py
│   ├── profiler.py            # Dataset and column profiling
│   └── scorer.py              # AI-readiness scoring
├── sample_data/               # Deliberately imperfect datasets
├── static/                    # FastAPI-served frontend assets
├── ui/
│   └── app.py                # Streamlit dashboard
├── generate_extra_datasets.py
├── generate_retail_inventory.py
├── generate_sample_data.py
├── Modelfile                 # Optional local model configuration
├── pyproject.toml
├── requirements.txt
├── run.sh
├── server.py                 # FastAPI application
└── README.md
```

---

## 📊 Sample datasets

The included datasets contain intentional quality problems so the full analysis workflow can be tested immediately.

### Customer transactions

`sample_data/customer_transactions.csv`

Designed to demonstrate:

- missing email addresses
- correlated missing region and product values
- revenue outliers
- duplicate transactions
- mixed-type order references

### Customer profiles

`sample_data/customer_profiles_enriched.csv`

Synthetic customer-level aggregates for:

- segmentation
- lifetime-value analysis
- churn and VIP detection
- return-rate analysis
- aggregate-level profiling

Additional sample domains include HR records, hospital admissions, e-commerce catalogs, and retail inventory data.

---

## 🧪 Example findings

A typical report may identify findings such as:

- revenue outliers concentrated in a narrow ingestion window
- missing region values correlated with another incomplete field
- duplicate rows inflating transaction or customer counts
- inconsistent identifier formats that may break joins
- stale data that could violate reporting expectations

The exact results depend on the uploaded dataset and its columns.

---

## 🧩 Technology stack

| Layer | Technologies |
|---|---|
| Data processing | Python, Pandas, NumPy, SciPy |
| Spreadsheet support | OpenPyXL |
| Dashboard | Streamlit |
| API | FastAPI, Uvicorn |
| Local AI | Ollama |
| Packaging | Setuptools, `pyproject.toml` |

Python **3.10 or newer** is recommended.

---

## 🎯 Use cases

- Validate data before publishing a dashboard
- Assess AI/ML readiness before model training
- Investigate recurring ETL and ingestion defects
- Review CRM, finance, HR, healthcare, and e-commerce datasets
- Quantify data risk for non-technical stakeholders
- Create a repeatable first-pass data health review

---

## 🔐 Privacy and local processing

The core profiling workflow runs locally. Uploaded datasets are processed by the application on your machine.

If Ollama is enabled, the optional AI analyst also runs through a local model rather than requiring a hosted LLM API. Always review your own deployment configuration before using sensitive production data.

---

## 🗺️ Roadmap

- [ ] Dataset comparison across time periods
- [ ] Exportable HTML and PDF reports
- [ ] Warehouse connectors
- [ ] BI and pipeline integrations
- [ ] Configurable quality rules and thresholds
- [ ] More advanced anomaly-detection strategies
- [ ] Historical data-health tracking

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make your change and add supporting documentation
4. Test the dashboard or analysis workflow
5. Open a pull request with a clear explanation of the change

Good contribution areas include new quality rules, better evidence generation, additional sample datasets, UI improvements, and integrations.

---

## 📄 License

No license file is currently included in the repository. If you plan to distribute or reuse this project publicly, add an appropriate license before publishing it as an open-source package.

---

<div align="center">

### Trust your data before it costs you the decision.

**DataGuard AI** · profile it · understand it · improve it

</div>
