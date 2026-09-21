# YourName_ProjectName

Project: DataGuard AI — sample analysis and profiling demo

Files included:

- `YourName_ProjectName.ipynb` — Jupyter notebook demonstrating loading and summarising the new `customer_profiles_enriched.csv` dataset.
- `requirements.txt` — Python dependencies needed to run the notebook and generate the Word report.
- `YourName_ProjectReport.docx` — Project report (generated programmatically).

Setup and run

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the notebook with Jupyter or open in VS Code.
3. The notebook saves a small `submission_profile_summary.csv` in the working directory.

Dataset

The notebook reads `sample_data/customer_profiles_enriched.csv` from the repository sample_data folder. Ensure the repository root is the current working directory.

Description

This project demonstrates basic data profiling steps for a synthetic customer-level dataset and shows how to integrate such aggregates into a broader data quality workflow.
