"""
Generate three additional sample datasets for DataGuard AI:
  1. hr_employee_records.csv     — HR / People Analytics domain
  2. hospital_patient_admissions.csv — Healthcare domain
  3. ecommerce_product_catalog.csv   — E-commerce / Inventory domain

Each dataset has a unique column schema and a distinct mix of injected
quality problems so the DataGuard AI engine surfaces different issue types
for each one.

Run:  python3 generate_extra_datasets.py
"""
from __future__ import annotations

import os
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(OUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. HR Employee Records
#    Quality problems injected:
#      • salary outliers (a whole department got 100× salary due to unit bug)
#      • ~12% missing performance_score (new KPI, not back-filled)
#      • ~5% missing manager_id (top-level employees + ETL gap)
#      • ~7% duplicate rows (HR system exported same batch twice)
#      • join_date partially stale (some records from 2019 never refreshed)
#      • phone_number column: mixed formats (with/without country code, dashes)
# ═════════════════════════════════════════════════════════════════════════════
def make_hr(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    N = 1500

    emp_ids = [f"EMP{str(i).zfill(6)}" for i in rng.integers(100000, 999999, N)]
    departments = rng.choice(
        ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Legal"],
        N, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05],
    )
    job_titles = {
        "Engineering": ["SDE I", "SDE II", "Senior SDE", "Staff Engineer", "EM"],
        "Sales":        ["SDR", "AE", "Senior AE", "Sales Manager", "VP Sales"],
        "Marketing":    ["Analyst", "Specialist", "Manager", "Director"],
        "Finance":      ["Analyst", "Senior Analyst", "Manager", "CFO"],
        "HR":           ["HRBP", "Recruiter", "HR Manager"],
        "Operations":   ["Ops Analyst", "Ops Lead", "COO"],
        "Legal":        ["Counsel", "Senior Counsel", "General Counsel"],
    }
    titles = [random.choice(job_titles[d]) for d in departments]

    base = datetime(2018, 6, 1)
    join_dates = [base + timedelta(days=int(d)) for d in rng.integers(0, 2000, N)]

    # Salary: ₹4L–₹35L per annum (stored as monthly)
    salary_monthly = rng.integers(33_000, 292_000, N).astype(float)

    # Inject salary outlier: entire "Finance" dept in one batch got 100× salary
    finance_idx = np.where(departments == "Finance")[0]
    bad_finance = rng.choice(finance_idx, size=min(30, len(finance_idx)), replace=False)
    salary_monthly[bad_finance] *= 100   # unit bug: annual stored as monthly×100

    # Gender
    genders = rng.choice(["Male", "Female", "Non-binary", "Prefer not to say"], N,
                          p=[0.48, 0.44, 0.04, 0.04])

    # Location
    cities = rng.choice(
        ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata"],
        N, p=[0.28, 0.22, 0.18, 0.12, 0.10, 0.06, 0.04],
    )

    # Performance score 1–5 (12% missing — not back-filled for old employees)
    perf = rng.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.10, 0.35, 0.35, 0.15]).astype(float)

    # Manager ID (5% missing for top-level or ETL gap)
    manager_ids = [f"EMP{str(i).zfill(6)}" for i in rng.integers(100000, 999999, N)]

    # Phone: mixed formats
    def phone(i: int) -> str:
        num = f"{rng.integers(6000000000, 9999999999)}"
        style = i % 4
        if style == 0: return f"+91-{num[:5]}-{num[5:]}"
        if style == 1: return num
        if style == 2: return f"91{num}"
        return f"({num[:3]}) {num[3:8]}-{num[8:]}"

    phones = [phone(i) for i in range(N)]

    df = pd.DataFrame({
        "employee_id":       emp_ids,
        "department":        list(departments),
        "job_title":         titles,
        "join_date":         [d.strftime("%Y-%m-%d") for d in join_dates],
        "city":              list(cities),
        "gender":            list(genders),
        "salary_monthly_inr": salary_monthly,
        "performance_score": perf,
        "manager_id":        manager_ids,
        "phone_number":      phones,
        "is_active":         rng.choice([True, False], N, p=[0.88, 0.12]),
    })

    # Missing values
    df.loc[rng.choice(N, int(N * 0.12), replace=False), "performance_score"] = np.nan
    df.loc[rng.choice(N, int(N * 0.05), replace=False), "manager_id"] = np.nan
    df.loc[rng.choice(N, int(N * 0.03), replace=False), "city"] = np.nan

    # Duplicate rows (7%)
    dup = df.sample(int(N * 0.07), random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    path = os.path.join(OUT_DIR, "hr_employee_records.csv")
    df.to_csv(path, index=False)
    print(f"✅ hr_employee_records.csv — {len(df):,} rows")
    print(f"   Salary outliers (Finance ×100): {(df['salary_monthly_inr'] > 5_000_000).sum()}")
    print(f"   Missing performance_score: {df['performance_score'].isnull().sum()}")
    print(f"   Duplicates: {df.duplicated().sum()}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Hospital Patient Admissions
#    Quality problems injected:
#      • age outliers: 15 records with age > 130 (data-entry error: DOB typo)
#      • length_of_stay outliers: a handful of negative values (discharge < admit)
#      • ~9% missing diagnosis_code (emergency admissions, coded later)
#      • ~6% missing discharge_date (still admitted OR coding backlog)
#      • ward column: inconsistent casing/spelling ("ICU", "icu", "I.C.U")
#      • blood_type: 4% invalid entries ("AB++" typos)
#      • completely empty column: insurance_claim_id (new field, not populated)
# ═════════════════════════════════════════════════════════════════════════════
def make_hospital(seed: int = 13) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    N = 1800

    patient_ids = [f"PAT{str(i).zfill(7)}" for i in rng.integers(1000000, 9999999, N)]

    base = datetime(2023, 1, 1)
    admit_dates = [base + timedelta(days=int(d)) for d in rng.integers(0, 540, N)]
    los = rng.integers(1, 30, N).astype(float)   # length of stay in days
    discharge_dates = [
        admit_dates[i] + timedelta(days=int(los[i])) for i in range(N)
    ]

    # Inject negative LoS: discharge before admission (10 records)
    neg_idx = rng.choice(N, 10, replace=False)
    los[neg_idx] = rng.integers(-5, 0, 10).astype(float)

    ages = rng.integers(1, 90, N).astype(float)
    # Age outliers: 15 records with age > 130
    age_outlier_idx = rng.choice(N, 15, replace=False)
    ages[age_outlier_idx] = rng.integers(131, 200, 15).astype(float)

    genders = rng.choice(["M", "F", "Other"], N, p=[0.48, 0.48, 0.04])

    wards_clean = rng.choice(
        ["ICU", "General", "Maternity", "Cardiology", "Orthopaedics", "Neurology"],
        N, p=[0.12, 0.38, 0.12, 0.15, 0.13, 0.10],
    )
    # Inconsistent casing/spelling for ICU
    ward_variants = {"ICU": ["ICU", "icu", "I.C.U", "Icu", "ICu"]}
    wards = []
    for w in wards_clean:
        if w == "ICU" and random.random() < 0.55:
            wards.append(random.choice(ward_variants["ICU"]))
        else:
            wards.append(w)

    blood_types_clean = rng.choice(
        ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], N,
        p=[0.27, 0.06, 0.23, 0.06, 0.05, 0.01, 0.28, 0.04],
    )
    blood_types = list(blood_types_clean)
    invalid_bt_idx = rng.choice(N, int(N * 0.04), replace=False)
    for idx in invalid_bt_idx:
        blood_types[idx] = random.choice(["AB++", "O+-", "B++", "XY", "N/A"])

    icd_codes = [
        f"{''.join(random.choices(string.ascii_uppercase, k=1))}"
        f"{rng.integers(10, 99)}"
        f".{''.join(random.choices(string.digits, k=1))}"
        for _ in range(N)
    ]

    df = pd.DataFrame({
        "patient_id":          patient_ids,
        "admission_date":      [d.strftime("%Y-%m-%d") for d in admit_dates],
        "discharge_date":      [d.strftime("%Y-%m-%d") for d in discharge_dates],
        "age":                 ages,
        "gender":              list(genders),
        "ward":                wards,
        "blood_type":          blood_types,
        "diagnosis_code":      icd_codes,
        "length_of_stay_days": los,
        "readmitted_30d":      rng.choice([True, False], N, p=[0.09, 0.91]),
        "insurance_claim_id":  [np.nan] * N,   # completely empty — new field
    })

    # Missing values
    df.loc[rng.choice(N, int(N * 0.09), replace=False), "diagnosis_code"] = np.nan
    df.loc[rng.choice(N, int(N * 0.06), replace=False), "discharge_date"] = np.nan

    path = os.path.join(OUT_DIR, "hospital_patient_admissions.csv")
    df.to_csv(path, index=False)
    print(f"✅ hospital_patient_admissions.csv — {len(df):,} rows")
    print(f"   Age outliers (>130): {(df['age'] > 130).sum()}")
    print(f"   Negative LoS: {(df['length_of_stay_days'] < 0).sum()}")
    print(f"   Missing diagnosis_code: {df['diagnosis_code'].isnull().sum()}")
    print(f"   Invalid blood_type: {df['blood_type'].isin(['AB++','O+-','B++','XY','N/A']).sum()}")
    print(f"   Fully empty column (insurance_claim_id): {df['insurance_claim_id'].isnull().all()}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. E-Commerce Product Catalog
#    Quality problems injected:
#      • price_usd outliers: 40 products with price = 0.00 (missed during migration)
#      • stock_quantity outliers: negative stock values (returns > sales bug)
#      • ~15% missing description (scraped catalog, descriptions not fetched)
#      • ~8% missing category (uncategorised new listings)
#      • sku column: duplicate SKUs (same product listed twice by different sellers)
#      • rating: values outside 1–5 range (scraper returned raw API float 5.1–9.9)
#      • last_updated: 30% of rows are > 180 days stale (supplier feed lapsed)
# ═════════════════════════════════════════════════════════════════════════════
def make_ecommerce(seed: int = 21) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    N = 1200

    def rand_sku() -> str:
        return (
            "".join(random.choices(string.ascii_uppercase, k=3))
            + "-"
            + "".join(random.choices(string.digits, k=5))
        )

    skus = [rand_sku() for _ in range(N)]
    # Inject ~5% duplicate SKUs
    dup_sku_src = random.sample(range(N), int(N * 0.05))
    for idx in dup_sku_src:
        target = random.randint(0, N - 1)
        skus[target] = skus[idx]

    categories = rng.choice(
        ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Beauty", "Toys"],
        N, p=[0.22, 0.20, 0.18, 0.12, 0.12, 0.09, 0.07],
    )

    brands = rng.choice(
        ["AlphaGear", "BrightHome", "CoreTech", "DenimCo", "EasyLife",
         "FitPro", "GreenLeaf", "HomeBase", "Innovex", "JetStar"],
        N,
    )

    prices = rng.uniform(5.0, 500.0, N).round(2)
    # Price = 0.00 for 40 products (migration bug)
    zero_price_idx = rng.choice(N, 40, replace=False)
    prices[zero_price_idx] = 0.00

    stock = rng.integers(0, 500, N).astype(float)
    # Negative stock for 25 products (returns > sales bug)
    neg_stock_idx = rng.choice(N, 25, replace=False)
    stock[neg_stock_idx] = rng.integers(-200, -1, 25).astype(float)

    ratings = rng.uniform(1.0, 5.0, N).round(1)
    # Out-of-range ratings from scraper (6% of records)
    bad_rating_idx = rng.choice(N, int(N * 0.06), replace=False)
    ratings[bad_rating_idx] = rng.uniform(5.1, 9.9, len(bad_rating_idx)).round(1)

    review_counts = rng.integers(0, 5000, N)

    # last_updated: 70% fresh (within 60 days), 30% stale (180–600 days old)
    now = datetime(2025, 6, 1)
    last_updated = []
    for i in range(N):
        if rng.random() < 0.70:
            last_updated.append(now - timedelta(days=int(rng.integers(1, 60))))
        else:
            last_updated.append(now - timedelta(days=int(rng.integers(180, 600))))

    descriptions = [
        f"High quality {categories[i].lower()} product from {brands[i]}. "
        f"Trusted by thousands of customers."
        for i in range(N)
    ]

    df = pd.DataFrame({
        "sku":            skus,
        "brand":          list(brands),
        "category":       list(categories),
        "price_usd":      prices,
        "stock_quantity": stock,
        "rating":         ratings,
        "review_count":   review_counts,
        "description":    descriptions,
        "last_updated":   [d.strftime("%Y-%m-%d") for d in last_updated],
        "is_active":      rng.choice([True, False], N, p=[0.92, 0.08]),
    })

    # Missing values
    df.loc[rng.choice(N, int(N * 0.15), replace=False), "description"] = np.nan
    df.loc[rng.choice(N, int(N * 0.08), replace=False), "category"] = np.nan

    path = os.path.join(OUT_DIR, "ecommerce_product_catalog.csv")
    df.to_csv(path, index=False)
    print(f"✅ ecommerce_product_catalog.csv — {len(df):,} rows")
    print(f"   Zero-price products: {(df['price_usd'] == 0).sum()}")
    print(f"   Negative stock: {(df['stock_quantity'] < 0).sum()}")
    print(f"   Out-of-range ratings (>5): {(df['rating'] > 5).sum()}")
    print(f"   Missing description: {df['description'].isnull().sum()}")
    print(f"   Duplicate SKUs: {df.duplicated(subset=['sku']).sum()}")
    stale = (pd.to_datetime(df['last_updated']) < (datetime(2025, 6, 1) - timedelta(days=180))).sum()
    print(f"   Stale last_updated (>180d): {stale}")


# ═════════════════════════════════════════════════════════════════════════════
# Run all generators
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating datasets...\n")
    make_hr()
    print()
    make_hospital()
    print()
    make_ecommerce()
    print(f"\nAll datasets saved to: {OUT_DIR}/")
