"""
Generate four realistic synthetic datasets for DataGuard AI demo.
Each dataset intentionally produces a different quality grade so the
dashboard showcases the full scoring range.

  customer_transactions.csv  → Grade C / D  (revenue outliers + heavy missing)
  hr_employee_records.csv    → Grade B      (salary unit bug + format issues)
  hospital_patient_admissions.csv → Grade D (age outliers + negative LOS + empty col)
  ecommerce_product_catalog.csv   → Grade B (zero prices + bad ratings + stale)
"""
from __future__ import annotations

import os
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
rng  = np.random.default_rng(SEED)
random.seed(SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. customer_transactions.csv  → target Grade C (score ~55–69)
# ═══════════════════════════════════════════════════════════════════════════════
def make_transactions():
    N = 2000
    base = datetime(2024, 1, 1)
    dates = [base + timedelta(days=int(d)) for d in rng.integers(0, 450, N)]

    customer_ids = [f"CUST{rng.integers(1000,9999):05d}" for _ in range(N)]
    regions  = rng.choice(["North","South","East","West","Central"], N,
                           p=[0.25,0.20,0.20,0.20,0.15])
    products = rng.choice(
        ["Widget A","Widget B","Gadget Pro","Service Pack","Premium Bundle"],
        N, p=[0.30,0.25,0.20,0.15,0.10])
    channels = rng.choice(["Online","Retail","Partner","Direct"], N)

    # Normal revenue ₹500–₹50,000
    revenue = rng.integers(500, 50_000, N).astype(float)

    # Inject 220 outliers (11% of rows) clustered in a 2-day ingestion window
    outlier_idx = rng.choice(N, 220, replace=False)
    revenue[outlier_idx] = rng.integers(500_001, 3_000_000, 220).astype(float)
    bad_start = datetime(2024, 3, 12)
    for i in outlier_idx[:150]:
        dates[i] = bad_start + timedelta(hours=int(rng.integers(0, 48)))

    domains = ["gmail.com","yahoo.com","company.in","outlook.com"]
    emails  = [f"user{random.randint(1000,9999)}@{random.choice(domains)}" for _ in range(N)]

    # mixed-type order_ref (12% strings)
    order_refs: list = [int(rng.integers(100_000, 999_999)) for _ in range(N)]
    for i in rng.choice(N, int(N * 0.12), replace=False):
        order_refs[i] = f"ORD-{''.join(random.choices(string.ascii_uppercase, k=6))}"

    df = pd.DataFrame({
        "transaction_date": [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates],
        "customer_id": customer_ids,
        "email": emails,
        "region": list(regions),
        "product": list(products),
        "channel": list(channels),
        "revenue_inr": revenue,
        "quantity": rng.integers(1, 50, N),
        "discount_pct": rng.choice([0, 5, 10, 15, 20], N),
        "order_ref": order_refs,
        "is_returned": rng.choice([True, False], N, p=[0.08, 0.92]),
    })

    # Missing values — aggressive
    df.loc[rng.choice(N, int(N * 0.18), replace=False), "email"]   = np.nan   # 18%
    df.loc[rng.choice(N, int(N * 0.12), replace=False), "region"]  = np.nan   # 12%
    df.loc[rng.choice(N, int(N * 0.09), replace=False), "product"] = np.nan   # 9%
    df.loc[rng.choice(N, int(N * 0.06), replace=False), "channel"] = np.nan   # 6%

    # Duplicates — 8%
    dup = df.sample(int(N * 0.08), random_state=SEED)
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    path = os.path.join(OUT_DIR, "customer_transactions.csv")
    df.to_csv(path, index=False)
    print(f"customer_transactions.csv  → {len(df):,} rows  "
          f"outliers={int((df['revenue_inr']>500_000).sum())}  "
          f"dupes={df.duplicated().sum()}  "
          f"missing_email={df['email'].isna().sum()}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 2. hr_employee_records.csv  → target Grade B (score ~70–84)
# ═══════════════════════════════════════════════════════════════════════════════
def make_hr():
    N = 1500
    # Recent data — last updated within the past 20 days
    base = datetime.now() - timedelta(days=10)
    join_dates = [base - timedelta(days=int(d)) for d in rng.integers(30, 3650, N)]

    depts = rng.choice(["Engineering","Sales","HR","Finance","Marketing","Operations"], N,
                        p=[0.25,0.20,0.15,0.15,0.15,0.10])
    roles = {
        "Engineering": ["SDE I","SDE II","Senior SDE","Staff SDE"],
        "Sales": ["SDR","Account Executive","Sales Manager","VP Sales"],
        "HR": ["HR Analyst","HR Manager","HRBP","VP HR"],
        "Finance": ["Analyst","Senior Analyst","Manager","Director"],
        "Marketing": ["Marketing Analyst","Campaign Manager","CMO"],
        "Operations": ["Ops Analyst","Ops Manager","COO"],
    }
    designations = [random.choice(roles[d]) for d in depts]

    # Normal salary ₹4L–₹50L per annum
    salaries = rng.integers(400_000, 5_000_000, N).astype(float)
    # Salary unit bug: 120 rows accidentally stored in monthly (divide by 12)
    bug_idx = rng.choice(N, 120, replace=False)
    salaries[bug_idx] = salaries[bug_idx] / 12   # looks like monthly salary

    ages = rng.integers(22, 58, N)
    performance_scores = rng.uniform(2.0, 5.0, N)

    phones = [f"+91-{rng.integers(7000000000, 9999999999)}" for _ in range(N)]
    # 8% wrong phone format
    for i in rng.choice(N, int(N * 0.08), replace=False):
        phones[i] = f"{''.join(str(rng.integers(0,9)) for _ in range(8))}"

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:05d}" for i in range(1, N+1)],
        "join_date": [d.strftime("%Y-%m-%d") for d in join_dates],
        "department": list(depts),
        "designation": designations,
        "age": ages,
        "salary_inr": salaries,
        "performance_score": performance_scores,
        "phone": phones,
        "is_active": rng.choice([True, False], N, p=[0.88, 0.12]),
        "manager_id": [f"EMP{rng.integers(1, 200):05d}" for _ in range(N)],
    })

    # Missing values — moderate
    df.loc[rng.choice(N, int(N * 0.11), replace=False), "performance_score"] = np.nan  # 11%
    df.loc[rng.choice(N, int(N * 0.07), replace=False), "manager_id"]        = np.nan  # 7%
    df.loc[rng.choice(N, int(N * 0.04), replace=False), "phone"]             = np.nan  # 4%

    # Duplicates — 4%
    dup = df.sample(int(N * 0.04), random_state=SEED)
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    path = os.path.join(OUT_DIR, "hr_employee_records.csv")
    df.to_csv(path, index=False)
    print(f"hr_employee_records.csv    → {len(df):,} rows  "
          f"salary_bug={len(bug_idx)}  "
          f"missing_perf={df['performance_score'].isna().sum()}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 3. hospital_patient_admissions.csv  → target Grade D (score ~40–54)
# ═══════════════════════════════════════════════════════════════════════════════
def make_hospital():
    N = 1800
    # Very stale — 2+ years old
    base = datetime(2022, 1, 1)
    admit_dates = [base + timedelta(days=int(d)) for d in rng.integers(0, 730, N)]

    departments = rng.choice(
        ["Cardiology","Orthopedics","General","Neurology","Oncology","Pediatrics"],
        N, p=[0.20,0.18,0.22,0.15,0.12,0.13])
    diagnoses = rng.choice(
        ["Hypertension","Fracture","Fever","Stroke","Cancer","Bronchitis","Diabetes"],
        N, p=[0.20,0.15,0.20,0.10,0.10,0.15,0.10])

    ages = rng.integers(18, 85, N).astype(float)
    # Age outliers: 120 records with impossible age (>120 or <0) — ~6.7%
    bad_age_idx = rng.choice(N, 120, replace=False)
    ages[bad_age_idx] = rng.choice([-5, 130, 145, 200, -10, 999], 120)

    los = rng.integers(1, 30, N).astype(float)  # length of stay (days)
    # Negative LOS: 150 records (data-entry error) — ~8.3%
    neg_los_idx = rng.choice(N, 150, replace=False)
    los[neg_los_idx] = rng.integers(-20, 0, 150).astype(float)

    blood_types = rng.choice(["A+","A-","B+","B-","O+","O-","AB+","AB-"], N)

    df = pd.DataFrame({
        "patient_id": [f"PAT{rng.integers(10000,99999):05d}" for _ in range(N)],
        "admit_date": [d.strftime("%Y-%m-%d") for d in admit_dates],
        "department": list(departments),
        "diagnosis": list(diagnoses),
        "age": ages,
        "length_of_stay": los,
        "blood_type": list(blood_types),
        "insurance_provider": [np.nan] * N,          # entire column empty
        "attending_doctor": [f"DR{rng.integers(100,999)}" for _ in range(N)],
        "ward": rng.choice(["General","ICU","Private","Semi-Private"], N),
    })

    # Missing values — very heavy
    df.loc[rng.choice(N, int(N * 0.28), replace=False), "diagnosis"]        = np.nan  # 28%
    df.loc[rng.choice(N, int(N * 0.20), replace=False), "blood_type"]       = np.nan  # 20%
    df.loc[rng.choice(N, int(N * 0.13), replace=False), "attending_doctor"] = np.nan  # 13%
    df.loc[rng.choice(N, int(N * 0.10), replace=False), "department"]       = np.nan  # 10%

    # Duplicates — 10%
    dup = df.sample(int(N * 0.10), random_state=SEED)
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    path = os.path.join(OUT_DIR, "hospital_patient_admissions.csv")
    df.to_csv(path, index=False)
    print(f"hospital_patient_admissions.csv → {len(df):,} rows  "
          f"age_outliers={len(bad_age_idx)}  neg_los={len(neg_los_idx)}  "
          f"empty_insurance=100%")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ecommerce_product_catalog.csv  → target Grade B (score ~70–84)
# ═══════════════════════════════════════════════════════════════════════════════
def make_ecommerce():
    N = 1200
    # Recent catalog — updated within last 15 days
    base = datetime.now() - timedelta(days=7)
    update_dates = [base - timedelta(days=int(d)) for d in rng.integers(0, 14, N)]

    categories = rng.choice(
        ["Electronics","Clothing","Home & Kitchen","Sports","Books","Toys"],
        N, p=[0.22,0.20,0.18,0.15,0.15,0.10])
    brands = rng.choice(
        ["BrandA","BrandB","BrandC","NoName","OffBrand","Premium"],
        N, p=[0.22,0.18,0.15,0.20,0.15,0.10])

    prices = rng.integers(100, 50_000, N).astype(float)
    # Zero-price products (3.5%) — listing error
    zero_idx = rng.choice(N, int(N * 0.035), replace=False)
    prices[zero_idx] = 0.0

    stock = rng.integers(0, 500, N).astype(int)
    # Negative stock (4%) — inventory sync bug
    neg_stock_idx = rng.choice(N, int(N * 0.04), replace=False)
    stock[neg_stock_idx] = rng.integers(-100, -1, len(neg_stock_idx))

    ratings = rng.uniform(1.0, 5.0, N)
    # Out-of-range ratings (2.5%) — scraped from bad source
    bad_rating_idx = rng.choice(N, int(N * 0.025), replace=False)
    ratings[bad_rating_idx] = rng.uniform(5.5, 10.0, len(bad_rating_idx))

    df = pd.DataFrame({
        "product_id": [f"PROD{rng.integers(10000,99999):05d}" for _ in range(N)],
        "last_updated": [d.strftime("%Y-%m-%d") for d in update_dates],
        "category": list(categories),
        "brand": list(brands),
        "price_inr": prices,
        "stock_quantity": stock,
        "rating": ratings,
        "review_count": rng.integers(0, 5000, N),
        "is_active": rng.choice([True, False], N, p=[0.90, 0.10]),
        "sku": [f"SKU-{rng.integers(100000,999999)}" for _ in range(N)],
    })

    # Missing values — moderate
    df.loc[rng.choice(N, int(N * 0.08), replace=False), "brand"]        = np.nan  # 8%
    df.loc[rng.choice(N, int(N * 0.05), replace=False), "review_count"] = np.nan  # 5%
    df.loc[rng.choice(N, int(N * 0.03), replace=False), "rating"]       = np.nan  # 3%

    # Duplicates — 3%
    dup = df.sample(int(N * 0.03), random_state=SEED)
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    path = os.path.join(OUT_DIR, "ecommerce_product_catalog.csv")
    df.to_csv(path, index=False)
    print(f"ecommerce_product_catalog.csv   → {len(df):,} rows  "
          f"zero_price={len(zero_idx)}  neg_stock={len(neg_stock_idx)}  "
          f"bad_rating={len(bad_rating_idx)}")
    return path


if __name__ == "__main__":
    print("Generating DataGuard AI sample datasets…\n")
    make_transactions()
    make_hr()
    make_hospital()
    make_ecommerce()
    print("\nDone. Files saved to sample_data/")
