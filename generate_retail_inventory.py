"""
Generate retail_inventory_sales.csv for DataGuard AI demo.

Domain: SKU-level retail inventory & sales — stock levels, sell-through rates,
reorder points, supplier lead times, warehouse locations.

Deliberately injected quality problems (targeting Grade C, score ~60–70):

  COMPLETENESS
  - supplier_contact   → 19% missing  (supplier records not always captured)
  - reorder_point      → 14% missing  (SKUs added before ops team set thresholds)
  - last_audit_date    → 11% missing  (audits skipped for low-priority SKUs)
  - sell_through_rate  → 7%  missing

  VALIDITY / OUTLIERS
  - stock_on_hand      → negative values (returns mis-booked, ~5% of SKUs)
  - lead_time_days     → extreme outliers (200–900 days; data-entry keyed wrong unit)
  - unit_cost_inr      → 0-price ghost SKUs + extreme high outliers (~8%)
  - sell_through_rate  → values > 1.0 (>100%, scraped from wrong column, ~4%)
  - reorder_point      → fractional values mixed with integers (unit confusion)

  CONSISTENCY / SCHEMA
  - lead_time_days     → mixed type: mostly int, some strings like "~3 weeks"
  - warehouse_zone     → inconsistent casing  ("Zone A" / "zone a" / "ZONE A")

  UNIQUENESS
  - 9% duplicate rows  (daily snapshot merge without dedup)

  FRESHNESS
  - last_sale_date     → 650+ days stale (legacy catalog, no recent sales)
"""
from __future__ import annotations

import os
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 7
rng  = np.random.default_rng(SEED)
random.seed(SEED)

N       = 2500
OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── SKU catalogue ─────────────────────────────────────────────────────────────
categories = rng.choice(
    ["Electronics", "Apparel", "Home & Kitchen", "Sports & Outdoors",
     "Beauty", "Toys", "Automotive", "Stationery"],
    N, p=[0.18, 0.20, 0.17, 0.12, 0.11, 0.10, 0.07, 0.05],
)
brands = rng.choice(
    ["AlphaEdge", "ZenCraft", "NovaLine", "SwiftBrand",
     "CorePeak", "UrbanWave", "TrueForm", "GenericPlus"],
    N, p=[0.16, 0.14, 0.13, 0.13, 0.12, 0.12, 0.11, 0.09],
)
skus = [
    f"{cat[:3].upper()}-{rng.integers(10000,99999)}"
    for cat in categories
]

# ── Warehouse / supplier ──────────────────────────────────────────────────────
# Inconsistent zone casing — injected deliberately
_zones_clean = ["Zone A", "Zone B", "Zone C", "Zone D"]
_zones_dirty = ["zone a", "zone b", "zone c", "zone d",
                "ZONE A", "ZONE B", "ZONE C", "ZONE D"]
warehouse_zones = []
for _ in range(N):
    if rng.random() < 0.30:          # 30% have bad casing
        warehouse_zones.append(random.choice(_zones_dirty))
    else:
        warehouse_zones.append(random.choice(_zones_clean))

suppliers = rng.choice(
    ["SupplyCo India", "GlobalTrade Ltd", "FastShip Pvt", "LocalSource",
     "PrimeParts", "QuickStock", "DirectImport"],
    N,
)
supplier_contacts = [
    f"contact{rng.integers(100,999)}@{s.lower().replace(' ','')}.com"
    for s in suppliers
]

# ── Dates ─────────────────────────────────────────────────────────────────────
# last_sale_date — stale: between 600–900 days ago
stale_base = datetime.now() - timedelta(days=650)
last_sale_dates = [
    stale_base - timedelta(days=int(d))
    for d in rng.integers(0, 250, N)
]
# last_audit_date — more recent but still old-ish
audit_base = datetime.now() - timedelta(days=180)
last_audit_dates = [
    audit_base - timedelta(days=int(d))
    for d in rng.integers(0, 365, N)
]
# restock_date — varies
restock_dates = [
    datetime.now() - timedelta(days=int(d))
    for d in rng.integers(0, 120, N)
]

# ── Numeric columns ───────────────────────────────────────────────────────────

# stock_on_hand: normally 0–500, negative for ~5% (returns mis-booking)
stock_on_hand = rng.integers(0, 500, N).astype(float)
neg_stock_idx = rng.choice(N, int(N * 0.05), replace=False)
stock_on_hand[neg_stock_idx] = rng.integers(-80, -1, len(neg_stock_idx))

# units_sold_30d
units_sold = rng.integers(0, 300, N).astype(int)

# sell_through_rate: 0.0–1.0 normally; >1.0 for 4% (scraped from wrong column)
sell_through = rng.uniform(0.0, 1.0, N)
bad_str_idx = rng.choice(N, int(N * 0.04), replace=False)
sell_through[bad_str_idx] = rng.uniform(1.05, 3.5, len(bad_str_idx))

# unit_cost_inr: normally ₹50–₹8,000
unit_cost = rng.integers(50, 8_000, N).astype(float)
# Ghost SKUs — 0 price (3%)
zero_cost_idx = rng.choice(N, int(N * 0.03), replace=False)
unit_cost[zero_cost_idx] = 0.0
# Extreme cost outliers — keyed in paise instead of rupees (~5%)
high_cost_idx = rng.choice(
    [i for i in range(N) if i not in zero_cost_idx.tolist()],
    int(N * 0.05), replace=False,
)
unit_cost[high_cost_idx] = rng.integers(800_000, 4_000_000, len(high_cost_idx))

# mrp_inr: always >= unit_cost normally; occasionally < unit_cost (margin error)
mrp = unit_cost * rng.uniform(1.15, 3.5, N)
# 3% have mrp < unit_cost (negative margin flag)
bad_mrp_idx = rng.choice(N, int(N * 0.03), replace=False)
mrp[bad_mrp_idx] = unit_cost[bad_mrp_idx] * rng.uniform(0.5, 0.95, len(bad_mrp_idx))

# reorder_point: integer normally; some fractional due to unit confusion
reorder_point = rng.integers(10, 200, N).astype(float)
frac_idx = rng.choice(N, int(N * 0.06), replace=False)
reorder_point[frac_idx] = rng.uniform(0.1, 0.9, len(frac_idx))  # <1 = fractional

# lead_time_days: normally 3–45 days
# mixed type: some stored as strings "~3 weeks", "2-3 weeks" etc.
lead_time_raw: list = [int(rng.integers(3, 45)) for _ in range(N)]
# Extreme numeric outliers: days entered as hours (~4%)
extreme_lt_idx = rng.choice(N, int(N * 0.04), replace=False)
for i in extreme_lt_idx:
    lead_time_raw[i] = int(rng.integers(200, 900))
# String values (~5%)
string_lt_idx = rng.choice(
    [i for i in range(N) if i not in extreme_lt_idx.tolist()],
    int(N * 0.05), replace=False,
)
_lt_strings = ["~3 weeks", "2-3 weeks", "TBD", "varies", "~1 month",
               "4-6 weeks", "express", "on demand"]
for i in string_lt_idx:
    lead_time_raw[i] = random.choice(_lt_strings)

# ── Assemble ──────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "sku":               skus,
    "category":          list(categories),
    "brand":             list(brands),
    "warehouse_zone":    warehouse_zones,
    "supplier":          list(suppliers),
    "supplier_contact":  supplier_contacts,
    "last_sale_date":    [d.strftime("%Y-%m-%d") for d in last_sale_dates],
    "last_audit_date":   [d.strftime("%Y-%m-%d") for d in last_audit_dates],
    "restock_date":      [d.strftime("%Y-%m-%d") for d in restock_dates],
    "stock_on_hand":     stock_on_hand,
    "units_sold_30d":    units_sold,
    "sell_through_rate": sell_through,
    "unit_cost_inr":     unit_cost,
    "mrp_inr":           mrp,
    "reorder_point":     reorder_point,
    "lead_time_days":    lead_time_raw,
    "is_active":         rng.choice([True, False], N, p=[0.88, 0.12]),
})

# ── Inject missing values ─────────────────────────────────────────────────────
df.loc[rng.choice(N, int(N * 0.19), replace=False), "supplier_contact"]  = np.nan
df.loc[rng.choice(N, int(N * 0.14), replace=False), "reorder_point"]     = np.nan
df.loc[rng.choice(N, int(N * 0.11), replace=False), "last_audit_date"]   = np.nan
df.loc[rng.choice(N, int(N * 0.07), replace=False), "sell_through_rate"] = np.nan

# ── Duplicate rows (9%) ───────────────────────────────────────────────────────
dup = df.sample(int(N * 0.09), random_state=SEED)
df  = pd.concat([df, dup], ignore_index=True)
df  = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
path = os.path.join(OUT_DIR, "retail_inventory_sales.csv")
df.to_csv(path, index=False)

total = len(df)
print(f"retail_inventory_sales.csv  →  {total:,} rows  ×  {len(df.columns)} columns")
print(f"  neg stock        : {(pd.to_numeric(df['stock_on_hand'], errors='coerce') < 0).sum()} rows")
print(f"  zero unit_cost   : {(df['unit_cost_inr'] == 0).sum()} rows")
print(f"  extreme lead_time: {sum(1 for v in df['lead_time_days'] if str(v).isdigit() and int(v) > 100)} rows")
print(f"  string lead_time : {sum(1 for v in df['lead_time_days'] if not str(v).replace('.','').isdigit())} rows")
print(f"  sell_through > 1 : {(pd.to_numeric(df['sell_through_rate'], errors='coerce') > 1.0).sum()} rows")
print(f"  bad mrp          : {(df['mrp_inr'] < df['unit_cost_inr']).sum()} rows")
print(f"  duplicates       : {df.duplicated().sum()} rows")
print(f"  missing supplier_contact : {df['supplier_contact'].isna().sum()}")
print(f"  missing reorder_point    : {df['reorder_point'].isna().sum()}")
print(f"  missing last_audit_date  : {df['last_audit_date'].isna().sum()}")
