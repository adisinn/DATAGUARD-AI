"""
DataGuard AI — Business Impact Estimator
Translates abstract data-quality issues into concrete business consequences:
  - Monetary exposure (if a numeric/revenue column is affected)
  - Record-level impact (how many decisions / reports could be wrong)
  - Downstream risk label

The estimator is heuristic-driven and works purely from the dataset and the
root-cause report — no external configuration required.  When an obvious
"value" column (revenue, amount, price, sales, cost, …) is present the
estimator quantifies monetary risk; otherwise it falls back to record counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dataguard_ai.ai.root_cause import IssueInsight, RootCauseReport
from dataguard_ai.core.profiler import ProfileResult


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImpactEstimate:
    issue_title: str
    dimension: str
    severity: str
    affected_rows: int
    affected_pct: float

    # monetary impact (None when no value column found)
    monetary_impact: float | None
    currency_symbol: str
    monetary_note: str          # how the figure was derived

    # record impact
    record_impact_description: str

    # downstream risk
    downstream_risks: list[str]
    risk_level: str             # HIGH / MEDIUM / LOW


@dataclass
class BusinessImpactReport:
    estimates: list[ImpactEstimate]
    total_monetary_exposure: float | None
    currency_symbol: str
    total_affected_rows: int
    unique_affected_pct: float

    # overall risk narrative
    executive_summary: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VALUE_KEYWORDS = [
    "revenue", "amount", "price", "sales", "cost", "value", "total",
    "payment", "transaction", "spend", "income", "earning", "fee",
    "charge", "invoice", "balance", "profit", "loss",
]

_CURRENCY_HINTS = {
    "inr": "₹", "rupee": "₹", "rs": "₹",
    "usd": "$", "dollar": "$",
    "eur": "€", "euro": "€",
    "gbp": "£", "pound": "£",
}


def _detect_value_column(df: pd.DataFrame) -> tuple[str | None, str]:
    """
    Return (column_name, currency_symbol) for the most likely monetary column.
    Falls back to (None, "$") if nothing convincing found.
    """
    candidates: list[tuple[str, float]] = []
    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        # currency hint in column name?
        currency = "$"
        for hint, sym in _CURRENCY_HINTS.items():
            if hint in col_lower:
                currency = sym
                break
        # keyword match score
        score = sum(kw in col_lower for kw in _VALUE_KEYWORDS)
        if score == 0:
            continue
        # must be numeric-ish
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() < 0.7:
            continue
        # prefer columns with larger absolute values (likely to be monetary)
        median_val = float(numeric.median())
        candidates.append((col, score * 10 + min(median_val / 1000, 50)))

    if not candidates:
        return None, "$"

    best_col = max(candidates, key=lambda x: x[1])[0]
    # re-detect currency symbol
    col_lower = best_col.lower()
    currency = "$"
    for hint, sym in _CURRENCY_HINTS.items():
        if hint in col_lower:
            currency = sym
            break
    return best_col, currency


def _estimate_monetary(
    df: pd.DataFrame,
    value_col: str,
    affected_rows: int,
    issue: IssueInsight,
) -> tuple[float, str]:
    """
    Return (monetary_exposure, explanation_note).
    Strategy:
      - For outlier issues: sum of outlier values − expected range midpoint × count
      - For missing values: affected_rows × mean value (lost/uncertain revenue)
      - For duplicates: affected_rows × mean value (over-counted revenue)
      - For others: affected_rows × mean value
    """
    numeric = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if numeric.empty:
        return 0.0, "No numeric data in value column."

    mean_val = float(numeric.mean())

    if issue.dimension == "validity":
        # Outlier monetary impact = sum of outlier values (they distort aggregates)
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        outlier_vals = numeric[(numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)]
        if not outlier_vals.empty:
            expected = float(numeric.median()) * len(outlier_vals)
            exposure = abs(float(outlier_vals.sum()) - expected)
            note = (
                f"Difference between sum of anomalous {value_col} values "
                f"and their expected value at median = exposure."
            )
            return round(exposure, 2), note

    if issue.dimension == "uniqueness":
        exposure = affected_rows * mean_val
        return round(exposure, 2), (
            f"Potential double-counted {value_col}: "
            f"{affected_rows} duplicate rows × mean {value_col}."
        )

    if issue.dimension == "completeness":
        exposure = affected_rows * mean_val
        return round(exposure, 2), (
            f"Revenue/value at risk from missing records: "
            f"{affected_rows} rows × mean {value_col}."
        )

    # generic
    exposure = affected_rows * mean_val * 0.1   # conservative 10 % effect
    return round(exposure, 2), (
        f"Conservative estimate: 10% of affected row value in '{value_col}'."
    )


def _downstream_risks(issue: IssueInsight) -> list[str]:
    risks: dict[str, list[str]] = {
        "completeness": [
            "KPI dashboards showing incomplete aggregates (under-reported totals).",
            "ML model training on incomplete feature sets → biased predictions.",
            "Segment-level reports missing entire customer groups.",
        ],
        "validity": [
            "Revenue/metric dashboards inflated or deflated by anomalous values.",
            "Statistical models trained on outlier-contaminated data → poor generalization.",
            "Anomaly detection baselines skewed by existing anomalies.",
        ],
        "consistency": [
            "Incorrect joins or lookups when ID/key columns have format mismatches.",
            "Natural language processing or embedding models confused by inconsistent text.",
        ],
        "uniqueness": [
            "Aggregate metrics (SUM, COUNT) over-counted due to duplicates.",
            "Customer/entity counts inflated, distorting segmentation analyses.",
        ],
        "schema": [
            "Pipeline failures when downstream consumers expect a strict schema.",
            "Silent type coercion causing numeric columns to be treated as text.",
        ],
        "freshness": [
            "Decisions based on outdated trends — potential to miss recent business shifts.",
            "Time-series forecasts anchored to stale data may produce wrong projections.",
        ],
    }
    return risks.get(issue.dimension, ["Undefined downstream risk."])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def estimate(
    df: pd.DataFrame,
    profile: ProfileResult,
    root_cause_report: RootCauseReport,
) -> BusinessImpactReport:
    """
    Produce a BusinessImpactReport from the dataset, its profile and the
    root-cause findings.
    """
    value_col, currency = _detect_value_column(df)
    estimates: list[ImpactEstimate] = []
    total_monetary: float = 0.0
    affected_row_sets: set[int] = set()

    for issue in root_cause_report.issues:
        # Monetary impact
        if value_col and issue.affected_rows > 0:
            monetary, note = _estimate_monetary(df, value_col, issue.affected_rows, issue)
            total_monetary += monetary
        else:
            monetary = None
            note = "No monetary value column identified in dataset."

        # Record impact description
        pct_str = f"{issue.affected_pct:.1f}%"
        record_desc = (
            f"{issue.affected_rows:,} records ({pct_str} of dataset) are affected. "
            f"Any report, dashboard, or model consuming this column may produce incorrect results."
        )

        # Downstream risks
        risks = _downstream_risks(issue)

        est = ImpactEstimate(
            issue_title=issue.title,
            dimension=issue.dimension,
            severity=issue.severity,
            affected_rows=issue.affected_rows,
            affected_pct=issue.affected_pct,
            monetary_impact=monetary,
            currency_symbol=currency,
            monetary_note=note,
            record_impact_description=record_desc,
            downstream_risks=risks,
            risk_level=issue.severity,
        )
        estimates.append(est)

    # Total affected rows (unique across issues where col != __dataset__)
    total_aff = sum(
        e.affected_rows for e in estimates if e.affected_rows > 0
    )
    total_rows = max(profile.total_rows, 1)
    unique_aff_pct = round(min(total_aff / total_rows * 100, 100), 1)

    # Executive summary
    if not estimates:
        exec_summary = "No business impact identified. Data appears reliable."
    else:
        high_count = sum(1 for e in estimates if e.severity == "HIGH")
        mon_str = (
            f"{currency}{total_monetary:,.0f} in potential monetary exposure"
            if total_monetary > 0 and value_col
            else "monetary exposure could not be quantified (no value column found)"
        )
        exec_summary = (
            f"{len(estimates)} quality issue(s) could affect business outcomes. "
            f"{high_count} are HIGH severity. "
            f"Estimated {mon_str}. "
            f"Approximately {unique_aff_pct}% of dataset rows are implicated across all issues."
        )

    return BusinessImpactReport(
        estimates=estimates,
        total_monetary_exposure=round(total_monetary, 2) if value_col else None,
        currency_symbol=currency,
        total_affected_rows=total_aff,
        unique_affected_pct=unique_aff_pct,
        executive_summary=exec_summary,
    )
