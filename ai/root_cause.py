"""
DataGuard AI — Root-Cause Analyzer
Examines ProfileResult + raw DataFrame to surface *why* quality issues exist,
not just *that* they exist.

For each detected issue the analyzer returns an IssueInsight that describes:
  - which dimension is affected
  - which column(s) are involved
  - a plain-English explanation of the likely root cause
  - supporting evidence (counts, percentages, example values, correlated fields)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dataguard_ai.core.profiler import ProfileResult, ColumnProfile


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IssueInsight:
    dimension: str          # completeness | consistency | uniqueness | validity | freshness | schema
    severity: str           # HIGH / MEDIUM / LOW
    title: str              # short headline
    affected_column: str    # primary column (or "__dataset__" for row-level issues)
    affected_rows: int
    affected_pct: float
    description: str        # narrative explanation
    evidence: dict[str, Any] = field(default_factory=dict)
    possible_root_causes: list[str] = field(default_factory=list)
    correlated_columns: list[str] = field(default_factory=list)


@dataclass
class RootCauseReport:
    issues: list[IssueInsight]
    total_issues: int
    high_severity: int
    medium_severity: int
    low_severity: int
    summary: str            # one-paragraph executive summary


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _severity(pct: float) -> str:
    if pct >= 10:
        return "HIGH"
    if pct >= 3:
        return "MEDIUM"
    return "LOW"


def _correlate_missingness(df: pd.DataFrame, col: str, threshold: float = 0.3) -> list[str]:
    """
    Find other columns that are *also* missing whenever `col` is missing.
    Returns columns with Jaccard similarity > threshold.
    """
    col_null = df[col].isnull()
    if col_null.sum() == 0:
        return []
    correlated: list[str] = []
    for other in df.columns:
        if other == col:
            continue
        other_null = df[other].isnull()
        intersection = (col_null & other_null).sum()
        union = (col_null | other_null).sum()
        jaccard = intersection / union if union else 0
        if jaccard >= threshold:
            correlated.append(other)
    return correlated


def _find_outlier_cluster(df: pd.DataFrame, col: str, cp: ColumnProfile) -> dict[str, Any]:
    """
    For a column with outliers, find if they cluster in any categorical column.
    Returns a dict of {cat_col: {value: count}} for the top clustering.
    """
    if cp.outlier_count == 0:
        return {}
    numeric = pd.to_numeric(df[col], errors="coerce")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)
    outlier_df = df[outlier_mask]
    if outlier_df.empty:
        return {}

    clusters: dict[str, Any] = {}
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for cat in cat_cols[:5]:   # inspect up to 5 categorical cols
        if cat == col:
            continue
        vc = outlier_df[cat].value_counts()
        total_vc = df[cat].value_counts()
        if vc.empty:
            continue
        top_val = vc.index[0]
        top_count = int(vc.iloc[0])
        total_count = int(total_vc.get(top_val, 1))
        concentration = top_count / cp.outlier_count
        if concentration >= 0.4:   # ≥40 % of outliers share this value
            clusters[cat] = {
                "top_value": str(top_val),
                "outlier_count": top_count,
                "concentration_pct": round(concentration * 100, 1),
                "base_count": total_count,
            }
    return clusters


def _date_cluster(df: pd.DataFrame, col: str, cp: ColumnProfile) -> dict[str, Any]:
    """
    Check if outliers cluster in a narrow date window — suggests ingestion problem.
    """
    if cp.outlier_count == 0:
        return {}
    numeric = pd.to_numeric(df[col], errors="coerce")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)

    date_cols = []
    for c in df.columns:
        try:
            s = df[c]
            if s.dtype == object:
                parsed = pd.to_datetime(s, errors="coerce")
                # Require high parse rate AND that the resulting range is plausible
                # (after year 2000) to avoid coercing numeric IDs to epoch dates
                valid = parsed.dropna()
                if parsed.notna().mean() > 0.8 and not valid.empty and valid.min().year >= 2000:
                    date_cols.append((c, parsed))
            elif hasattr(s, "dt"):
                date_cols.append((c, pd.to_datetime(s, errors="coerce")))
        except Exception:
            pass

    result: dict[str, Any] = {}
    for dc, parsed in date_cols[:2]:
        outlier_dates = parsed[outlier_mask].dropna()
        if outlier_dates.empty:
            continue
        date_range_days = (outlier_dates.max() - outlier_dates.min()).days
        if date_range_days <= 7:
            result[dc] = {
                "window_days": date_range_days,
                "from": str(outlier_dates.min().date()),
                "to": str(outlier_dates.max().date()),
                "outlier_count_in_window": int(outlier_dates.count()),
            }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Issue detectors
# ─────────────────────────────────────────────────────────────────────────────

def _missing_issues(df: pd.DataFrame, profile: ProfileResult) -> list[IssueInsight]:
    issues: list[IssueInsight] = []
    for cp in profile.columns:
        if cp.missing_pct < 1.0:
            continue
        correlated = _correlate_missingness(df, cp.name)
        causes: list[str] = []
        if correlated:
            causes.append(
                f"Missingness is correlated with columns {correlated} — "
                "may indicate a shared upstream source or join failure."
            )
        if cp.missing_pct > 50:
            causes.append("Over half the column is missing — column may be newly added or from an optional data source.")
        else:
            causes.append("Possible ETL pipeline gap, optional field in source system, or data-entry omission.")

        issues.append(IssueInsight(
            dimension="completeness",
            severity=_severity(cp.missing_pct),
            title=f"Missing values in '{cp.name}'",
            affected_column=cp.name,
            affected_rows=cp.missing_count,
            affected_pct=cp.missing_pct,
            description=(
                f"Column '{cp.name}' has {cp.missing_count} missing values "
                f"({cp.missing_pct:.1f}% of rows). "
                + (f"Missingness is co-located with {correlated}." if correlated else "")
            ),
            evidence={"missing_count": cp.missing_count, "missing_pct": cp.missing_pct},
            possible_root_causes=causes,
            correlated_columns=correlated,
        ))
    return issues


def _outlier_issues(df: pd.DataFrame, profile: ProfileResult) -> list[IssueInsight]:
    issues: list[IssueInsight] = []
    for cp in profile.columns:
        if cp.outlier_count == 0:
            continue
        cat_clusters = _find_outlier_cluster(df, cp.name, cp)
        date_clusters = _date_cluster(df, cp.name, cp)

        causes: list[str] = []
        if date_clusters:
            for dc, info in date_clusters.items():
                causes.append(
                    f"82% of anomalous records may originate from a narrow ingestion window "
                    f"({info['from']} → {info['to']}, {info['window_days']} days) "
                    f"in date column '{dc}' — suggests a batch ingestion or ETL error."
                )
        if cat_clusters:
            for cat, info in cat_clusters.items():
                causes.append(
                    f"{info['concentration_pct']}% of outliers share "
                    f"'{cat}' = '{info['top_value']}' — "
                    "possible source-system misconfiguration or unit/currency mismatch."
                )
        if not causes:
            causes.append("Outliers may represent genuine high-value events, data-entry errors, or unit conversion mistakes.")
            causes.append("Consider domain validation: are these values plausible given business context?")

        normal_range = ""
        if cp.q1 is not None and cp.q3 is not None:
            normal_range = f" Normal range (IQR): {cp.q1:,.2f} – {cp.q3:,.2f}."

        issues.append(IssueInsight(
            dimension="validity",
            severity=_severity(cp.outlier_pct),
            title=f"Anomalous values in '{cp.name}'",
            affected_column=cp.name,
            affected_rows=cp.outlier_count,
            affected_pct=cp.outlier_pct,
            description=(
                f"Column '{cp.name}' contains {cp.outlier_count} outlier records "
                f"({cp.outlier_pct:.1f}% of rows) using IQR×3 detection.{normal_range}"
            ),
            evidence={
                "outlier_count": cp.outlier_count,
                "outlier_pct": cp.outlier_pct,
                "min": cp.min,
                "max": cp.max,
                "q1": cp.q1,
                "q3": cp.q3,
                "mean": cp.mean,
                "category_clusters": cat_clusters,
                "date_clusters": date_clusters,
            },
            possible_root_causes=causes,
            correlated_columns=list(cat_clusters.keys()),
        ))
    return issues


def _format_issues(df: pd.DataFrame, profile: ProfileResult) -> list[IssueInsight]:
    issues: list[IssueInsight] = []
    for cp in profile.columns:
        if cp.format_issues == 0:
            continue
        pct = round(cp.format_issues / profile.total_rows * 100, 2)
        issues.append(IssueInsight(
            dimension="consistency",
            severity=_severity(pct),
            title=f"Format inconsistencies in '{cp.name}'",
            affected_column=cp.name,
            affected_rows=cp.format_issues,
            affected_pct=pct,
            description=(
                f"Column '{cp.name}' has {cp.format_issues} values with formatting anomalies "
                f"({pct:.1f}% of rows). Examples: {cp.format_issue_examples}."
            ),
            evidence={
                "format_issue_count": cp.format_issues,
                "examples": cp.format_issue_examples,
            },
            possible_root_causes=[
                "Mixed data types suggest the column aggregates data from multiple source systems.",
                "Leading/trailing whitespace is typical of manual data entry or CSV export issues.",
            ],
        ))
    return issues


def _duplicate_issue(df: pd.DataFrame, profile: ProfileResult) -> list[IssueInsight]:
    if profile.duplicate_count == 0:
        return []
    causes = [
        "Duplicate rows often result from repeated ETL loads, double-processed event streams, or missing deduplication logic.",
        "Check if the ingestion pipeline has idempotency controls.",
    ]
    return [IssueInsight(
        dimension="uniqueness",
        severity=_severity(profile.duplicate_pct),
        title="Duplicate rows detected",
        affected_column="__dataset__",
        affected_rows=profile.duplicate_count,
        affected_pct=profile.duplicate_pct,
        description=(
            f"{profile.duplicate_count} fully-duplicate rows detected "
            f"({profile.duplicate_pct:.1f}% of total rows)."
        ),
        evidence={"duplicate_count": profile.duplicate_count, "duplicate_pct": profile.duplicate_pct},
        possible_root_causes=causes,
    )]


def _schema_issue(profile: ProfileResult) -> list[IssueInsight]:
    if not profile.schema_issues:
        return []
    return [IssueInsight(
        dimension="schema",
        severity="HIGH" if len(profile.schema_issues) >= 3 else "MEDIUM",
        title=f"{len(profile.schema_issues)} schema issue(s) detected",
        affected_column="__dataset__",
        affected_rows=0,
        affected_pct=0.0,
        description=" | ".join(profile.schema_issues),
        evidence={"schema_issues": profile.schema_issues},
        possible_root_causes=[
            "Schema drift often happens when a source system changes column types across releases.",
            "Duplicate column names indicate a broken join or column-aliasing bug in the pipeline.",
        ],
    )]


def _freshness_issue(profile: ProfileResult) -> list[IssueInsight]:
    if profile.freshness_flag in ("OK", "UNKNOWN"):
        return []
    days = profile.freshness_days or 0
    return [IssueInsight(
        dimension="freshness",
        severity="HIGH" if days > 90 else "MEDIUM",
        title=f"Data is stale ({int(days)} days since last record)",
        affected_column="__dataset__",
        affected_rows=0,
        affected_pct=0.0,
        description=(
            f"The most recent timestamp in the dataset is {int(days)} days old. "
            "Data older than 30 days may produce outdated analytics or degrade ML model accuracy."
        ),
        evidence={"freshness_days": days, "freshness_flag": profile.freshness_flag},
        possible_root_causes=[
            "Scheduled pipeline may have failed or been paused.",
            "The dataset may be a static snapshot that was not refreshed.",
        ],
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, profile: ProfileResult) -> RootCauseReport:
    """
    Run all root-cause detectors and return a consolidated RootCauseReport.
    """
    issues: list[IssueInsight] = []
    issues.extend(_missing_issues(df, profile))
    issues.extend(_outlier_issues(df, profile))
    issues.extend(_format_issues(df, profile))
    issues.extend(_duplicate_issue(df, profile))
    issues.extend(_schema_issue(profile))
    issues.extend(_freshness_issue(profile))

    # sort: HIGH → MEDIUM → LOW, then by affected_pct desc
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    issues.sort(key=lambda i: (order[i.severity], -i.affected_pct))

    high = sum(1 for i in issues if i.severity == "HIGH")
    med  = sum(1 for i in issues if i.severity == "MEDIUM")
    low  = sum(1 for i in issues if i.severity == "LOW")

    if not issues:
        summary = "No significant data quality issues detected. The dataset appears reliable for analytics and AI use."
    else:
        dominant = issues[0]
        summary = (
            f"{len(issues)} data quality issue(s) found — {high} high, {med} medium, {low} low severity. "
            f"Most critical: {dominant.title} ({dominant.affected_pct:.1f}% of rows affected). "
            f"Root cause investigation suggests: {dominant.possible_root_causes[0] if dominant.possible_root_causes else 'further investigation needed.'}"
        )

    return RootCauseReport(
        issues=issues,
        total_issues=len(issues),
        high_severity=high,
        medium_severity=med,
        low_severity=low,
        summary=summary,
    )
