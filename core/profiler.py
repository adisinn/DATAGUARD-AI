"""
DataGuard AI — Data Profiler
Analyses a DataFrame across seven quality dimensions and returns a
structured ProfileResult that the rest of the pipeline consumes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    # numeric stats (None for non-numeric columns)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    outlier_count: int = 0
    outlier_pct: float = 0.0
    # categorical / string stats
    top_values: dict[str, int] = field(default_factory=dict)
    # format / pattern issues
    format_issues: int = 0
    format_issue_examples: list[str] = field(default_factory=list)


@dataclass
class ProfileResult:
    # ── row-level stats ──────────────────────────────────────────────────────
    total_rows: int
    total_cols: int
    duplicate_count: int
    duplicate_pct: float

    # ── dimension scores (0-100) ─────────────────────────────────────────────
    completeness_score: float      # 100 − % missing across all cells
    consistency_score: float       # based on format / type anomalies
    uniqueness_score: float        # 100 − duplicate row %
    validity_score: float          # format / range rule passes

    # ── freshness ────────────────────────────────────────────────────────────
    freshness_days: float | None   # None if no datetime col found
    freshness_flag: str            # "OK" / "STALE" / "UNKNOWN"

    # ── per-column detail ────────────────────────────────────────────────────
    columns: list[ColumnProfile]

    # ── global outlier summary ───────────────────────────────────────────────
    total_outlier_rows: int
    total_outlier_pct: float

    # ── schema ───────────────────────────────────────────────────────────────
    schema_issues: list[str]      # e.g. mixed-type columns

    # ── raw missing summary ──────────────────────────────────────────────────
    missing_cells: int
    missing_cell_pct: float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$")
_PHONE_RE = re.compile(r"^[\+\d\s\-\(\)]{7,15}$")


def _detect_format_issues(series: pd.Series) -> tuple[int, list[str]]:
    """
    Heuristically detect formatting problems in object/string columns.
    Checks for: mixed types masquerading as strings, leading/trailing
    whitespace, obvious email/phone format violations.
    """
    if series.dtype != object:
        return 0, []

    sample = series.dropna()
    if sample.empty:
        return 0, []

    issues: list[str] = []
    issue_count = 0

    # whitespace anomalies
    ws = sample[sample.astype(str).str.strip() != sample.astype(str)]
    if len(ws):
        issue_count += len(ws)
        issues.extend(ws.astype(str).head(3).tolist())

    # numeric-looking values mixed with non-numeric
    numeric_mask = pd.to_numeric(sample, errors="coerce").notna()
    if 0 < numeric_mask.sum() < len(sample) * 0.9:
        # column looks like it wants to be numeric but has strings
        non_numeric = sample[~numeric_mask]
        if len(non_numeric) > 0:
            issue_count += len(non_numeric)
            issues.extend(non_numeric.head(3).astype(str).tolist())

    return issue_count, list(set(issues))[:5]


def _negative_value_count(series: pd.Series) -> int:
    """Count non-null negative values — used for columns that can't be negative."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return int((clean < 0).sum())


def _outlier_count_iqr(series: pd.Series) -> tuple[int, float]:
    """IQR-based outlier detection for numeric columns.

    Uses IQR×1.5 (standard Tukey fence) rather than IQR×3 so that
    moderate outliers are caught and reflected in the quality score.
    """
    clean = series.dropna()
    if len(clean) < 10:
        return 0, 0.0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (clean < lower) | (clean > upper)
    cnt = int(mask.sum())
    pct = round(cnt / len(series) * 100, 2)
    return cnt, pct


def _freshness(df: pd.DataFrame) -> tuple[float | None, str]:
    """Return days since the most recent timestamp found in the data."""
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # also try to coerce object columns that look like dates
    for col in df.select_dtypes(include="object").columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                date_cols.append(col)
                df = df.copy()
                df[col] = parsed
        except Exception:
            pass

    if not date_cols:
        return None, "UNKNOWN"

    latest = max(
        df[col].dropna().max()
        for col in date_cols
        if not df[col].dropna().empty
    )
    if pd.isnull(latest):
        return None, "UNKNOWN"

    delta = (pd.Timestamp.now() - pd.Timestamp(latest)).days
    flag = "OK" if delta <= 30 else "STALE"
    return float(delta), flag


def _schema_issues(df: pd.DataFrame) -> list[str]:
    """Detect mixed-type columns and other schema anomalies."""
    issues: list[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            # Check if numeric values are stored as strings
            num_pct = pd.to_numeric(df[col], errors="coerce").notna().mean()
            if 0.1 < num_pct < 0.9:
                issues.append(
                    f"Column '{col}' appears to mix numeric and non-numeric values "
                    f"({num_pct*100:.0f}% parseable as numbers)."
                )
    # duplicate column names
    dupes = [c for c in df.columns if list(df.columns).count(c) > 1]
    if dupes:
        issues.append(f"Duplicate column names detected: {list(set(dupes))}")
    # all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"Completely empty columns: {null_cols}")
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Main profiler
# ─────────────────────────────────────────────────────────────────────────────

def profile(df: pd.DataFrame) -> ProfileResult:
    """
    Run the full data-quality profile on *df* and return a ProfileResult.
    """
    total_rows, total_cols = df.shape

    # ── duplicates ────────────────────────────────────────────────────────────
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    dup_pct = round(dup_count / total_rows * 100, 2) if total_rows else 0.0

    # ── missing ───────────────────────────────────────────────────────────────
    missing_cells = int(df.isnull().sum().sum())
    total_cells = total_rows * total_cols
    missing_cell_pct = round(missing_cells / total_cells * 100, 2) if total_cells else 0.0
    completeness_score = round(100 - missing_cell_pct, 2)

    # ── per-column ────────────────────────────────────────────────────────────
    col_profiles: list[ColumnProfile] = []
    total_format_issues = 0
    total_outlier_rows_set: set[int] = set()

    for col in df.columns:
        series = df[col]
        miss_cnt = int(series.isnull().sum())
        miss_pct = round(miss_cnt / total_rows * 100, 2) if total_rows else 0.0
        uniq_cnt = int(series.nunique(dropna=True))
        uniq_pct = round(uniq_cnt / total_rows * 100, 2) if total_rows else 0.0

        cp = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            missing_count=miss_cnt,
            missing_pct=miss_pct,
            unique_count=uniq_cnt,
            unique_pct=uniq_pct,
        )

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.8 and series.dtype != bool:
            clean_num = numeric.dropna()
            if len(clean_num):
                cp.mean = round(float(clean_num.mean()), 4)
                cp.std = round(float(clean_num.std()), 4)
                cp.min = round(float(clean_num.min()), 4)
                cp.max = round(float(clean_num.max()), 4)
                cp.q1 = round(float(clean_num.quantile(0.25)), 4)
                cp.median = round(float(clean_num.median()), 4)
                cp.q3 = round(float(clean_num.quantile(0.75)), 4)
            out_cnt, out_pct = _outlier_count_iqr(numeric)
            # Also catch negative values in columns whose name implies non-negativity
            _neg_keywords = ("age","qty","quantity","count","los","stay","price",
                             "stock","duration","amount","revenue","salary","cost",
                             "rating","score","length")
            col_lower = col.lower()
            if any(kw in col_lower for kw in _neg_keywords):
                neg_cnt = _negative_value_count(numeric)
                if neg_cnt > 0:
                    neg_idx = numeric[numeric < 0].index
                    total_outlier_rows_set.update(neg_idx.tolist())
                    out_cnt = max(out_cnt, neg_cnt)
            out_pct = round(out_cnt / max(len(series), 1) * 100, 2)
            cp.outlier_count = out_cnt
            cp.outlier_pct = out_pct
            if out_cnt:
                q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
                iqr = q3 - q1
                outlier_idx = numeric[(numeric < q1 - 1.5*iqr) | (numeric > q3 + 1.5*iqr)].index
                total_outlier_rows_set.update(outlier_idx.tolist())
        else:
            fmt_cnt, fmt_ex = _detect_format_issues(series)
            cp.format_issues = fmt_cnt
            cp.format_issue_examples = fmt_ex
            total_format_issues += fmt_cnt
            # top values for categoricals
            vc = series.value_counts().head(5)
            cp.top_values = {str(k): int(v) for k, v in vc.items()}

        col_profiles.append(cp)

    # ── aggregate outlier stat ────────────────────────────────────────────────
    total_outlier_rows = len(total_outlier_rows_set)
    total_outlier_pct = round(total_outlier_rows / total_rows * 100, 2) if total_rows else 0.0

    # ── consistency / validity ────────────────────────────────────────────────
    consistency_score = round(
        max(0, 100 - (total_format_issues / max(total_cells, 1)) * 100), 2
    )
    validity_score = round(
        (1 - total_outlier_rows / max(total_rows, 1)) * 100, 2
    )

    # ── uniqueness ────────────────────────────────────────────────────────────
    uniqueness_score = round(100 - dup_pct, 2)

    # ── freshness ─────────────────────────────────────────────────────────────
    freshness_days, freshness_flag = _freshness(df)

    # ── schema ────────────────────────────────────────────────────────────────
    schema_issues = _schema_issues(df)

    return ProfileResult(
        total_rows=total_rows,
        total_cols=total_cols,
        duplicate_count=dup_count,
        duplicate_pct=dup_pct,
        completeness_score=completeness_score,
        consistency_score=consistency_score,
        uniqueness_score=uniqueness_score,
        validity_score=validity_score,
        freshness_days=freshness_days,
        freshness_flag=freshness_flag,
        columns=col_profiles,
        total_outlier_rows=total_outlier_rows,
        total_outlier_pct=total_outlier_pct,
        schema_issues=schema_issues,
        missing_cells=missing_cells,
        missing_cell_pct=missing_cell_pct,
    )
