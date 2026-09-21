"""
DataGuard AI — Quality Scorer
Converts a ProfileResult into a single AI-Readiness score (0–100)
and per-dimension grades.
"""
from __future__ import annotations

from dataclasses import dataclass

from dataguard_ai.core.profiler import ProfileResult


# ─────────────────────────────────────────────────────────────────────────────
# Weights (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS = {
    "completeness": 0.30,
    "consistency":  0.15,
    "uniqueness":   0.15,
    "validity":     0.20,
    "freshness":    0.10,
    "schema":       0.10,
}

FRESHNESS_SCORE_MAP = {
    "OK":      100.0,
    "STALE":    40.0,
    "UNKNOWN":  60.0,
}

SCHEMA_PENALTY_PER_ISSUE = 15.0   # deducted per schema issue, floor 0


@dataclass
class QualityScore:
    ai_readiness: float          # 0–100 final score

    completeness:  float
    consistency:   float
    uniqueness:    float
    validity:      float
    freshness:     float
    schema:        float

    grade: str                   # A / B / C / D / F
    verdict: str                 # human-readable one-liner

    # raw inputs for transparency
    missing_pct: float
    duplicate_pct: float
    outlier_pct: float
    schema_issue_count: int
    freshness_days: float | None
    freshness_flag: str


def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _verdict(score: float, grade: str) -> str:
    verdicts = {
        "A": "Data is highly reliable and AI-ready.",
        "B": "Data is broadly usable; minor issues should be monitored.",
        "C": "Data has notable quality gaps — remediation recommended before AI use.",
        "D": "Data quality is poor; significant remediation required.",
        "F": "Data is not trustworthy for analytics or AI. Do not use without major remediation.",
    }
    return verdicts[grade]


def score(profile: ProfileResult) -> QualityScore:
    """Compute the composite AI-Readiness score from a ProfileResult."""

    # ── per-dimension scores ──────────────────────────────────────────────────
    completeness = profile.completeness_score

    # Consistency: penalise heavily for format issues (already done in profiler)
    consistency  = profile.consistency_score

    # Uniqueness: sharpen penalty for high duplicate rates
    dup_pct = profile.duplicate_pct
    uniqueness = max(0.0, 100.0 - dup_pct * 2.5)   # 10% dupes → 75, 20% → 50

    # Validity: non-linear penalty so 10%+ outliers really hurts
    outlier_pct = profile.total_outlier_pct
    # 0% → 100, 10% → ~70, 20% → ~45, 30% → ~25
    validity = max(0.0, 100.0 - outlier_pct * 3.0)

    freshness_raw = FRESHNESS_SCORE_MAP.get(profile.freshness_flag, 60.0)
    # If we have an exact age, penalise more granularly
    if profile.freshness_days is not None and profile.freshness_flag != "OK":
        days = profile.freshness_days
        # linear decay: 30 days → 100, 365 days → 20
        freshness_raw = max(20.0, 100.0 - (days - 30) * (80 / 335))

    schema_raw = max(
        0.0,
        100.0 - len(profile.schema_issues) * SCHEMA_PENALTY_PER_ISSUE,
    )

    # ── weighted composite ────────────────────────────────────────────────────
    raw = (
        WEIGHTS["completeness"] * completeness
        + WEIGHTS["consistency"]  * consistency
        + WEIGHTS["uniqueness"]   * uniqueness
        + WEIGHTS["validity"]     * validity
        + WEIGHTS["freshness"]    * freshness_raw
        + WEIGHTS["schema"]       * schema_raw
    )
    ai_readiness = round(min(100.0, max(0.0, raw)), 1)

    grade = _grade(ai_readiness)

    return QualityScore(
        ai_readiness=ai_readiness,
        completeness=round(completeness, 1),
        consistency=round(consistency, 1),
        uniqueness=round(uniqueness, 1),
        validity=round(validity, 1),
        freshness=round(freshness_raw, 1),
        schema=round(schema_raw, 1),
        grade=grade,
        verdict=_verdict(ai_readiness, grade),
        missing_pct=profile.missing_cell_pct,
        duplicate_pct=profile.duplicate_pct,
        outlier_pct=profile.total_outlier_pct,
        schema_issue_count=len(profile.schema_issues),
        freshness_days=profile.freshness_days,
        freshness_flag=profile.freshness_flag,
    )
