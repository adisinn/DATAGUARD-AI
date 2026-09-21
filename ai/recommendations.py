"""
DataGuard AI — Recommendations Engine
Generates prioritised, actionable remediation recommendations from the
root-cause report and business impact report.

Each recommendation is ranked by business impact and feasibility.
The output is structured so the UI can render it as a clear action plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dataguard_ai.ai.root_cause import IssueInsight, RootCauseReport
from dataguard_ai.ai.impact import ImpactEstimate, BusinessImpactReport
from dataguard_ai.core.scorer import QualityScore


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    priority: int               # 1 = highest
    action: str                 # short imperative headline
    detail: str                 # step-by-step or explanatory paragraph
    dimension: str
    effort: str                 # Low / Medium / High
    impact: str                 # Low / Medium / High
    issue_ref: str              # title of the originating issue


@dataclass
class RecommendationReport:
    recommendations: list[Recommendation]
    remediation_plan: str       # ordered paragraph for exec reading
    estimated_score_lift: float # rough score improvement if all HIGH recs implemented


# ─────────────────────────────────────────────────────────────────────────────
# Template library
# ─────────────────────────────────────────────────────────────────────────────

def _missing_recs(issue: IssueInsight, impact: ImpactEstimate | None) -> list[Recommendation]:
    recs: list[Recommendation] = []
    pct = issue.affected_pct

    # Immediate: flag / quarantine
    recs.append(Recommendation(
        priority=0,
        action=f"Quarantine rows with missing '{issue.affected_column}' for manual review",
        detail=(
            f"Create a separate quarantine table/view containing all {issue.affected_rows:,} rows "
            f"where '{issue.affected_column}' is NULL. Do not include these rows in KPIs or "
            "model training until values are validated or imputed."
        ),
        dimension="completeness",
        effort="Low",
        impact="High" if pct >= 10 else "Medium",
        issue_ref=issue.title,
    ))

    # Root-cause fix
    if issue.correlated_columns:
        recs.append(Recommendation(
            priority=0,
            action=f"Investigate upstream source for columns {issue.correlated_columns}",
            detail=(
                f"Missingness in '{issue.affected_column}' is correlated with "
                f"{issue.correlated_columns}. This pattern suggests a shared upstream join or "
                "ETL stage is failing. Audit the pipeline step that populates these columns."
            ),
            dimension="completeness",
            effort="Medium",
            impact="High",
            issue_ref=issue.title,
        ))
    else:
        recs.append(Recommendation(
            priority=0,
            action=f"Add a NOT NULL validation rule on '{issue.affected_column}' in the pipeline",
            detail=(
                f"Implement a pipeline-level assertion that rejects or flags records where "
                f"'{issue.affected_column}' is NULL before they are written to the target table. "
                "This prevents future silent data loss."
            ),
            dimension="completeness",
            effort="Low",
            impact="Medium",
            issue_ref=issue.title,
        ))

    # Long-term: imputation strategy
    if pct < 20:
        recs.append(Recommendation(
            priority=0,
            action=f"Impute missing '{issue.affected_column}' values using median/mode strategy",
            detail=(
                f"Since only {pct:.1f}% of rows are missing, statistical imputation (median for "
                "numeric, mode for categorical) is safe for analytical purposes. Document the "
                "imputation in column metadata so downstream consumers are aware."
            ),
            dimension="completeness",
            effort="Low",
            impact="Medium",
            issue_ref=issue.title,
        ))

    return recs


def _outlier_recs(issue: IssueInsight, impact: ImpactEstimate | None) -> list[Recommendation]:
    recs: list[Recommendation] = []

    # Date-cluster evidence?
    date_ev = issue.evidence.get("date_clusters", {})
    cat_ev = issue.evidence.get("category_clusters", {})

    if date_ev:
        for dc, info in date_ev.items():
            recs.append(Recommendation(
                priority=0,
                action=f"Validate source-system records for {info['from']} → {info['to']}",
                detail=(
                    f"Anomalous values in '{issue.affected_column}' cluster within a "
                    f"{info['window_days']}-day window in '{dc}'. "
                    "Reprocess the affected ingestion batch and compare against source-system logs. "
                    "Check for unit conversion bugs (e.g., paise vs rupees) or multiplier errors."
                ),
                dimension="validity",
                effort="Medium",
                impact="High",
                issue_ref=issue.title,
            ))

    if cat_ev:
        for cat, info in cat_ev.items():
            recs.append(Recommendation(
                priority=0,
                action=f"Audit '{cat}' = '{info['top_value']}' for data-entry or source-config issues",
                detail=(
                    f"{info['concentration_pct']}% of outliers in '{issue.affected_column}' "
                    f"share the value '{info['top_value']}' in column '{cat}'. "
                    "Investigate whether this segment uses a different unit, currency, or scale. "
                    "Apply a segment-specific validation rule if confirmed."
                ),
                dimension="validity",
                effort="Medium",
                impact="High",
                issue_ref=issue.title,
            ))

    # Always add: add automated range rule
    recs.append(Recommendation(
        priority=0,
        action=f"Add an automated range-validation rule for '{issue.affected_column}'",
        detail=(
            f"Define acceptable bounds for '{issue.affected_column}' (e.g., "
            f"[{issue.evidence.get('q1', 'Q1')} – {issue.evidence.get('q3', 'Q3')} × 3×IQR]) "
            "and enforce this check in the ingestion pipeline. Alert on-call data engineers when "
            "records exceed the threshold."
        ),
        dimension="validity",
        effort="Low",
        impact="High",
        issue_ref=issue.title,
    ))

    if impact and impact.monetary_impact:
        recs.append(Recommendation(
            priority=0,
            action=f"Recalculate affected KPIs after quarantining anomalous records",
            detail=(
                f"Estimated {impact.currency_symbol}{impact.monetary_impact:,.0f} of reported "
                f"values may be distorted. Re-run revenue/metric aggregations excluding the "
                f"{issue.affected_rows:,} anomalous rows and compare results to current dashboards."
            ),
            dimension="validity",
            effort="Low",
            impact="High",
            issue_ref=issue.title,
        ))

    return recs


def _format_recs(issue: IssueInsight) -> list[Recommendation]:
    return [Recommendation(
        priority=0,
        action=f"Standardise formatting in '{issue.affected_column}'",
        detail=(
            f"Apply a transformation step that strips whitespace, enforces consistent casing, "
            f"and coerces the column to its intended data type. "
            f"Example problem values: {issue.evidence.get('examples', [])}. "
            "Add a schema enforcement step in the ETL to prevent recurrence."
        ),
        dimension="consistency",
        effort="Low",
        impact="Medium",
        issue_ref=issue.title,
    )]


def _duplicate_recs(issue: IssueInsight) -> list[Recommendation]:
    return [
        Recommendation(
            priority=0,
            action="Add deduplication logic to the ingestion pipeline",
            detail=(
                f"{issue.affected_rows:,} exact duplicate rows detected ({issue.affected_pct:.1f}%). "
                "Implement an INSERT … ON CONFLICT IGNORE (or equivalent MERGE/UPSERT) pattern "
                "in the pipeline. Use a composite natural key for deduplication rather than "
                "relying on auto-increment IDs."
            ),
            dimension="uniqueness",
            effort="Medium",
            impact="High",
            issue_ref=issue.title,
        ),
        Recommendation(
            priority=0,
            action="Audit pipeline idempotency — re-runs should not double-load records",
            detail=(
                "Verify that the pipeline tracks processed watermarks/offsets. "
                "If using batch loads, ensure file manifests or change-data-capture (CDC) "
                "checkpoints prevent re-ingestion of already-processed files."
            ),
            dimension="uniqueness",
            effort="Medium",
            impact="Medium",
            issue_ref=issue.title,
        ),
    ]


def _schema_recs(issue: IssueInsight) -> list[Recommendation]:
    return [
        Recommendation(
            priority=0,
            action="Enforce schema contracts at the pipeline boundary",
            detail=(
                "Adopt a schema registry (e.g., Apache Avro, Great Expectations, Pandera) "
                "to validate incoming data against expected column types before ingestion. "
                f"Detected issues: {issue.evidence.get('schema_issues', [])}."
            ),
            dimension="schema",
            effort="Medium",
            impact="High",
            issue_ref=issue.title,
        ),
    ]


def _freshness_recs(issue: IssueInsight) -> list[Recommendation]:
    days = issue.evidence.get("freshness_days", "unknown")
    return [
        Recommendation(
            priority=0,
            action="Investigate and restart the stale data pipeline",
            detail=(
                f"The dataset is {days} days old. Check scheduler logs (Airflow, cron, dbt) "
                "for failed pipeline runs. Verify the data source is still emitting records. "
                "Set up a freshness SLA alert that fires if data is not updated within 24 hours."
            ),
            dimension="freshness",
            effort="Low",
            impact="High",
            issue_ref=issue.title,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def recommend(
    root_cause_report: RootCauseReport,
    impact_report: BusinessImpactReport,
    quality_score: QualityScore,
) -> RecommendationReport:
    """
    Generate prioritised recommendations from root-cause findings and impact estimates.
    """
    impact_map: dict[str, ImpactEstimate] = {
        e.issue_title: e for e in impact_report.estimates
    }

    raw_recs: list[Recommendation] = []
    for issue in root_cause_report.issues:
        impact = impact_map.get(issue.title)
        if issue.dimension == "completeness":
            raw_recs.extend(_missing_recs(issue, impact))
        elif issue.dimension == "validity":
            raw_recs.extend(_outlier_recs(issue, impact))
        elif issue.dimension == "consistency":
            raw_recs.extend(_format_recs(issue))
        elif issue.dimension == "uniqueness":
            raw_recs.extend(_duplicate_recs(issue))
        elif issue.dimension == "schema":
            raw_recs.extend(_schema_recs(issue))
        elif issue.dimension == "freshness":
            raw_recs.extend(_freshness_recs(issue))

    # ── global catch-all recs ─────────────────────────────────────────────────
    if quality_score.ai_readiness < 70:
        raw_recs.append(Recommendation(
            priority=0,
            action="Establish a continuous data-quality monitoring process",
            detail=(
                "AI-Readiness score is below 70. Set up scheduled profiling runs (daily/weekly) "
                "with alerting on key quality thresholds. Track the score over time to measure "
                "remediation progress."
            ),
            dimension="general",
            effort="Medium",
            impact="High",
            issue_ref="Overall data quality",
        ))

    # ── prioritise: High impact + Low effort first (quick wins first) ─────────
    effort_order = {"Low": 0, "Medium": 1, "High": 2}
    impact_order = {"High": 0, "Medium": 1, "Low": 2}
    raw_recs.sort(key=lambda r: (impact_order[r.impact], effort_order[r.effort]))

    # assign sequential priority numbers
    for i, rec in enumerate(raw_recs, start=1):
        rec.priority = i

    # ── remediation plan narrative ────────────────────────────────────────────
    if not raw_recs:
        plan = "No remediation actions required. Data is in good health."
        lift = 0.0
    else:
        quick = [r for r in raw_recs if r.effort == "Low" and r.impact == "High"]
        medium = [r for r in raw_recs if r.effort == "Medium"]
        plan_parts: list[str] = []
        if quick:
            plan_parts.append(
                f"Immediate actions ({len(quick)} quick wins): "
                + "; ".join(r.action for r in quick[:3])
                + ("." if len(quick) <= 3 else f" and {len(quick)-3} more.")
            )
        if medium:
            plan_parts.append(
                f"Medium-term fixes ({len(medium)} items): "
                + "; ".join(r.action for r in medium[:3])
                + "."
            )
        long_term = [r for r in raw_recs if r.effort == "High"]
        if long_term:
            plan_parts.append(
                f"Long-term investments ({len(long_term)} items): "
                + "; ".join(r.action for r in long_term[:2])
                + "."
            )
        plan = " → ".join(plan_parts) if plan_parts else "Implement all listed recommendations."

        # rough score lift: each HIGH-severity issue fixed ≈ 5–10 points
        high_issues = root_cause_report.high_severity
        lift = min(30.0, high_issues * 8.0 + root_cause_report.medium_severity * 3.0)

    return RecommendationReport(
        recommendations=raw_recs,
        remediation_plan=plan,
        estimated_score_lift=round(lift, 1),
    )
