"""
DataGuard AI — LLM Narrative Generator (Option A)
Generates an AI-written executive narrative for the report header.
Thin wrapper around ai.llm.generate_narrative with caching support.
"""
from __future__ import annotations

from dataguard_ai.core.profiler import ProfileResult
from dataguard_ai.core.scorer import QualityScore
from dataguard_ai.ai.root_cause import RootCauseReport
from dataguard_ai.ai.impact import BusinessImpactReport
from dataguard_ai.ai.recommendations import RecommendationReport
from dataguard_ai.ai.llm import generate_narrative, ollama_status


def get_narrative(
    profile: ProfileResult,
    quality_score: QualityScore,
    root_cause: RootCauseReport,
    impact: BusinessImpactReport,
    recommendations: RecommendationReport,
    dataset_name: str = "the dataset",
) -> tuple[str, bool]:
    """
    Returns (narrative_text, was_ai_generated).
    If Ollama is unavailable, returns a static rule-based narrative instead
    so the report always has something useful to show.
    """
    status = ollama_status()
    if status["available"]:
        text = generate_narrative(
            profile, quality_score, root_cause, impact, recommendations, dataset_name
        )
        return text, True

    # ── Static fallback narrative ─────────────────────────────────────────────
    grade = quality_score.grade
    score = quality_score.ai_readiness
    issues = root_cause.total_issues
    high = root_cause.high_severity
    monetary = (
        f"{impact.currency_symbol}{impact.total_monetary_exposure:,.0f}"
        if impact.total_monetary_exposure else "undetermined"
    )
    dominant = root_cause.issues[0] if root_cause.issues else None

    para1 = (
        f"**{dataset_name}** received an AI Readiness score of **{score}/100 (Grade {grade})**, "
        f"indicating {'high reliability' if grade == 'A' else 'moderate reliability with notable gaps' if grade == 'B' else 'significant quality concerns requiring attention'}. "
        f"The analysis profiled {profile.total_rows:,} rows across {profile.total_cols} columns, "
        f"revealing {profile.missing_cell_pct:.1f}% missing cells, {profile.duplicate_pct:.1f}% duplicate rows, "
        f"and {profile.total_outlier_pct:.1f}% outlier records."
    )

    if dominant:
        para2 = (
            f"The most critical finding is **{dominant.title}** ({dominant.severity} severity, "
            f"affecting {dominant.affected_pct:.1f}% of records). "
            + (dominant.possible_root_causes[0] if dominant.possible_root_causes else "")
            + f" Across all {issues} issue(s) detected, estimated business exposure stands at **{monetary}**."
        )
    else:
        para2 = f"No significant quality issues were detected. The dataset appears ready for analytics and AI use."

    quick_wins = [r for r in recommendations.recommendations if r.effort == "Low" and r.impact == "High"]
    if quick_wins:
        para3 = (
            f"Immediate remediation should focus on {len(quick_wins)} quick-win action(s): "
            + "; ".join(r.action for r in quick_wins[:2])
            + ". Implementing HIGH-severity fixes is estimated to lift the AI Readiness score by "
            f"+{recommendations.estimated_score_lift:.0f} points."
        )
    else:
        para3 = (
            f"Implementing the {len(recommendations.recommendations)} recommended actions "
            f"is estimated to lift the AI Readiness score by +{recommendations.estimated_score_lift:.0f} points."
        )

    fallback_text = f"{para1}\n\n{para2}\n\n{para3}"
    return fallback_text, False
