"""
DataGuard AI — AI layer public API

Usage:
    from dataguard_ai.ai import run_analysis
    from dataguard_ai.ai.llm import chat_stream, ollama_status
    from dataguard_ai.ai.narrative import get_narrative

    report = run_analysis(df, profile, quality_score)
    narrative, is_ai = get_narrative(profile, quality_score, report.root_cause,
                                     report.impact, report.recommendations)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dataguard_ai.core.profiler import ProfileResult
from dataguard_ai.core.scorer import QualityScore
from dataguard_ai.ai.root_cause import RootCauseReport, analyze
from dataguard_ai.ai.impact import BusinessImpactReport, estimate
from dataguard_ai.ai.recommendations import RecommendationReport, recommend


@dataclass
class AnalysisReport:
    root_cause: RootCauseReport
    impact: BusinessImpactReport
    recommendations: RecommendationReport


def run_analysis(
    df: pd.DataFrame,
    profile: ProfileResult,
    quality_score: QualityScore,
) -> AnalysisReport:
    """
    Run the full AI analysis pipeline:
      1. Root-cause analysis
      2. Business impact estimation
      3. Actionable recommendations

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset (used for cluster detection and monetary estimation).
    profile : ProfileResult
        Output of dataguard_ai.core.profiler.profile(df).
    quality_score : QualityScore
        Output of dataguard_ai.core.scorer.score(profile).

    Returns
    -------
    AnalysisReport
        Consolidated findings from all three AI layers.
    """
    root_cause = analyze(df, profile)
    impact = estimate(df, profile, root_cause)
    recs = recommend(root_cause, impact, quality_score)
    return AnalysisReport(
        root_cause=root_cause,
        impact=impact,
        recommendations=recs,
    )
