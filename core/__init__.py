"""
DataGuard AI — Core layer public API

Usage:
    from dataguard_ai.core import run_profile

    profile, quality_score = run_profile(df)
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from dataguard_ai.core.profiler import ProfileResult, profile
from dataguard_ai.core.scorer import QualityScore, score


def run_profile(df: pd.DataFrame) -> Tuple[ProfileResult, QualityScore]:
    """
    Profile the DataFrame and compute the AI-Readiness quality score.

    Returns
    -------
    (ProfileResult, QualityScore)
    """
    p = profile(df)
    q = score(p)
    return p, q
