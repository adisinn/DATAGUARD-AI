"""
DataGuard AI — LLM Client
Thin wrapper around the Ollama Python SDK.

Design principles:
  - Every public function has a graceful fallback: if Ollama is not running or
    the model isn't available, it returns a clearly-labelled static string so
    the rest of the app never crashes.
  - Streaming is supported so the chat interface feels responsive.
  - Model selection is configurable via DATAGUARD_LLM_MODEL env var (default: llama3.2).
"""
from __future__ import annotations

import os
import json
import textwrap
from typing import Generator, Any

try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

from dataguard_ai.core.profiler import ProfileResult
from dataguard_ai.core.scorer import QualityScore
from dataguard_ai.ai.root_cause import RootCauseReport
from dataguard_ai.ai.impact import BusinessImpactReport
from dataguard_ai.ai.recommendations import RecommendationReport

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
# Prefer our custom-trained model; fall back to base llama3.2 if not yet created
DEFAULT_MODEL   = os.getenv("DATAGUARD_LLM_MODEL", "dataguard-ai")
_FALLBACK_MODEL = "llama3.2"

_FALLBACK_MSG = (
    "⚠️ **AI analyst unavailable** — Ollama is not running or no model is installed.\n\n"
    "**To enable:**\n"
    "1. Open the **Ollama app** on your Mac (or run `ollama serve` in a terminal)\n"
    "2. The `dataguard-ai` model should already be available\n"
    "3. Refresh this page — the AI Analyst will activate automatically"
)


def _is_ollama_up() -> bool:
    """Quick connectivity check — tries to list models."""
    if not _OLLAMA_AVAILABLE:
        return False
    try:
        _ollama.list()
        return True
    except Exception:
        return False


def _available_model() -> str | None:
    """
    Return the best available model name:
    1. dataguard-ai (our custom model) — preferred
    2. llama3.2 (base model fallback)
    3. Any other installed model
    4. None if nothing is available
    """
    if not _OLLAMA_AVAILABLE:
        return None
    try:
        models = _ollama.list()
        names = [m.model for m in models.models]
        # prefer our custom model
        for n in names:
            if n.startswith(DEFAULT_MODEL):
                return n
        # fall back to base llama3.2
        for n in names:
            if n.startswith(_FALLBACK_MODEL):
                return n
        # any available model
        if names:
            return names[0]
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _profile_context(
    profile: ProfileResult,
    quality_score: QualityScore,
    root_cause: RootCauseReport,
    impact: BusinessImpactReport,
    recommendations: RecommendationReport,
    dataset_name: str = "the dataset",
) -> str:
    """
    Serialise the full analysis result into a compact, structured context block
    that fits comfortably within a small model's context window (~1500 tokens).
    """
    issues_text = "\n".join(
        f"  - [{i.severity}] {i.title} ({i.affected_pct:.1f}% of rows): "
        f"{i.possible_root_causes[0] if i.possible_root_causes else 'unknown cause'}"
        for i in root_cause.issues[:8]
    ) or "  None detected."

    recs_text = "\n".join(
        f"  {r.priority}. {r.action} [Effort:{r.effort}, Impact:{r.impact}]"
        for r in recommendations.recommendations[:6]
    ) or "  No recommendations."

    cols_text = "\n".join(
        f"  - {cp.name} ({cp.dtype}): "
        + (f"missing={cp.missing_pct:.1f}%, outliers={cp.outlier_count}" if cp.mean is not None
           else f"missing={cp.missing_pct:.1f}%, format_issues={cp.format_issues}")
        for cp in profile.columns[:15]
    )

    monetary = (
        f"{impact.currency_symbol}{impact.total_monetary_exposure:,.0f}"
        if impact.total_monetary_exposure else "unknown"
    )

    return textwrap.dedent(f"""
        DATASET: {dataset_name}
        SHAPE: {profile.total_rows:,} rows × {profile.total_cols} columns

        QUALITY SCORES (0-100):
          AI Readiness: {quality_score.ai_readiness}/100 (Grade {quality_score.grade})
          Completeness: {quality_score.completeness}
          Validity: {quality_score.validity}
          Uniqueness: {quality_score.uniqueness}
          Consistency: {quality_score.consistency}
          Freshness: {quality_score.freshness} ({profile.freshness_flag})
          Schema: {quality_score.schema}

        RAW METRICS:
          Missing cells: {profile.missing_cell_pct:.2f}%
          Duplicate rows: {profile.duplicate_pct:.2f}%
          Outlier rows: {profile.total_outlier_pct:.2f}%
          Schema issues: {len(profile.schema_issues)}
          Freshness: {int(profile.freshness_days or 0)} days since latest record

        COLUMNS:
        {cols_text}

        QUALITY ISSUES FOUND ({root_cause.total_issues} total):
        {issues_text}

        BUSINESS IMPACT:
          Estimated monetary exposure: {monetary}
          Total affected records: {impact.total_affected_rows:,} ({impact.unique_affected_pct:.1f}%)
          Summary: {impact.executive_summary}

        TOP RECOMMENDATIONS:
        {recs_text}
    """).strip()


# When using the custom dataguard-ai model, the full persona/rules are already
# baked into the Modelfile SYSTEM block. This runtime system message only needs
# to inject the live dataset context + a brief reminder to stay grounded.
SYSTEM_PROMPT_WITH_CONTEXT = textwrap.dedent("""
    You are currently analysing the dataset described below.
    Always ground your answers in these specific numbers — do not invent figures.
    If you do not know something from the context, say so honestly.

    LIVE DATASET ANALYSIS:
    {context}
""").strip()

# Fallback system prompt used when running on the base llama3.2 model
# (before the custom model is created) — includes full behavioural instructions
SYSTEM_PROMPT_BASE = textwrap.dedent("""
    You are DataGuard AI — a specialist data quality analyst built into the DataGuard AI platform.
    You only discuss data quality, this dataset's analysis results, and AI-readiness topics.
    Never answer off-topic questions. Never make up numbers not present in the context.
    Be direct, specific, and concise. No filler phrases like "Great question!" or "Certainly!".
    If the AI Readiness score is below 70, do not recommend the data for ML use without caveats.

    LIVE DATASET ANALYSIS:
    {context}
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrative(
    profile: ProfileResult,
    quality_score: QualityScore,
    root_cause: RootCauseReport,
    impact: BusinessImpactReport,
    recommendations: RecommendationReport,
    dataset_name: str = "the dataset",
) -> str:
    """
    Option A: Generate a full AI-written executive narrative for the report.
    Returns a Markdown string. Falls back gracefully if Ollama is unavailable.
    """
    model = _available_model()
    if not model:
        return _FALLBACK_MSG

    context = _profile_context(
        profile, quality_score, root_cause, impact, recommendations, dataset_name
    )

    prompt = textwrap.dedent(f"""
        Based on the data quality analysis below, write a concise executive narrative
        (3-4 paragraphs) that:
        1. Summarises the overall data health and AI-readiness score
        2. Highlights the most critical issues with their business consequences
        3. Identifies the likely root causes based on the evidence
        4. Closes with a clear remediation priority

        Write in plain English for a business audience. Reference specific numbers.
        Do NOT use bullet points — flowing paragraphs only.

        ANALYSIS CONTEXT:
        {context}
    """).strip()

    is_custom = model.startswith(DEFAULT_MODEL)
    sys_template = SYSTEM_PROMPT_WITH_CONTEXT if is_custom else SYSTEM_PROMPT_BASE
    system_msg = sys_template.format(context=context)

    try:
        response = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.message.content.strip()
    except Exception as e:
        return f"{_FALLBACK_MSG}\n\n*Error: {e}*"


def chat_stream(
    user_message: str,
    conversation_history: list[dict[str, str]],
    profile: ProfileResult,
    quality_score: QualityScore,
    root_cause: RootCauseReport,
    impact: BusinessImpactReport,
    recommendations: RecommendationReport,
    dataset_name: str = "the dataset",
) -> Generator[str, None, None]:
    """
    Option C: Stream a chat response token-by-token.
    Yields string chunks. Falls back to a single yielded string if Ollama is down.
    """
    model = _available_model()
    if not model:
        yield _FALLBACK_MSG
        return

    # ── Content guard: reject clearly off-topic messages before hitting the model ──
    _off_topic_triggers = [
        "poem", "joke", "story", "recipe", "weather", "news", "sports",
        "translate", "write me", "write a", "sing", "lyrics", "fiction",
        "stock price", "crypto", "investment advice", "personal advice",
        "what is the capital", "who is the president", "history of",
        "meaning of life", "tell me about yourself",
    ]
    _msg_lower = user_message.lower()
    if any(t in _msg_lower for t in _off_topic_triggers):
        yield (
            "I'm DataGuard AI — I'm specialised in data quality analysis. "
            "I can only help with questions about this dataset's quality, "
            "trustworthiness, and AI-readiness. What would you like to know about your data?"
        )
        return

    context = _profile_context(
        profile, quality_score, root_cause, impact, recommendations, dataset_name
    )

    is_custom = model.startswith(DEFAULT_MODEL)
    sys_template = SYSTEM_PROMPT_WITH_CONTEXT if is_custom else SYSTEM_PROMPT_BASE
    system_msg = sys_template.format(context=context)

    messages = [{"role": "system", "content": system_msg}]

    # Include last N turns of history to keep context window manageable
    for turn in conversation_history[-6:]:
        messages.append(turn)

    messages.append({"role": "user", "content": user_message})

    try:
        stream = _ollama.chat(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            token = chunk.message.content
            if token:
                yield token
    except Exception as e:
        yield f"\n\n⚠️ *LLM error: {e}*"


def ollama_status() -> dict[str, Any]:
    """Return a status dict for display in the UI."""
    if not _OLLAMA_AVAILABLE:
        return {"available": False, "server_up": False, "model": None,
                "reason": "ollama Python package not installed", "is_custom": False}
    up = _is_ollama_up()
    model = _available_model() if up else None
    is_custom = bool(model and model.startswith(DEFAULT_MODEL))
    return {
        "available": up and model is not None,
        "server_up": up,
        "model": model,
        "is_custom": is_custom,
        "reason": (
            f"dataguard-ai model active" if is_custom
            else f"Using base {model} (dataguard-ai not found)" if (up and model)
            else "No model installed — run: ollama pull llama3.2" if up
            else "Ollama server not running — open the Ollama app"
        ),
    }
