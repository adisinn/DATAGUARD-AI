"""
DataGuard AI — FastAPI Backend
Serves the analysis API and static frontend.

Run:  python3 server.py
      → http://localhost:8000
"""
from __future__ import annotations

import io
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from dataguard_ai.core import run_profile
from dataguard_ai.ai import run_analysis
from dataguard_ai.ai.llm import ollama_status, chat_stream
from dataguard_ai.ai.narrative import get_narrative

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="DataGuard AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
SAMPLE_DIR = Path(__file__).parent / "sample_data"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _profile_to_dict(p, qs, a, dataset_name: str) -> dict[str, Any]:
    """Serialise the full analysis result to a JSON-safe dict."""
    rc = a.root_cause
    imp = a.impact
    rr = a.recommendations

    return {
        "dataset_name": dataset_name,
        "total_rows": p.total_rows,
        "total_cols": p.total_cols,

        "score": {
            "ai_readiness": qs.ai_readiness,
            "grade": qs.grade,
            "verdict": qs.verdict,
            "completeness": qs.completeness,
            "validity": qs.validity,
            "uniqueness": qs.uniqueness,
            "consistency": qs.consistency,
            "freshness": qs.freshness,
            "schema": qs.schema,
        },

        "metrics": {
            "missing_cell_pct": p.missing_cell_pct,
            "duplicate_pct": p.duplicate_pct,
            "total_outlier_pct": p.total_outlier_pct,
            "schema_issues": p.schema_issues,
            "freshness_flag": p.freshness_flag,
            "freshness_days": p.freshness_days,
        },

        "columns": [
            {
                "name": cp.name,
                "dtype": cp.dtype,
                "missing_pct": cp.missing_pct,
                "missing_count": cp.missing_count,
                "unique_count": cp.unique_count,
                "outlier_count": cp.outlier_count,
                "outlier_pct": cp.outlier_pct,
                "format_issues": cp.format_issues,
                "mean": cp.mean,
                "std": cp.std,
                "min": cp.min,
                "max": cp.max,
                "q1": cp.q1,
                "median": cp.median,
                "q3": cp.q3,
                "top_values": cp.top_values,
            }
            for cp in p.columns
        ],

        "root_cause": {
            "total_issues": rc.total_issues,
            "high_severity": rc.high_severity,
            "medium_severity": rc.medium_severity,
            "low_severity": rc.low_severity,
            "summary": rc.summary,
            "issues": [
                {
                    "dimension": i.dimension,
                    "severity": i.severity,
                    "title": i.title,
                    "affected_column": i.affected_column,
                    "affected_rows": i.affected_rows,
                    "affected_pct": i.affected_pct,
                    "description": i.description,
                    "possible_root_causes": i.possible_root_causes,
                    "correlated_columns": i.correlated_columns,
                    "evidence": {
                        k: v for k, v in i.evidence.items()
                        if k in ("missing_count", "missing_pct", "outlier_count",
                                 "outlier_pct", "min", "max", "q1", "q3", "mean",
                                 "category_clusters", "date_clusters",
                                 "format_issue_count", "examples",
                                 "duplicate_count", "duplicate_pct",
                                 "schema_issues", "freshness_days", "freshness_flag")
                    },
                }
                for i in rc.issues
            ],
        },

        "impact": {
            "executive_summary": imp.executive_summary,
            "total_monetary_exposure": imp.total_monetary_exposure,
            "currency_symbol": imp.currency_symbol,
            "total_affected_rows": imp.total_affected_rows,
            "unique_affected_pct": imp.unique_affected_pct,
            "estimates": [
                {
                    "issue_title": e.issue_title,
                    "dimension": e.dimension,
                    "severity": e.severity,
                    "affected_rows": e.affected_rows,
                    "affected_pct": e.affected_pct,
                    "monetary_impact": e.monetary_impact,
                    "currency_symbol": e.currency_symbol,
                    "monetary_note": e.monetary_note,
                    "record_impact_description": e.record_impact_description,
                    "downstream_risks": e.downstream_risks,
                }
                for e in imp.estimates
            ],
        },

        "recommendations": {
            "remediation_plan": rr.remediation_plan,
            "estimated_score_lift": rr.estimated_score_lift,
            "items": [
                {
                    "priority": r.priority,
                    "action": r.action,
                    "detail": r.detail,
                    "dimension": r.dimension,
                    "effort": r.effort,
                    "impact": r.impact,
                    "issue_ref": r.issue_ref,
                }
                for r in rr.recommendations
            ],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>DataGuard AI</h1><p>Frontend not found.</p>")


@app.get("/api/status")
async def status():
    return ollama_status()


@app.get("/api/samples")
async def list_samples():
    samples = [
        {"filename": "customer_transactions.csv",
         "label": "Customer Transactions", "icon": "🛒",
         "desc": "Revenue outliers clustered in 2-day ingestion window · stale · duplicates"},
        {"filename": "hr_employee_records.csv",
         "label": "HR Employee Records", "icon": "👥",
         "desc": "Salary unit bug · missing performance scores · phone format issues"},
        {"filename": "hospital_patient_admissions.csv",
         "label": "Hospital Admissions", "icon": "🏥",
         "desc": "Heavy missingness · impossible ages · negative length-of-stay · empty column"},
        {"filename": "ecommerce_product_catalog.csv",
         "label": "E-Commerce Catalog", "icon": "🛍️",
         "desc": "Zero prices · negative stock · out-of-range ratings · fresh data"},
    ]
    return {"samples": [s for s in samples if (SAMPLE_DIR / s["filename"]).exists()]}


@app.post("/api/analyse")
async def analyse(file: UploadFile = File(...)):
    content = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(400, "File is empty.")

    p, qs = run_profile(df)
    a = run_analysis(df, p, qs)
    return _profile_to_dict(p, qs, a, file.filename)


@app.get("/api/analyse/sample/{filename}")
async def analyse_sample(filename: str):
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Sample not found.")
    df = pd.read_csv(path)
    p, qs = run_profile(df)
    a = run_analysis(df, p, qs)
    return _profile_to_dict(p, qs, a, filename)


@app.post("/api/narrative")
async def narrative(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith(".csv") else pd.read_excel(io.BytesIO(content))
    p, qs = run_profile(df)
    a = run_analysis(df, p, qs)
    text, was_ai = get_narrative(p, qs, a.root_cause, a.impact, a.recommendations, file.filename)
    return {"narrative": text, "was_ai": was_ai}


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)):
    """Return first 200 rows + column names for the Raw Data tab."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content)) if file.filename.endswith(".csv") else pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    preview_df = df.head(200).fillna("").astype(str)
    return {
        "columns": list(preview_df.columns),
        "rows": preview_df.values.tolist(),
        "total_rows": len(df),
    }


@app.get("/api/preview/sample/{filename}")
async def preview_sample(filename: str):
    """Return first 200 rows of a sample dataset for the Raw Data tab."""
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Sample not found.")
    df = pd.read_csv(path)
    preview_df = df.head(200).fillna("").astype(str)
    return {
        "columns": list(preview_df.columns),
        "rows": preview_df.values.tolist(),
        "total_rows": len(df),
    }


@app.get("/api/narrative/sample/{filename}")
async def narrative_sample(filename: str):
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Sample not found.")
    df = pd.read_csv(path)
    p, qs = run_profile(df)
    a = run_analysis(df, p, qs)
    text, was_ai = get_narrative(p, qs, a.root_cause, a.impact, a.recommendations, filename)
    return {"narrative": text, "was_ai": was_ai}


@app.post("/api/chat")
async def chat(request: dict):
    """
    Body: { message, history, analysis }
    Streams the LLM response as text/event-stream.
    """
    from dataguard_ai.core.profiler import ProfileResult, ColumnProfile
    from dataguard_ai.core.scorer import QualityScore
    from dataguard_ai.ai.root_cause import RootCauseReport, IssueInsight
    from dataguard_ai.ai.impact import BusinessImpactReport, ImpactEstimate
    from dataguard_ai.ai.recommendations import RecommendationReport, Recommendation

    message = request.get("message", "")
    history = request.get("history", [])
    data = request.get("analysis", {})

    if not message.strip():
        raise HTTPException(400, "Empty message.")
    if not data:
        raise HTTPException(400, "No analysis context provided.")

    # Reconstruct lightweight objects from JSON for the LLM context builder
    score_d = data.get("score", {})
    metrics_d = data.get("metrics", {})

    # Build minimal ProfileResult
    p = ProfileResult(
        total_rows=data.get("total_rows", 0),
        total_cols=data.get("total_cols", 0),
        duplicate_count=0,
        duplicate_pct=metrics_d.get("duplicate_pct", 0),
        completeness_score=score_d.get("completeness", 100),
        consistency_score=score_d.get("consistency", 100),
        uniqueness_score=score_d.get("uniqueness", 100),
        validity_score=score_d.get("validity", 100),
        freshness_days=metrics_d.get("freshness_days"),
        freshness_flag=metrics_d.get("freshness_flag", "UNKNOWN"),
        columns=[],
        total_outlier_rows=0,
        total_outlier_pct=metrics_d.get("total_outlier_pct", 0),
        schema_issues=metrics_d.get("schema_issues", []),
        missing_cells=0,
        missing_cell_pct=metrics_d.get("missing_cell_pct", 0),
    )

    qs = QualityScore(
        ai_readiness=score_d.get("ai_readiness", 0),
        completeness=score_d.get("completeness", 0),
        consistency=score_d.get("consistency", 0),
        uniqueness=score_d.get("uniqueness", 0),
        validity=score_d.get("validity", 0),
        freshness=score_d.get("freshness", 0),
        schema=score_d.get("schema", 0),
        grade=score_d.get("grade", "F"),
        verdict=score_d.get("verdict", ""),
        missing_pct=metrics_d.get("missing_cell_pct", 0),
        duplicate_pct=metrics_d.get("duplicate_pct", 0),
        outlier_pct=metrics_d.get("total_outlier_pct", 0),
        schema_issue_count=len(metrics_d.get("schema_issues", [])),
        freshness_days=metrics_d.get("freshness_days"),
        freshness_flag=metrics_d.get("freshness_flag", "UNKNOWN"),
    )

    rc_d = data.get("root_cause", {})
    issues = [
        IssueInsight(
            dimension=i["dimension"], severity=i["severity"],
            title=i["title"], affected_column=i.get("affected_column", ""),
            affected_rows=i.get("affected_rows", 0),
            affected_pct=i.get("affected_pct", 0),
            description=i.get("description", ""),
            evidence=i.get("evidence", {}),
            possible_root_causes=i.get("possible_root_causes", []),
            correlated_columns=i.get("correlated_columns", []),
        )
        for i in rc_d.get("issues", [])
    ]
    rc = RootCauseReport(
        issues=issues,
        total_issues=rc_d.get("total_issues", 0),
        high_severity=rc_d.get("high_severity", 0),
        medium_severity=rc_d.get("medium_severity", 0),
        low_severity=rc_d.get("low_severity", 0),
        summary=rc_d.get("summary", ""),
    )

    imp_d = data.get("impact", {})
    estimates = [
        ImpactEstimate(
            issue_title=e["issue_title"], dimension=e["dimension"],
            severity=e["severity"], affected_rows=e.get("affected_rows", 0),
            affected_pct=e.get("affected_pct", 0),
            monetary_impact=e.get("monetary_impact"),
            currency_symbol=e.get("currency_symbol", "$"),
            monetary_note=e.get("monetary_note", ""),
            record_impact_description=e.get("record_impact_description", ""),
            downstream_risks=e.get("downstream_risks", []),
            risk_level=e.get("severity", "LOW"),
        )
        for e in imp_d.get("estimates", [])
    ]
    imp = BusinessImpactReport(
        estimates=estimates,
        total_monetary_exposure=imp_d.get("total_monetary_exposure"),
        currency_symbol=imp_d.get("currency_symbol", "$"),
        total_affected_rows=imp_d.get("total_affected_rows", 0),
        unique_affected_pct=imp_d.get("unique_affected_pct", 0),
        executive_summary=imp_d.get("executive_summary", ""),
    )

    rr_d = data.get("recommendations", {})
    recs = [
        Recommendation(
            priority=r["priority"], action=r["action"],
            detail=r["detail"], dimension=r["dimension"],
            effort=r["effort"], impact=r["impact"],
            issue_ref=r.get("issue_ref", ""),
        )
        for r in rr_d.get("items", [])
    ]
    rr = RecommendationReport(
        recommendations=recs,
        remediation_plan=rr_d.get("remediation_plan", ""),
        estimated_score_lift=rr_d.get("estimated_score_lift", 0),
    )

    dataset_name = data.get("dataset_name", "dataset")

    def generate():
        for token in chat_stream(
            user_message=message,
            conversation_history=history,
            profile=p, quality_score=qs,
            root_cause=rc, impact=imp,
            recommendations=rr,
            dataset_name=dataset_name,
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("\n🛡️  DataGuard AI")
    print("   → http://localhost:8000\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True,
                reload_dirs=[str(Path(__file__).parent)])
