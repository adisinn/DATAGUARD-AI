"""
DataGuard AI — Premium Dashboard
Run with:  streamlit run ui/app.py
"""
from __future__ import annotations

import io, sys, os, math, html as _html
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from dataguard_ai.core import run_profile
from dataguard_ai.ai import run_analysis
from dataguard_ai.ai.llm import chat_stream, ollama_status
from dataguard_ai.ai.narrative import get_narrative

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2d47 !important;
}

/* Hide default header */
[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* Main content padding */
[data-testid="stMainBlockContainer"] {
    padding: 0 2rem 4rem 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Tabs */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid #1e2d47 !important;
    gap: 0 !important;
}
button[data-testid="stTab"] {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.1rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
button[data-testid="stTab"]:hover {
    color: #94a3b8 !important;
}
button[data-testid="stTab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #60a5fa !important;
    background: transparent !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #0d1628 !important;
    border: 1px solid #1e2d47 !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2d47 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #0d1628 !important;
    border: 1px solid #1e2d47 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    transform: translateY(-1px) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1628 !important;
    border: 1px dashed #334155 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Alerts / info boxes */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
    font-size: 0.85rem !important;
}

/* Selectbox / multiselect */
[data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
    background: #0d1628 !important;
}
[data-baseweb="select"] > div {
    background: #0d1628 !important;
    border: 1px solid #1e2d47 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Divider */
hr { border-color: #1e2d47 !important; margin: 1.5rem 0 !important; }

/* Progress */
[data-testid="stProgressBar"] > div {
    background: #1e2d47 !important;
    border-radius: 4px !important;
    height: 6px !important;
}
[data-testid="stProgressBar"] > div > div {
    border-radius: 4px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d47; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* Sidebar text */
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def score_color(s: float) -> str:
    if s >= 85: return "#22d3ee"
    if s >= 70: return "#34d399"
    if s >= 55: return "#fbbf24"
    if s >= 40: return "#fb923c"
    return "#f87171"

def score_bg(s: float) -> str:
    if s >= 85: return "rgba(34,211,238,0.08)"
    if s >= 70: return "rgba(52,211,153,0.08)"
    if s >= 55: return "rgba(251,191,36,0.08)"
    if s >= 40: return "rgba(251,146,60,0.08)"
    return "rgba(248,113,113,0.08)"

def severity_chip(sev: str) -> str:
    cfg = {
        "HIGH":   ("#fca5a5", "#450a0a", "●"),
        "MEDIUM": ("#fde68a", "#451a03", "●"),
        "LOW":    ("#6ee7b7", "#022c22", "●"),
    }
    dot, bg, icon = cfg.get(sev, ("#94a3b8","#0f172a","●"))
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'background:{bg};color:{dot};border:1px solid {dot}22;'
        f'padding:2px 9px;border-radius:20px;font-size:0.7rem;font-weight:600;'
        f'letter-spacing:0.05em">{icon} {sev}</span>'
    )

def dim_icon(d: str) -> str:
    return {"completeness":"◈","validity":"⬡","consistency":"⟳",
            "uniqueness":"⊕","schema":"⊞","freshness":"◷","general":"⊙"}.get(d,"•")

def radial_gauge(score: float, size: int = 160) -> str:
    """Clean arc gauge in SVG."""
    color = score_color(score)
    bg_col = score_bg(score)
    cx = cy = size / 2
    r = size * 0.36
    stroke = size * 0.07
    # Arc: 225° to 315° = 270° sweep
    start, sweep = 225, 270
    end_angle = start + sweep * (score / 100)
    def pt(deg):
        a = math.radians(deg)
        return cx + r * math.cos(a), cy + r * math.sin(a)
    sx, sy = pt(start)
    ex, ey = pt(end_angle)
    bx, by = pt(start + sweep)
    lfg = 1 if sweep * score / 100 > 180 else 0
    grade = "A" if score>=85 else "B" if score>=70 else "C" if score>=55 else "D" if score>=40 else "F"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <defs>
        <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <circle cx="{cx}" cy="{cy}" r="{r+stroke/2+4}" fill="{bg_col}" opacity="0.6"/>
      <path d="M{sx:.2f},{sy:.2f} A{r:.2f},{r:.2f} 0 1 1 {bx:.2f},{by:.2f}"
            fill="none" stroke="#1e2d47" stroke-width="{stroke:.2f}" stroke-linecap="round"/>
      {'<path d="M'+f"{sx:.2f},{sy:.2f}"+' A'+f"{r:.2f},{r:.2f}"+' 0 '+str(lfg)+' 1 '+f"{ex:.2f},{ey:.2f}"+'"'
        +f' fill="none" stroke="{color}" stroke-width="{stroke:.2f}" stroke-linecap="round" filter="url(#glow)"/>'
        if score > 0 else ''}
      <text x="{cx}" y="{cy-2}" text-anchor="middle" font-family="Inter,sans-serif"
            font-size="{size*0.21:.0f}" font-weight="800" fill="{color}">{score:.0f}</text>
      <text x="{cx}" y="{cy+size*0.14:.0f}" text-anchor="middle" font-family="Inter,sans-serif"
            font-size="{size*0.095:.0f}" font-weight="600" fill="#475569">Grade {grade}</text>
    </svg>"""

def stat_card(label: str, value: str, sub: str = "", color: str = "#60a5fa", icon: str = "") -> str:
    return f"""
    <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:14px;
                padding:1.2rem 1.4rem;height:100%">
      <div style="color:#475569;font-size:0.7rem;font-weight:600;
                  letter-spacing:0.07em;text-transform:uppercase;margin-bottom:0.5rem">
        {icon} {label}
      </div>
      <div style="color:{color};font-size:1.85rem;font-weight:800;
                  letter-spacing:-0.03em;line-height:1">{value}</div>
      <div style="color:#475569;font-size:0.75rem;margin-top:0.4rem">{sub}</div>
    </div>"""

def mini_bar(value: float, color: str) -> str:
    pct = max(0, min(100, value))
    return f"""
    <div style="background:#1e2d47;border-radius:4px;height:6px;width:100%;overflow:hidden">
      <div style="background:{color};height:100%;width:{pct}%;border-radius:4px;
                  transition:width 0.6s ease"></div>
    </div>"""

def issue_card(icon: str, title: str, affected_pct: float, sev: str,
               description: str, causes: list[str], col_name: str) -> str:
    color = {"HIGH":"#f87171","MEDIUM":"#fbbf24","LOW":"#34d399"}.get(sev,"#94a3b8")
    bg    = {"HIGH":"rgba(248,113,113,0.06)","MEDIUM":"rgba(251,191,36,0.06)",
             "LOW":"rgba(52,211,153,0.06)"}.get(sev,"rgba(148,163,184,0.06)")
    safe_title = _html.escape(title)
    safe_desc  = _html.escape(description)
    causes_html = "".join(
        f'<li style="margin:3px 0;color:#94a3b8;font-size:0.8rem">{_html.escape(c)}</li>'
        for c in causes[:3]
    )
    return f"""
    <div style="background:{bg};border:1px solid {color}22;border-left:3px solid {color};
                border-radius:10px;padding:1rem 1.2rem;margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem">
        <div style="flex:1">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="font-size:1rem">{icon}</span>
            <span style="color:#e2e8f0;font-weight:600;font-size:0.9rem">{safe_title}</span>
            {severity_chip(sev)}
          </div>
          <p style="color:#94a3b8;font-size:0.8rem;margin:0 0 8px 0;line-height:1.5">{safe_desc}</p>
          <ul style="margin:0;padding-left:1rem">{causes_html}</ul>
        </div>
        <div style="text-align:right;min-width:80px">
          <div style="color:{color};font-size:1.4rem;font-weight:800">{affected_pct:.1f}%</div>
          <div style="color:#475569;font-size:0.68rem;text-transform:uppercase;
                      letter-spacing:0.05em">affected</div>
        </div>
      </div>
    </div>"""

def rec_card(priority: int, action: str, detail: str,
             effort: str, impact: str, dimension: str) -> str:
    effort_c = {"Low":"#34d399","Medium":"#fbbf24","High":"#f87171"}.get(effort,"#94a3b8")
    impact_c = {"High":"#f87171","Medium":"#fbbf24","Low":"#34d399"}.get(impact,"#94a3b8")
    num_bg   = {"High":"rgba(248,113,113,0.15)","Medium":"rgba(251,191,36,0.12)",
                "Low":"rgba(52,211,153,0.10)"}.get(impact,"rgba(96,165,250,0.10)")
    safe_action = _html.escape(action)
    safe_detail = _html.escape(detail[:200]) + ("…" if len(detail) > 200 else "")
    return f"""
    <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:12px;
                padding:1rem 1.2rem;margin-bottom:8px;display:flex;gap:1rem;align-items:flex-start">
      <div style="background:{num_bg};border-radius:8px;min-width:36px;height:36px;
                  display:flex;align-items:center;justify-content:center;
                  font-weight:800;color:{impact_c};font-size:0.9rem;flex-shrink:0">
        #{priority}
      </div>
      <div style="flex:1;min-width:0">
        <div style="font-weight:600;color:#e2e8f0;font-size:0.88rem;margin-bottom:4px">{safe_action}</div>
        <div style="color:#64748b;font-size:0.78rem;line-height:1.5;margin-bottom:8px">{safe_detail}</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <span style="background:rgba(52,211,153,0.1);color:{effort_c};border:1px solid {effort_c}33;
                       padding:1px 8px;border-radius:20px;font-size:0.68rem;font-weight:600">
            Effort: {effort}
          </span>
          <span style="background:rgba(248,113,113,0.1);color:{impact_c};border:1px solid {impact_c}33;
                       padding:1px 8px;border-radius:20px;font-size:0.68rem;font-weight:600">
            Impact: {impact}
          </span>
          <span style="background:rgba(96,165,250,0.1);color:#60a5fa;border:1px solid #60a5fa33;
                       padding:1px 8px;border-radius:20px;font-size:0.68rem;font-weight:600">
            {dimension}
          </span>
        </div>
      </div>
    </div>"""

def kv_row(k: str, v: str) -> str:
    return f"""<div style="display:flex;justify-content:space-between;padding:7px 0;
               border-bottom:1px solid #1e2d4744">
      <span style="color:#64748b;font-size:0.8rem">{k}</span>
      <span style="color:#e2e8f0;font-size:0.8rem;font-weight:500">{v}</span>
    </div>"""

def section_header(title: str, sub: str = "") -> str:
    return f"""
    <div style="margin:1.8rem 0 1rem 0">
      <h3 style="color:#f1f5f9;font-size:1.05rem;font-weight:700;margin:0 0 3px 0;
                 letter-spacing:-0.01em">{title}</h3>
      {'<p style="color:#475569;font-size:0.8rem;margin:0">'+sub+'</p>' if sub else ''}
    </div>"""

def horizontal_bar_chart(data: dict[str, float], color_fn=None, height: int = 36) -> str:
    if not data:
        return ""
    max_val = max(data.values()) or 1
    items = ""
    colors = ["#60a5fa","#818cf8","#a78bfa","#c084fc","#e879f9"]
    for i, (k, v) in enumerate(sorted(data.items(), key=lambda x: -x[1])):
        c = color_fn(v) if color_fn else colors[i % len(colors)]
        pct = v / max_val * 100
        items += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
          <div style="min-width:110px;color:#94a3b8;font-size:0.75rem;
                      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                      text-align:right">{k}</div>
          <div style="flex:1;background:#1e2d47;border-radius:3px;height:{height//2}px;overflow:hidden">
            <div style="background:{c};width:{pct:.1f}%;height:100%;border-radius:3px"></div>
          </div>
          <div style="min-width:42px;color:{c};font-size:0.75rem;font-weight:600;text-align:right">
            {int(v):,}
          </div>
        </div>"""
    return f'<div style="padding:0.5rem 0">{items}</div>'

def spark_mini(values: list[float], color: str = "#60a5fa", width: int = 120, height: int = 30) -> str:
    if len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    pts = []
    for i, v in enumerate(values):
        x = i / (len(values)-1) * width
        y = height - (v - mn) / rng * height
        pts.append(f"{x:.1f},{y:.1f}")
    path = "M" + " L".join(pts)
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
    </svg>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1rem 0">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.3rem">
        <span style="font-size:1.5rem">🛡️</span>
        <span style="color:#e2e8f0!important;font-size:1.1rem;font-weight:800;
                     letter-spacing:-0.02em">DataGuard AI</span>
      </div>
      <p style="color:#475569!important;font-size:0.75rem;margin:0">
        Data Reliability & AI-Readiness Analyst
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv","xlsx","xls"],
        help="CSV or Excel · up to 200 MB",
        label_visibility="collapsed",
    )

    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample_data", "customer_transactions.csv",
    )
    if uploaded is None:
        st.markdown("<p style='color:#475569;font-size:0.75rem;margin:0.5rem 0'>— or try a sample —</p>",
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🛒 Transactions", use_container_width=True):
                st.session_state["sample"] = "customer_transactions.csv"
            if st.button("🏥 Hospital", use_container_width=True):
                st.session_state["sample"] = "hospital_patient_admissions.csv"
        with col_b:
            if st.button("👥 HR Data", use_container_width=True):
                st.session_state["sample"] = "hr_employee_records.csv"
            if st.button("🛍️ Products", use_container_width=True):
                st.session_state["sample"] = "ecommerce_product_catalog.csv"

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#334155!important;line-height:1.8">
      <div style="color:#475569!important;font-weight:600;text-transform:uppercase;
                  letter-spacing:0.06em;font-size:0.65rem;margin-bottom:0.5rem">How it works</div>
      <div>① Upload CSV / Excel</div>
      <div>② AI profiles 6 quality dimensions</div>
      <div>③ Root-cause investigation</div>
      <div>④ Business impact quantified</div>
      <div>⑤ Prioritised remediation plan</div>
      <div>⑥ Ask the AI analyst anything</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Ollama status indicator ───────────────────────────────────────────────
    st.divider()
    _status = ollama_status()
    if _status["available"]:
        _model_label = _status.get("model","")
        _is_custom   = _status.get("is_custom", False)
        _badge_color = "#34d399" if _is_custom else "#fbbf24"
        _badge_bg    = "rgba(52,211,153,0.08)" if _is_custom else "rgba(251,191,36,0.06)"
        _badge_border= "#34d39933" if _is_custom else "#fbbf2433"
        _badge_icon  = "✦" if _is_custom else "●"
        _badge_text  = "CUSTOM MODEL ACTIVE" if _is_custom else "AI ANALYST ONLINE"
        st.markdown(f"""
        <div style="background:{_badge_bg};border:1px solid {_badge_border};
                    border-radius:8px;padding:0.6rem 0.8rem">
          <div style="color:{_badge_color};font-size:0.7rem;font-weight:600">
            {_badge_icon} {_badge_text}
          </div>
          <div style="color:#475569;font-size:0.65rem;margin-top:2px">
            {_model_label}
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(251,191,36,0.06);border:1px solid #fbbf2433;
                    border-radius:8px;padding:0.6rem 0.8rem">
          <div style="color:#fbbf24;font-size:0.7rem;font-weight:600">⚡ AI ANALYST OFFLINE</div>
          <div style="color:#475569;font-size:0.65rem;margin-top:2px">{_status['reason']}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_analyse(raw: bytes, filename: str):
    df = pd.read_csv(io.BytesIO(raw)) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    p, qs = run_profile(df)
    a = run_analysis(df, p, qs)
    return df, p, qs, a

df = p = qs = analysis = None

# Resolve which file to use — sidebar upload takes priority, then landing upload, then sample
_active_upload = uploaded or st.session_state.pop("landing_file", None)

if _active_upload:
    with st.spinner("🔍 Profiling your dataset…"):
        df, p, qs, analysis = load_and_analyse(_active_upload.read(), _active_upload.name)
elif st.session_state.get("sample"):
    fname = st.session_state["sample"]
    fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "sample_data", fname)
    if os.path.exists(fpath):
        with st.spinner(f"🔍 Loading {fname}…"):
            with open(fpath,"rb") as f: raw = f.read()
            df, p, qs, analysis = load_and_analyse(raw, fname)


# ─────────────────────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────────────────────
if df is None:
    st.markdown("""
    <div style="padding:5rem 0 2rem 0;text-align:center">
      <div style="display:inline-flex;align-items:center;gap:10px;
                  background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.2);
                  border-radius:100px;padding:6px 18px;font-size:0.78rem;font-weight:500;
                  color:#60a5fa;margin-bottom:1.5rem;letter-spacing:0.04em">
        ✦ &nbsp; AI-POWERED DATA RELIABILITY
      </div>
      <h1 style="font-size:clamp(2.2rem,5vw,3.8rem);font-weight:800;color:#f8fafc;
                 letter-spacing:-0.03em;line-height:1.1;margin:0 0 1rem 0">
        Know if your data<br>
        <span style="background:linear-gradient(135deg,#60a5fa,#818cf8,#c084fc);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent">
          can be trusted
        </span>
      </h1>
      <p style="font-size:1.05rem;color:#64748b;max-width:560px;margin:0 auto 2.5rem;
                line-height:1.65">
        Upload any CSV or Excel file. DataGuard AI profiles it across six quality 
        dimensions, identifies root causes, quantifies business impact, and delivers 
        an actionable remediation plan — in seconds.
      </p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("◈", "#60a5fa", "Deep Profiling",
         "7 quality dimensions analysed per column — completeness, validity, consistency, uniqueness, schema, freshness."),
        ("⬡", "#818cf8", "AI Root-Cause",
         "Detects ingestion windows, source-system clusters, and correlated failures — not just 'outliers exist'."),
        ("◈", "#c084fc", "Business Impact",
         "Monetary exposure estimated against your actual value columns. Real numbers, not vague risk labels."),
        ("✓", "#34d399", "Action Plan",
         "Prioritised recommendations ranked by Impact × Effort so you fix what matters first."),
    ]
    for col, (icon, color, title, body) in zip([f1,f2,f3,f4], features):
        with col:
            st.markdown(f"""
            <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:14px;
                        padding:1.4rem;height:100%">
              <div style="font-size:1.3rem;margin-bottom:0.7rem;color:{color}">{icon}</div>
              <div style="font-weight:700;color:#e2e8f0;font-size:0.9rem;
                          margin-bottom:0.5rem">{title}</div>
              <div style="color:#475569;font-size:0.78rem;line-height:1.55">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload & sample buttons directly on landing page ─────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(96,165,250,0.06),rgba(129,140,248,0.06));
                border:1px solid #1e2d47;border-radius:16px;padding:2rem 2rem 1.5rem 2rem">
      <div style="color:#e2e8f0;font-weight:700;font-size:1rem;margin-bottom:0.4rem">
        Upload your dataset to get started
      </div>
      <div style="color:#475569;font-size:0.82rem;margin-bottom:1.2rem">
        CSV or Excel · any size · all processing happens locally on your machine
      </div>
    </div>
    """, unsafe_allow_html=True)

    # File uploader embedded in landing
    land_upload = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key="landing_upload",
    )
    if land_upload:
        st.session_state["landing_file"] = land_upload

    st.markdown("""
    <div style="text-align:center;color:#334155;font-size:0.78rem;margin:1rem 0 0.5rem 0">
      — or try one of the sample datasets —
    </div>""", unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    samples = [
        (sc1, "🛒", "Transactions", "customer_transactions.csv",
         "2,070 rows · revenue outliers, missing values, duplicates"),
        (sc2, "👥", "HR Records",   "hr_employee_records.csv",
         "1,605 rows · salary anomalies, mixed phone formats"),
        (sc3, "🏥", "Hospital",     "hospital_patient_admissions.csv",
         "1,800 rows · age outliers, empty columns, stale data"),
        (sc4, "🛍️", "E-Commerce",  "ecommerce_product_catalog.csv",
         "1,200 rows · zero prices, negative stock, bad ratings"),
    ]
    for col, icon, label, fname, desc in samples:
        with col:
            st.markdown(f"""
            <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:12px;
                        padding:1rem;text-align:center;margin-bottom:6px">
              <div style="font-size:1.4rem">{icon}</div>
              <div style="color:#e2e8f0;font-weight:600;font-size:0.82rem;margin:4px 0 2px">{label}</div>
              <div style="color:#475569;font-size:0.68rem;line-height:1.4">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Load {label}", key=f"land_{fname}", use_container_width=True):
                st.session_state["sample"] = fname
                st.rerun()

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# TOP HEADER BAR
# ─────────────────────────────────────────────────────────────────────────────
fname_display = (uploaded.name if uploaded else st.session_state.get("sample","dataset.csv"))
ai_color = score_color(qs.ai_readiness)

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:1.2rem 0 1rem 0;border-bottom:1px solid #1e2d47;
            margin-bottom:1.5rem;flex-wrap:wrap;gap:1rem">
  <div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
      <span style="font-size:1.2rem">🛡️</span>
      <h1 style="font-size:1.25rem;font-weight:800;color:#f1f5f9;
                 margin:0;letter-spacing:-0.02em">DataGuard AI</h1>
      <span style="background:rgba(96,165,250,0.12);color:#60a5fa;border:1px solid #60a5fa33;
                   padding:2px 10px;border-radius:20px;font-size:0.68rem;font-weight:600;
                   letter-spacing:0.05em">DATA HEALTH REPORT</span>
    </div>
    <div style="color:#475569;font-size:0.78rem">
      📄 &nbsp;<span style="color:#64748b">{fname_display}</span>
      &nbsp;·&nbsp; {p.total_rows:,} rows &nbsp;·&nbsp; {p.total_cols} columns
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <div style="text-align:right">
      <div style="color:#475569;font-size:0.65rem;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.07em">AI Readiness</div>
      <div style="color:{ai_color};font-size:1.8rem;font-weight:800;
                  line-height:1;letter-spacing:-0.03em">{qs.ai_readiness:.0f}
        <span style="font-size:0.9rem;color:#475569">/100</span>
      </div>
    </div>
    <div style="background:{score_bg(qs.ai_readiness)};border:1px solid {ai_color}33;
                border-radius:10px;padding:6px 14px;text-align:center">
      <div style="color:{ai_color};font-size:1.5rem;font-weight:900">
        {("A" if qs.ai_readiness>=85 else "B" if qs.ai_readiness>=70 else "C" if qs.ai_readiness>=55 else "D" if qs.ai_readiness>=40 else "F")}
      </div>
      <div style="color:#475569;font-size:0.6rem;font-weight:600;text-transform:uppercase">Grade</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAT STRIP
# ─────────────────────────────────────────────────────────────────────────────
sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
stats = [
    (sc1, "Completeness", f"{qs.completeness:.0f}", "%", score_color(qs.completeness)),
    (sc2, "Validity",     f"{qs.validity:.0f}",     "%", score_color(qs.validity)),
    (sc3, "Uniqueness",   f"{qs.uniqueness:.0f}",   "%", score_color(qs.uniqueness)),
    (sc4, "Consistency",  f"{qs.consistency:.0f}",  "%", score_color(qs.consistency)),
    (sc5, "Missing",      f"{p.missing_cell_pct:.1f}", "%",
     "#f87171" if p.missing_cell_pct>10 else "#fbbf24" if p.missing_cell_pct>3 else "#34d399"),
    (sc6, "Duplicates",   f"{p.duplicate_pct:.1f}", "%",
     "#f87171" if p.duplicate_pct>5 else "#fbbf24" if p.duplicate_pct>1 else "#34d399"),
]
for col, label, val, unit, clr in stats:
    with col:
        st.markdown(f"""
        <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:12px;
                    padding:0.9rem 1rem;text-align:center">
          <div style="color:#475569;font-size:0.65rem;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px">{label}</div>
          <div style="color:{clr};font-size:1.55rem;font-weight:800;line-height:1">
            {val}<span style="font-size:0.8rem;color:#475569">{unit}</span>
          </div>
          {mini_bar(float(val), clr)}
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📊  Score & Dimensions",
    "🔬  Column Intelligence",
    "🔎  Issues & Root Cause",
    "💰  Business Impact",
    "✅  Remediation Plan",
    "📋  Raw Data",
    "🤖  AI Analyst",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Score
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown(f"""
        <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:16px;
                    padding:1.5rem;text-align:center">
          <div style="color:#475569;font-size:0.7rem;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.07em;margin-bottom:1rem">
            AI Readiness Score
          </div>
          <div style="display:flex;justify-content:center">
            {radial_gauge(qs.ai_readiness, 200)}
          </div>
          <div style="margin-top:0.8rem;color:{score_color(qs.ai_readiness)};
                      font-size:0.88rem;font-weight:600">{qs.verdict}</div>
          <hr style="border-color:#1e2d47;margin:1rem 0">
          {kv_row("Missing cells", f"{p.missing_cell_pct:.2f}%")}
          {kv_row("Duplicate rows", f"{p.duplicate_pct:.2f}%")}
          {kv_row("Outlier rows", f"{p.total_outlier_pct:.2f}%")}
          {kv_row("Schema issues", str(len(p.schema_issues)))}
          {kv_row("Freshness", p.freshness_flag + (" · " + str(int(p.freshness_days)) + "d" if p.freshness_days is not None else ""))}
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(section_header("Dimension Breakdown", "Weighted contributions to the AI Readiness score"),
                    unsafe_allow_html=True)

        dims = [
            ("Completeness", qs.completeness,  0.30, "100 − % missing cells across all columns"),
            ("Validity",     qs.validity,       0.20, "100 − % outlier rows (IQR × 3 method)"),
            ("Uniqueness",   qs.uniqueness,     0.15, "100 − % fully duplicate rows"),
            ("Consistency",  qs.consistency,    0.15, "Format anomaly penalty across string columns"),
            ("Freshness",    qs.freshness,      0.10, f"Data age: {int(p.freshness_days or 0)}d · Status: {p.freshness_flag}"),
            ("Schema",       qs.schema,         0.10, f"{len(p.schema_issues)} structural issue(s) detected"),
        ]
        for name, val, weight, note in dims:
            clr = score_color(val)
            st.markdown(f"""
            <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:10px;
                        padding:0.85rem 1rem;margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:6px">
                <div>
                  <span style="color:#e2e8f0;font-weight:600;font-size:0.85rem">{name}</span>
                  <span style="color:#334155;font-size:0.7rem;margin-left:8px">weight {weight:.0%}</span>
                </div>
                <div style="display:flex;align-items:center;gap:8px">
                  <span style="color:#475569;font-size:0.75rem">{note}</span>
                  <span style="color:{clr};font-weight:800;font-size:1rem;min-width:36px;text-align:right">{val:.0f}</span>
                </div>
              </div>
              {mini_bar(val, clr)}
            </div>""", unsafe_allow_html=True)

        if p.schema_issues:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(section_header("❌ Schema Issues"), unsafe_allow_html=True)
            for iss in p.schema_issues:
                st.markdown(f"""
                <div style="background:rgba(248,113,113,0.06);border:1px solid #f8717133;
                            border-radius:8px;padding:0.7rem 1rem;margin-bottom:6px;
                            color:#fca5a5;font-size:0.82rem">⚠ {iss}</div>
                """, unsafe_allow_html=True)

        if p.freshness_flag == "STALE":
            st.markdown(f"""
            <div style="background:rgba(251,191,36,0.06);border:1px solid #fbbf2433;
                        border-radius:8px;padding:0.7rem 1rem;margin-top:8px;
                        color:#fde68a;font-size:0.82rem">
              ⏰ &nbsp;Data is <strong>{int(p.freshness_days or 0)} days</strong> old.
              Freshness SLA may be violated.
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Column Intelligence
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown(section_header("Column Quality Matrix",
        f"Every column profiled across {p.total_cols} dimensions"), unsafe_allow_html=True)

    # Build rich table HTML
    rows_html = ""
    for cp in p.columns:
        miss_c  = "#f87171" if cp.missing_pct>20 else "#fbbf24" if cp.missing_pct>5 else "#34d399"
        out_c   = "#f87171" if cp.outlier_pct>10 else "#fbbf24" if cp.outlier_pct>2 else "#34d399"
        fmt_c   = "#fbbf24" if cp.format_issues>0 else "#34d399"

        status = "✓"
        status_c = "#34d399"
        if cp.missing_pct>20 or cp.outlier_pct>10:
            status, status_c = "✗", "#f87171"
        elif cp.missing_pct>5 or cp.outlier_pct>2 or cp.format_issues>0:
            status, status_c = "!", "#fbbf24"

        miss_bar  = mini_bar(cp.missing_pct, miss_c)
        out_bar   = mini_bar(cp.outlier_pct, out_c) if cp.outlier_count else ""
        safe_name = _html.escape(cp.name)  # prevent HTML injection from column names

        rows_html += f"""
        <tr style="border-bottom:1px solid #1e2d4744">
          <td style="padding:8px 12px;color:{status_c};font-weight:700;font-size:0.85rem">{status}</td>
          <td style="padding:8px 12px;color:#e2e8f0;font-size:0.82rem;font-weight:500">{safe_name}</td>
          <td style="padding:8px 12px">
            <span style="background:#1e2d47;color:#64748b;padding:2px 7px;
                         border-radius:4px;font-size:0.68rem;font-family:monospace">{cp.dtype}</span>
          </td>
          <td style="padding:8px 12px">
            <div style="color:{miss_c};font-size:0.78rem;font-weight:600;margin-bottom:2px">{cp.missing_pct:.1f}%</div>
            {miss_bar}
          </td>
          <td style="padding:8px 12px;color:#64748b;font-size:0.78rem">{cp.unique_count:,}</td>
          <td style="padding:8px 12px">
            {'<div style="color:'+out_c+';font-size:0.78rem;font-weight:600;margin-bottom:2px">'+str(cp.outlier_count)+' ('+str(cp.outlier_pct)+'%)</div>'+out_bar if cp.outlier_count else '<span style="color:#334155;font-size:0.75rem">—</span>'}
          </td>
          <td style="padding:8px 12px">
            {'<span style="color:'+fmt_c+';font-size:0.78rem">'+str(cp.format_issues)+' issues</span>' if cp.format_issues else '<span style="color:#334155;font-size:0.75rem">—</span>'}
          </td>
          <td style="padding:8px 12px;color:#64748b;font-size:0.78rem">
            {"₹ " + f"{cp.mean:,.0f}" if cp.mean is not None else "—"}
          </td>
        </tr>"""

    st.markdown(f"""
    <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:14px;overflow:hidden">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="background:#0a1020;border-bottom:1px solid #1e2d47">
            <th style="padding:10px 12px;color:#475569;font-size:0.68rem;font-weight:600;
                       text-transform:uppercase;letter-spacing:0.07em;text-align:left">St</th>
            <th style="padding:10px 12px;color:#475569;font-size:0.68rem;font-weight:600;
                       text-transform:uppercase;letter-spacing:0.07em;text-align:left">Column</th>
            <th style="padding:10px 12px;color:#475569;font-size:0.68rem;font-weight:600;
                       text-transform:uppercase;letter-spacing:0.07em;text-align:left">Type</th>
            <th style="padding:10px 12px;color:#475569;font-size:0.68rem;font-weight:600;
                       text-transform:uppercase;letter-spacing:0.07em;text-align:left">Missing</th>
            <th style="padding:10px 12px;color:#475569;font-size:0.68rem;font-weight:600;
                       text-transform:uppercase;letter-spacing:0.07em;text-align:left">Unique</th>
            <th style="padding:10px 12px;color:#475569;font-size:0.68rem;font-weight:600;
                       text-transform:uppercase;letter-spacing:0.07em;text-align:left">Outliers</th>
            <th style="padding:10px 12px;color:#475569;font-size:0.68rem;font-weight:600;
                       text-transform:uppercase;letter-spacing:0.07em;text-align:left">Format</th>
            <th style="padding:10px 12px;color:#475569;font-size:0.68rem;font-weight:600;
                       text-transform:uppercase;letter-spacing:0.07em;text-align:left">Mean</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Numeric stats
    num_cols = [cp for cp in p.columns if cp.mean is not None]
    if num_cols:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("Numeric Column Statistics"), unsafe_allow_html=True)
        for cp in num_cols:
            rng_val = (cp.max or 0) - (cp.min or 0)
            norm_q1  = max(0, min(100, ((cp.q1  or 0) - (cp.min or 0)) / rng_val * 100)) if rng_val else 25
            norm_q3  = max(0, min(100, ((cp.q3  or 0) - (cp.min or 0)) / rng_val * 100)) if rng_val else 75
            norm_med = max(0, min(100, ((cp.median or 0) - (cp.min or 0)) / rng_val * 100)) if rng_val else 50
            bar_width = max(0, norm_q3 - norm_q1)  # clamp negative widths (q1==q3 edge case)
            out_c = "#f87171" if cp.outlier_pct>5 else "#fbbf24" if cp.outlier_pct>1 else "#34d399"
            safe_col_name = _html.escape(cp.name)
            st.markdown(f"""
            <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:12px;
                        padding:1rem 1.2rem;margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:10px;flex-wrap:wrap;gap:8px">
                <span style="color:#e2e8f0;font-weight:600;font-size:0.88rem">{safe_col_name}</span>
                <div style="display:flex;gap:16px;flex-wrap:wrap">
                  {kv_row("min",f"{cp.min:,.2f}") if cp.min is not None else ""}
                  {kv_row("mean",f"{cp.mean:,.2f}") if cp.mean is not None else ""}
                  {kv_row("median",f"{cp.median:,.2f}") if cp.median is not None else ""}
                  {kv_row("max",f"{cp.max:,.2f}") if cp.max is not None else ""}
                </div>
              </div>
              <div style="position:relative;height:20px;background:#1e2d47;
                          border-radius:6px;overflow:hidden">
                <div style="position:absolute;left:{norm_q1:.1f}%;width:{bar_width:.1f}%;
                             height:100%;background:rgba(96,165,250,0.25)"></div>
                <div style="position:absolute;left:{norm_med:.1f}%;width:2px;
                             height:100%;background:#60a5fa"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:4px">
                <span style="color:#334155;font-size:0.68rem">{f"{cp.min:,.0f}" if cp.min is not None else "0"}</span>
                <span style="color:#60a5fa;font-size:0.68rem">Q1–Q3 range</span>
                <span style="color:#334155;font-size:0.68rem">{f"{cp.max:,.0f}" if cp.max is not None else "0"}</span>
              </div>
              {'<div style="color:'+out_c+';font-size:0.75rem;margin-top:6px">⚠ '+str(cp.outlier_count)+' outliers ('+str(cp.outlier_pct)+'%)</div>' if cp.outlier_count else ''}
            </div>""", unsafe_allow_html=True)

    # Categorical distributions
    cat_cols_list = [cp for cp in p.columns if cp.top_values]
    if cat_cols_list:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("Categorical Distributions"), unsafe_allow_html=True)
        n_cat_cols = min(3, len(cat_cols_list))
        cat_grid = st.columns(n_cat_cols)
        for i, cp in enumerate(cat_cols_list[:6]):
            with cat_grid[i % n_cat_cols]:
                colors_list = ["#60a5fa","#818cf8","#a78bfa","#c084fc","#e879f9"]
                bars = ""
                max_v = max(cp.top_values.values()) if cp.top_values else 1
                for j, (k, v) in enumerate(list(cp.top_values.items())[:5]):
                    pct = v / max_v * 100
                    c = colors_list[j % len(colors_list)]
                    bars += f"""
                    <div style="margin-bottom:5px">
                      <div style="display:flex;justify-content:space-between;
                                  margin-bottom:2px;font-size:0.72rem">
                        <span style="color:#94a3b8;overflow:hidden;white-space:nowrap;
                                     text-overflow:ellipsis;max-width:120px">{k}</span>
                        <span style="color:{c};font-weight:600">{v:,}</span>
                      </div>
                      <div style="background:#1e2d47;border-radius:3px;height:4px">
                        <div style="background:{c};width:{pct:.1f}%;height:100%;border-radius:3px"></div>
                      </div>
                    </div>"""
                st.markdown(f"""
                <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:12px;
                            padding:1rem;margin-bottom:10px">
                  <div style="color:#e2e8f0;font-weight:600;font-size:0.82rem;
                              margin-bottom:10px;overflow:hidden;text-overflow:ellipsis;
                              white-space:nowrap">{cp.name}</div>
                  {bars}
                  <div style="color:#334155;font-size:0.68rem;margin-top:6px">
                    {cp.unique_count:,} unique values · {cp.missing_pct:.1f}% missing
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Issues & Root Cause
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    rc = analysis.root_cause

    # Summary banner
    sev_color = "#f87171" if rc.high_severity > 0 else "#fbbf24" if rc.medium_severity > 0 else "#34d399"
    st.markdown(f"""
    <div style="background:rgba(96,165,250,0.05);border:1px solid #1e2d47;
                border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1.5rem">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <div style="display:flex;gap:20px">
          <div style="text-align:center">
            <div style="color:#f87171;font-size:1.5rem;font-weight:800">{rc.high_severity}</div>
            <div style="color:#475569;font-size:0.65rem;text-transform:uppercase;
                        letter-spacing:0.05em">High</div>
          </div>
          <div style="text-align:center">
            <div style="color:#fbbf24;font-size:1.5rem;font-weight:800">{rc.medium_severity}</div>
            <div style="color:#475569;font-size:0.65rem;text-transform:uppercase;
                        letter-spacing:0.05em">Medium</div>
          </div>
          <div style="text-align:center">
            <div style="color:#34d399;font-size:1.5rem;font-weight:800">{rc.low_severity}</div>
            <div style="color:#475569;font-size:0.65rem;text-transform:uppercase;
                        letter-spacing:0.05em">Low</div>
          </div>
        </div>
        <div style="flex:1;color:#94a3b8;font-size:0.82rem;line-height:1.5;
                    border-left:1px solid #1e2d47;padding-left:16px">
          <strong style="color:#e2e8f0">AI Investigation: </strong>{rc.summary}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not rc.issues:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#34d399">
          ✓ &nbsp; No significant data quality issues detected.
        </div>""", unsafe_allow_html=True)
    else:
        for issue in rc.issues:
            icon = dim_icon(issue.dimension)
            st.markdown(
                issue_card(icon, issue.title, issue.affected_pct, issue.severity,
                           issue.description, issue.possible_root_causes, issue.affected_column),
                unsafe_allow_html=True,
            )

            # Evidence details in expander
            ev = issue.evidence
            has_extra = ev.get("category_clusters") or ev.get("date_clusters")
            if has_extra:
                with st.expander("🔬 View evidence details", expanded=False):
                    if ev.get("date_clusters"):
                        st.markdown("**📅 Ingestion window clustering:**")
                        for dc, info in ev["date_clusters"].items():
                            st.markdown(f"""
                            <div style="background:#0d1628;border:1px solid #1e2d47;
                                        border-radius:8px;padding:0.8rem;font-size:0.8rem;
                                        color:#94a3b8;margin-bottom:8px">
                              Column <code style="color:#60a5fa">{dc}</code>:
                              outliers cluster between
                              <strong style="color:#e2e8f0">{info['from']}</strong> →
                              <strong style="color:#e2e8f0">{info['to']}</strong>
                              ({info['window_days']} days · {info['outlier_count_in_window']} records)
                            </div>""", unsafe_allow_html=True)
                    if ev.get("category_clusters"):
                        st.markdown("**📦 Source-category concentration:**")
                        for cat, info in ev["category_clusters"].items():
                            st.markdown(f"""
                            <div style="background:#0d1628;border:1px solid #1e2d47;
                                        border-radius:8px;padding:0.8rem;font-size:0.8rem;
                                        color:#94a3b8;margin-bottom:8px">
                              <code style="color:#60a5fa">{cat}</code> = 
                              <strong style="color:#fbbf24">'{info['top_value']}'</strong>:
                              {info['concentration_pct']}% of outliers originate here
                              ({info['outlier_count']} / {info['base_count']} records)
                            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Business Impact
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    imp = analysis.impact

    # Hero metrics
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(stat_card(
            "Monetary Exposure",
            f"{imp.currency_symbol}{imp.total_monetary_exposure:,.0f}" if imp.total_monetary_exposure else "N/A",
            "estimated across all issues",
            "#f87171" if (imp.total_monetary_exposure or 0)>0 else "#34d399",
            "💰",
        ), unsafe_allow_html=True)
    with h2:
        st.markdown(stat_card(
            "Affected Records",
            f"{imp.total_affected_rows:,}",
            f"{imp.unique_affected_pct:.1f}% of dataset",
            "#fbbf24",
            "📊",
        ), unsafe_allow_html=True)
    with h3:
        high_risk = sum(1 for e in imp.estimates if e.risk_level == "HIGH")
        st.markdown(stat_card(
            "High-Risk Issues",
            str(high_risk),
            "require immediate attention",
            "#f87171" if high_risk > 0 else "#34d399",
            "⚠️",
        ), unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(96,165,250,0.04);border:1px solid #1e2d47;
                border-radius:12px;padding:1rem 1.4rem;margin:1rem 0 1.5rem 0;
                color:#94a3b8;font-size:0.85rem;line-height:1.6">
      <strong style="color:#e2e8f0">Executive Summary: </strong>{imp.executive_summary}
    </div>""", unsafe_allow_html=True)

    for est in imp.estimates:
        clr  = {"HIGH":"#f87171","MEDIUM":"#fbbf24","LOW":"#34d399"}.get(est.risk_level,"#94a3b8")
        bg   = {"HIGH":"rgba(248,113,113,0.04)","MEDIUM":"rgba(251,191,36,0.04)",
                "LOW":"rgba(52,211,153,0.04)"}.get(est.risk_level,"rgba(148,163,184,0.04)")

        risks_html = "".join(
            f'<div style="display:flex;gap:6px;align-items:flex-start;margin-bottom:4px">'
            f'<span style="color:#475569;margin-top:1px">▸</span>'
            f'<span style="color:#94a3b8;font-size:0.78rem">{r}</span></div>'
            for r in est.downstream_risks
        )
        mon_html = (
            f'<div style="color:{clr};font-size:0.82rem;font-weight:600;margin-top:6px">'
            f'💰 &nbsp;{est.currency_symbol}{est.monetary_impact:,.0f} estimated exposure'
            f'<span style="color:#475569;font-weight:400"> — {est.monetary_note}</span></div>'
            if est.monetary_impact else ""
        )
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {clr}22;border-radius:12px;
                    padding:1.1rem 1.3rem;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap">
            <div style="flex:1">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                <span style="color:#e2e8f0;font-weight:600;font-size:0.88rem">{est.issue_title}</span>
                {severity_chip(est.risk_level)}
              </div>
              <div style="color:#94a3b8;font-size:0.8rem;line-height:1.5;margin-bottom:8px">
                {est.record_impact_description}
              </div>
              {mon_html}
              <div style="margin-top:10px">{risks_html}</div>
            </div>
            <div style="text-align:right;min-width:90px">
              <div style="color:{clr};font-size:1.6rem;font-weight:800">{est.affected_rows:,}</div>
              <div style="color:#475569;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em">records</div>
              <div style="color:{clr};font-size:0.95rem;font-weight:600;margin-top:4px">{est.affected_pct:.1f}%</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Recommendations
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    rr = analysis.recommendations
    quick_wins = [r for r in rr.recommendations if r.effort=="Low" and r.impact=="High"]

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(stat_card("Total Actions", str(len(rr.recommendations)),
                    "prioritised by impact × effort", "#60a5fa", "✅"), unsafe_allow_html=True)
    with r2:
        st.markdown(stat_card("⚡ Quick Wins", str(len(quick_wins)),
                    "low effort · high impact", "#34d399", "⚡"), unsafe_allow_html=True)
    with r3:
        st.markdown(stat_card("Score Lift", f"+{rr.estimated_score_lift:.0f}",
                    "estimated pts if HIGH issues fixed", "#818cf8", "📈"), unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(129,140,248,0.05);border:1px solid #1e2d47;
                border-radius:12px;padding:1rem 1.4rem;margin:1rem 0 1.5rem 0;
                color:#94a3b8;font-size:0.85rem;line-height:1.6">
      <strong style="color:#e2e8f0">Remediation Plan: </strong>{rr.remediation_plan}
    </div>""", unsafe_allow_html=True)

    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        effort_f = st.multiselect("Filter by Effort", ["Low","Medium","High"],
                                   default=["Low","Medium","High"], key="ef")
    with fc2:
        impact_f = st.multiselect("Filter by Impact", ["High","Medium","Low"],
                                   default=["High","Medium","Low"], key="if")

    filtered = [r for r in rr.recommendations if r.effort in effort_f and r.impact in impact_f]

    if not filtered:
        st.markdown('<div style="color:#475569;text-align:center;padding:2rem">No results for selected filters.</div>',
                    unsafe_allow_html=True)
    else:
        for rec in filtered:
            st.markdown(
                rec_card(rec.priority, rec.action, rec.detail, rec.effort, rec.impact, rec.dimension),
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Raw Data
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    st.markdown(section_header(
        f"Dataset Preview",
        f"{p.total_rows:,} rows × {p.total_cols} columns — showing first 500 rows"
    ), unsafe_allow_html=True)

    st.dataframe(
        df.head(500),
        use_container_width=True,
        height=480,
    )

    dl1, dl2 = st.columns([1,4])
    with dl1:
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name="dataguard_export.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AI Analyst Chat (Option A + C)
# ══════════════════════════════════════════════════════════════════════════════
with t7:
    _llm_status = ollama_status()

    # ── Narrative banner (Option A) ───────────────────────────────────────────
    fname_display_chat = (
        uploaded.name if uploaded else st.session_state.get("sample", "dataset.csv")
    )

    # Cache narrative in session state so it doesn't re-generate on every interaction
    narrative_key = f"narrative_{fname_display_chat}"
    if narrative_key not in st.session_state:
        st.session_state[narrative_key] = None

    col_narr, col_btn = st.columns([5, 1])
    with col_narr:
        st.markdown(section_header(
            "🤖 AI Analyst",
            "Powered by Ollama — runs 100% locally, no data leaves your machine"
        ), unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='margin-top:1.6rem'>", unsafe_allow_html=True)
        if st.button("✨ Generate Report Narrative", use_container_width=True):
            with st.spinner("AI is writing the executive narrative…"):
                narrative_text, was_ai = get_narrative(
                    p, qs,
                    analysis.root_cause,
                    analysis.impact,
                    analysis.recommendations,
                    dataset_name=fname_display_chat,
                )
                st.session_state[narrative_key] = (narrative_text, was_ai)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state[narrative_key]:
        narrative_text, was_ai = st.session_state[narrative_key]
        ai_badge = (
            '<span style="background:rgba(52,211,153,0.12);color:#34d399;'
            'border:1px solid #34d39933;padding:2px 8px;border-radius:20px;'
            'font-size:0.65rem;font-weight:600;letter-spacing:0.05em">✦ AI GENERATED</span>'
            if was_ai else
            '<span style="background:rgba(96,165,250,0.1);color:#60a5fa;'
            'border:1px solid #60a5fa33;padding:2px 8px;border-radius:20px;'
            'font-size:0.65rem;font-weight:600;letter-spacing:0.05em">RULE-BASED FALLBACK</span>'
        )
        st.markdown(f"""
        <div style="background:#0d1628;border:1px solid #1e2d47;border-radius:14px;
                    padding:1.4rem 1.6rem;margin-bottom:1.5rem">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem">
            <span style="color:#e2e8f0;font-size:0.8rem;font-weight:600">Executive Narrative</span>
            {ai_badge}
          </div>
          <div style="color:#94a3b8;font-size:0.85rem;line-height:1.75">
            {_html.escape(narrative_text).replace(chr(10), '<br>').replace('**', '<strong style=&quot;color:#e2e8f0&quot;>').replace('**', '</strong>')}
          </div>
        </div>
        """, unsafe_allow_html=True)
        # Cleaner markdown rendering
        with st.expander("View as formatted text", expanded=False):
            st.markdown(narrative_text)

    st.divider()

    # ── AI Status Banner ──────────────────────────────────────────────────────
    if not _llm_status["available"]:
        st.markdown(f"""
        <div style="background:rgba(251,191,36,0.06);border:1px solid #fbbf2433;
                    border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1.5rem">
          <div style="color:#fde68a;font-weight:600;font-size:0.88rem;margin-bottom:6px">
            ⚡ AI Analyst Offline
          </div>
          <div style="color:#94a3b8;font-size:0.82rem;line-height:1.6">
            {_llm_status['reason']}<br>
            <br>
            <strong style="color:#e2e8f0">To enable:</strong><br>
            1. Open the <strong style="color:#e2e8f0">Ollama app</strong> on your Mac
               (or run <code style="color:#60a5fa">ollama serve</code> in a terminal)<br>
            2. Pull a model: <code style="color:#60a5fa">ollama pull llama3.2</code><br>
            3. Refresh this page — the AI Analyst will activate automatically
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat Interface (Option C) ─────────────────────────────────────────────
    st.markdown(section_header(
        "Chat with your data",
        "Ask anything about the dataset quality, issues, or what to do next"
    ), unsafe_allow_html=True)

    # Initialise chat history per dataset
    chat_key = f"chat_{fname_display_chat}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Suggested questions
    if not st.session_state[chat_key]:
        st.markdown("""
        <div style="margin-bottom:1rem">
          <div style="color:#475569;font-size:0.72rem;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">
            Suggested questions
          </div>
        </div>""", unsafe_allow_html=True)

        suggestions = [
            "Is this data safe to use for machine learning?",
            "What is causing the missing values?",
            "Which issues have the highest business impact?",
            "What should I fix first?",
            "Explain the outliers in simple terms",
            "How would these issues affect a revenue dashboard?",
        ]
        s_cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with s_cols[i % 3]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state[chat_key].append(
                        {"role": "user", "content": suggestion}
                    )
                    st.rerun()

    # Render conversation history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state[chat_key]:
            is_user = msg["role"] == "user"
            bubble_bg  = "#1e2d47" if is_user else "#0d1628"
            bubble_border = "#334155" if is_user else "#1e2d47"
            align = "flex-end" if is_user else "flex-start"
            label_color = "#64748b"
            label = "You" if is_user else "DataGuard AI"
            label_icon = "👤" if is_user else "🤖"
            safe_content = _html.escape(msg["content"]).replace("\n", "<br>")
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin-bottom:12px">
              <div style="max-width:85%">
                <div style="color:{label_color};font-size:0.65rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.06em;
                            margin-bottom:4px;{'text-align:right' if is_user else ''}">
                  {label_icon} {label}
                </div>
                <div style="background:{bubble_bg};border:1px solid {bubble_border};
                            border-radius:{'12px 12px 4px 12px' if is_user else '12px 12px 12px 4px'};
                            padding:0.8rem 1rem;color:#e2e8f0;font-size:0.85rem;line-height:1.6">
                  {safe_content}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Streaming response handler ────────────────────────────────────────────
    # Check if the last message is from the user (needs a response)
    if st.session_state[chat_key] and st.session_state[chat_key][-1]["role"] == "user":
        last_user_msg = st.session_state[chat_key][-1]["content"]
        # History passed to LLM excludes the last user message (it's sent separately)
        history_for_llm = st.session_state[chat_key][:-1]

        with st.spinner("🤖 Thinking…"):
            full_response = ""
            response_placeholder = st.empty()

            if _llm_status["available"]:
                for token in chat_stream(
                    user_message=last_user_msg,
                    conversation_history=history_for_llm,
                    profile=p,
                    quality_score=qs,
                    root_cause=analysis.root_cause,
                    impact=analysis.impact,
                    recommendations=analysis.recommendations,
                    dataset_name=fname_display_chat,
                ):
                    full_response += token
                    # Live streaming display
                    response_placeholder.markdown(f"""
                    <div style="background:#0d1628;border:1px solid #1e2d47;
                                border-radius:12px 12px 12px 4px;padding:0.8rem 1rem;
                                color:#94a3b8;font-size:0.85rem;line-height:1.6;
                                margin-bottom:12px">
                      {_html.escape(full_response).replace(chr(10),'<br>')}▌
                    </div>""", unsafe_allow_html=True)
            else:
                full_response = (
                    "The AI analyst is currently offline. Start the Ollama app and "
                    f"pull a model (`ollama pull llama3.2`) to enable chat. "
                    f"\n\nIn the meantime, you can explore the Issues, Impact, and "
                    f"Recommendations tabs for the full analysis."
                )

            response_placeholder.empty()
            st.session_state[chat_key].append(
                {"role": "assistant", "content": full_response}
            )
            st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    with st.form(key="chat_form", clear_on_submit=True):
        inp_col, btn_col = st.columns([8, 1])
        with inp_col:
            user_input = st.text_input(
                "Ask a question",
                placeholder="e.g. Why are there missing values in the email column?",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state[chat_key].append(
            {"role": "user", "content": user_input.strip()}
        )
        st.rerun()

    # Clear chat button
    if st.session_state[chat_key]:
        if st.button("🗑 Clear conversation", key="clear_chat"):
            st.session_state[chat_key] = []
            st.rerun()
