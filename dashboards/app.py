"""
FinTech Fraud Detection Platform — Streamlit dashboard.

Consultant-tier monitoring & decision tool over the XGBoost champion model.
Read-only against precomputed artifacts in reports/ and monitoring/reports/.

Tabs:
  - Executive Summary  (hero charts, scenarios, status banner)
  - Business Impact    (live $-weighted metrics, threshold sweep)
  - Operations & Decile Lift
  - Model Performance
  - Segments
  - Monitoring & Drift
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MONITORING_REPORTS = Path("monitoring/reports")

BUSINESS_METRICS_JSON = REPORTS / "business_metrics.json"
BASELINE_METRICS_JSON = REPORTS / "baseline_metrics.json"
XGB_METRICS_JSON = REPORTS / "xgb_metrics.json"
COST_JSON = REPORTS / "cost_threshold_optimum.json"
DECILE_CSV = REPORTS / "decile_lift_table.csv"
DRIFT_CSV = MONITORING_REPORTS / "data_drift_report.csv"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data
def load_csv(p: Path):
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinTech Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — elevated card styling, status indicators, typography
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #0A0E13;
    --bg-card: #141A23;
    --bg-card-hover: #1A2230;
    --border: #2A3441;
    --accent: #00C896;
    --accent-amber: #FFD700;
    --accent-red: #FF4D4F;
    --text: #FAFAFA;
    --text-muted: #8B95A7;
}

/* Main app background */
.stApp {
    background: var(--bg);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #06090D;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--accent) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700 !important;
    margin-top: 1.2rem;
}

/* Title gradient */
h1 {
    background: linear-gradient(90deg, #00C896 0%, #00E5B0 50%, #00D4A8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

/* KPI metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg-card) 0%, #0F1620 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 18px 22px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    border-left-color: #00E5B0;
    box-shadow: 0 6px 16px rgba(0,200,150,0.15);
}

[data-testid="stMetricValue"] {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    line-height: 1.1 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600 !important;
    margin-bottom: 0.4rem;
}

[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border);
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 20px;
    font-weight: 600;
    color: var(--text-muted);
    background: transparent;
    border: none;
    font-size: 0.92rem;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
    background: rgba(0,200,150,0.05) !important;
}

/* Section headers */
h2, h3 {
    color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

/* Dividers */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}

/* Status banners */
.status-banner {
    padding: 14px 22px;
    border-radius: 10px;
    margin: 8px 0 16px 0;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.95rem;
    background: var(--bg-card);
}
.status-good   { border-color: var(--accent);       color: #B5F3DE; background: rgba(0,200,150,0.08); }
.status-warn   { border-color: var(--accent-amber); color: #FFE899; background: rgba(255,215,0,0.08); }
.status-alert  { border-color: var(--accent-red);   color: #FFC2C3; background: rgba(255,77,79,0.08); }
.status-icon { font-size: 1.4rem; }
.status-title { font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.75rem; }

/* Tag pills */
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-right: 6px;
}
.pill-green  { background: rgba(0,200,150,0.15);  color: #00C896; border: 1px solid #00C896; }
.pill-amber  { background: rgba(255,215,0,0.15);  color: #FFD700; border: 1px solid #FFD700; }
.pill-red    { background: rgba(255,77,79,0.15);  color: #FF4D4F; border: 1px solid #FF4D4F; }
.pill-muted  { background: rgba(139,149,167,0.15);color: #8B95A7; border: 1px solid #2A3441; }

/* Section card */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 14px;
}

/* Hero callout */
.hero-callout {
    background: linear-gradient(135deg, rgba(0,200,150,0.12) 0%, rgba(0,200,150,0.02) 100%);
    border: 1px solid rgba(0,200,150,0.3);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 18px 24px;
    margin: 8px 0;
}
.hero-callout-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 600; }
.hero-callout-value { font-size: 2.4rem; font-weight: 800; color: var(--accent); line-height: 1.1; margin: 4px 0; }
.hero-callout-detail { font-size: 0.85rem; color: var(--text-muted); }

/* Header pill row */
.header-pills { margin-top: -8px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
bm = load_json(BUSINESS_METRICS_JSON)
baseline_metrics = load_json(BASELINE_METRICS_JSON)
xgb_metrics_artifact = load_json(XGB_METRICS_JSON)
cost_opt_artifact = load_json(COST_JSON)
decile_df = load_csv(DECILE_CSV)
drift_df = load_csv(DRIFT_CSV)

if bm is None:
    st.error("Missing reports/business_metrics.json — run `python -m src.analytics.business_metrics`")
    st.stop()

sweep = bm["threshold_sweep"]
sweep_df = pd.DataFrame(sweep)

# ---------------------------------------------------------------------------
# Header with pill row
# ---------------------------------------------------------------------------
st.title("🛡️ FinTech Fraud Detection Platform")

pr_auc = xgb_metrics_artifact["pr_auc"] if xgb_metrics_artifact else 0
top_lift = decile_df.iloc[0]["lift"] if decile_df is not None else 0

st.markdown(f"""
<div class="header-pills">
<span class="pill pill-green">XGBoost Champion</span>
<span class="pill pill-muted">PR-AUC {pr_auc:.3f}</span>
<span class="pill pill-muted">{top_lift:.1f}× top-decile lift</span>
<span class="pill pill-muted">Kaggle CC Fraud</span>
<span class="pill pill-muted">Cost-based decisioning</span>
<span class="pill pill-muted">PSI + KS drift</span>
</div>
""", unsafe_allow_html=True)

st.caption(
    "XGBoost fraud detection on 284,807 transactions (0.17% fraud rate). End-to-end ML with "
    "$-weighted business metrics, threshold optimisation, decile lift, segmentation and drift monitoring. "
    "Drag the sidebar threshold to see live business impact."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Decision Controls")

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.00, max_value=1.00, value=0.97, step=0.01,
    help="Score ≥ threshold ⇒ flag as fraud.",
)
cost_fp_usd = st.sidebar.number_input(
    "Cost per false positive ($)",
    min_value=0.0, max_value=500.0, value=5.0, step=1.0,
    help="Investigation cost per alert.",
)
leaked_loss_multiplier = st.sidebar.number_input(
    "Leaked-fraud loss multiplier",
    min_value=0.0, max_value=10.0, value=1.0, step=0.1,
    help="Multiplier on leaked fraud $. 1.0 = face value, 1.5× includes chargeback fees.",
)

st.sidebar.header("Operations Capacity")
analyst_cases_per_hour = st.sidebar.number_input("Analyst cases per hour", min_value=1, max_value=100, value=12, step=1)
analyst_hours_per_day = st.sidebar.number_input("Analyst hours per day", min_value=1, max_value=24, value=8, step=1)
daily_alert_capacity = analyst_cases_per_hour * analyst_hours_per_day

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Holdout:** {bm['holdout_summary']['n_transactions']:,} txns · "
    f"{bm['holdout_summary']['n_fraud']} fraud · "
    f"${bm['holdout_summary']['fraud_volume_usd']:,.0f} fraud volume · "
    f"~{bm['holdout_summary']['time_window_hours_approx']:.1f}h window. "
    "Numbers are 20% test holdout, not annualized."
)

# ---------------------------------------------------------------------------
# Compute live metrics at chosen threshold
# ---------------------------------------------------------------------------
def get_sweep_point(thr: float) -> dict:
    idx = max(0, min(len(sweep) - 1, int(round(thr * 100))))
    return sweep[idx]

m = dict(get_sweep_point(threshold))
m["operational_cost_user"] = m["alert_volume"] * cost_fp_usd
m["leaked_loss_user"] = m["fraud_amount_leaked"] * leaked_loss_multiplier
m["net_benefit_user"] = m["fraud_amount_caught"] - m["operational_cost_user"] - m["leaked_loss_user"]

# Pre-compute optimal threshold under user's cost assumptions
sf = sweep_df.copy()
sf["op_cost_user"] = sf["alert_volume"] * cost_fp_usd
sf["leak_cost_user"] = sf["fraud_amount_leaked"] * leaked_loss_multiplier
sf["net_benefit_user"] = sf["fraud_amount_caught"] - sf["op_cost_user"] - sf["leak_cost_user"]
optimal_idx = sf["net_benefit_user"].idxmax()
optimal_threshold = float(sf.loc[optimal_idx, "threshold"])
optimal_net = float(sf.loc[optimal_idx, "net_benefit_user"])
gap = optimal_net - m["net_benefit_user"]

# ---------------------------------------------------------------------------
# Top KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("$ Catch Rate", f"{m['catch_rate_dollar']*100:.1f}%")
k2.metric("Fraud $ Caught", f"${m['fraud_amount_caught']:,.0f}",
          delta=f"of ${bm['holdout_summary']['fraud_volume_usd']:,.0f} at risk",
          delta_color="off")
k3.metric("Fraud $ Leaked", f"${m['fraud_amount_leaked']:,.0f}", delta_color="inverse")
k4.metric("Alerts", f"{int(m['alert_volume']):,}",
          delta=f"{m['alert_rate_pct']:.3f}% of traffic", delta_color="off")
k5.metric("Net Benefit", f"${m['net_benefit_user']:,.0f}")

# ---------------------------------------------------------------------------
# Status banner — color-coded by decision quality
# ---------------------------------------------------------------------------
if abs(threshold - optimal_threshold) <= 0.02:
    st.markdown(
        f"""<div class="status-banner status-good">
        <span class="status-icon">✓</span>
        <div><div class="status-title">Decision State: Optimal</div>
        Threshold <b>{threshold:.2f}</b> is within 0.02 of the cost-optimum under your assumptions. Net benefit <b>${m['net_benefit_user']:,.0f}</b>.</div>
        </div>""",
        unsafe_allow_html=True
    )
elif gap > 0 and gap / max(abs(optimal_net), 1) > 0.3:
    st.markdown(
        f"""<div class="status-banner status-alert">
        <span class="status-icon">▲</span>
        <div><div class="status-title">Decision State: Sub-Optimal</div>
        Current threshold <b>{threshold:.2f}</b> leaves <b>${gap:,.0f}</b> on the table. Switch to <b>{optimal_threshold:.2f}</b> for <b>${optimal_net:,.0f}</b> net benefit.</div>
        </div>""",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""<div class="status-banner status-warn">
        <span class="status-icon">●</span>
        <div><div class="status-title">Decision State: Near-Optimal</div>
        Threshold <b>{threshold:.2f}</b> vs optimal <b>{optimal_threshold:.2f}</b>. Net benefit gap: <b>${gap:,.0f}</b>.</div>
        </div>""",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
TAB_EXEC, TAB_BIZ, TAB_OPS, TAB_MODEL, TAB_SEG, TAB_DRIFT = st.tabs([
    "Executive Summary",
    "Business Impact",
    "Operations & Decile Lift",
    "Model Performance",
    "Segments",
    "Monitoring & Drift",
])

# ============================================================================
# TAB: Executive Summary — HERO TAB with charts upfront
# ============================================================================
with TAB_EXEC:
    # Hero callouts row
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        top = decile_df.iloc[0] if decile_df is not None else None
        if top is not None:
            st.markdown(f"""<div class="hero-callout">
            <div class="hero-callout-label">Top-decile capture</div>
            <div class="hero-callout-value">{top['pct_of_total_fraud_count']:.0f}%</div>
            <div class="hero-callout-detail">of all fraud captured by top 10% of scores · {top['lift']:.1f}× lift</div>
            </div>""", unsafe_allow_html=True)
    with hc2:
        st.markdown(f"""<div class="hero-callout">
        <div class="hero-callout-label">Cost-optimum net benefit</div>
        <div class="hero-callout-value">${bm['thresholds']['cost_optimum_0.97']['net_financial_benefit']:,.0f}</div>
        <div class="hero-callout-detail">at threshold 0.97 · ${bm['thresholds']['cost_optimum_0.97']['operational_cost']:,.0f} op cost</div>
        </div>""", unsafe_allow_html=True)
    with hc3:
        st.markdown(f"""<div class="hero-callout">
        <div class="hero-callout-label">Champion ROC-AUC</div>
        <div class="hero-callout-value">{xgb_metrics_artifact['roc_auc']:.3f}</div>
        <div class="hero-callout-detail">XGBoost · PR-AUC {xgb_metrics_artifact['pr_auc']:.3f} · Brier {bm['model_quality']['brier_score']:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Threshold trade-off curve")
    st.caption("Net financial benefit across every threshold (live, recalculated from your sidebar cost assumptions).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sf["threshold"], y=sf["net_benefit_user"],
        mode="lines", line=dict(color="#00C896", width=3),
        fill="tozeroy", fillcolor="rgba(0,200,150,0.08)",
        name="Net benefit",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#FAFAFA",
                  annotation_text=f"Current {threshold:.2f}", annotation_position="top right",
                  annotation_font_color="#FAFAFA")
    fig.add_vline(x=optimal_threshold, line_dash="dot", line_color="#FFD700",
                  annotation_text=f"Optimal {optimal_threshold:.2f}: ${optimal_net:,.0f}",
                  annotation_position="bottom left", annotation_font_color="#FFD700")
    fig.update_layout(
        xaxis_title="Decision threshold",
        yaxis_title="Net financial benefit ($)",
        height=380, hovermode="x unified", showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        xaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Headline scenarios")
    st.caption("Three operating points the platform supports — pick based on appetite for fraud loss vs analyst burden.")
    scenarios = bm["thresholds"]
    rows = []
    palette = {
        "high_recall_0.10": ("🟡 High Recall", "0.10"),
        "default_0.50":      ("🔵 Default", "0.50"),
        "cost_optimum_0.97": ("🟢 Cost Optimum", "0.97"),
    }
    for key, (label, thr_str) in palette.items():
        s = scenarios[key]
        rows.append({
            "Strategy": label,
            "Threshold": thr_str,
            "$ Catch Rate": f"{s['catch_rate_dollar']*100:.1f}%",
            "Fraud $ Caught": f"${s['fraud_amount_caught']:,.0f}",
            "Fraud $ Leaked": f"${s['fraud_amount_leaked']:,.0f}",
            "Alert Volume": f"{int(s['alert_volume']):,}",
            "Alert Rate": f"{s['alert_rate_pct']:.2f}%",
            "Precision": f"{s['precision_count']*100:.1f}%",
            "Net Benefit": f"${s['net_financial_benefit']:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Decile capture mini-chart
    if decile_df is not None:
        st.markdown("### Fraud concentration by score decile")
        st.caption("Industry-standard decile lift — top scores carry the bulk of fraud value.")
        d = decile_df.copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=d["decile"], y=d["pct_of_total_fraud_count"],
            marker_color=["#00C896" if i == 0 else "#1F4E3D" for i in range(len(d))],
            name="% of fraud (count)",
            text=d["pct_of_total_fraud_count"].map(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig2.update_layout(
            xaxis_title="Score decile (1 = highest scores)",
            yaxis_title="% of total fraud captured",
            height=340, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1F2937", dtick=1),
            yaxis=dict(gridcolor="#1F2937"),
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown("### Honest limitations")
    lim1, lim2 = st.columns(2)
    with lim1:
        st.markdown(
            "- **Benchmark data** — Kaggle Credit Card Fraud (PCA-anonymized V1-V28). Patterns are real, feature names are abstract.\n"
            "- **9.6-hour holdout** — test set covers ~9.6 hours of transactions. Numbers are not annualized.\n"
            "- **Drift is simulated** — the monitoring tab perturbs a copy of the dataset to demonstrate detection logic."
        )
    with lim2:
        st.markdown(
            "- **Cost params are configurable** — defaults ($5/FP, 1.0× leak) are illustrative. Adjust in sidebar.\n"
            "- **No real-time scoring on Cloud** — FastAPI endpoint runs locally; this dashboard is read-only against pre-computed artifacts.\n"
            "- **No external data joins** — single-table benchmark, not a multi-source pipeline."
        )

# ============================================================================
# TAB: Business Impact
# ============================================================================
with TAB_BIZ:
    st.markdown(f"### Live business metrics at threshold {threshold:.2f}")
    st.caption(f"Cost per FP = **${cost_fp_usd:.0f}** · Leak multiplier = **{leaked_loss_multiplier:.1f}×**")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Catch performance**")
        st.metric("Catch rate (count)", f"{m['catch_rate_count']*100:.1f}%")
        st.metric("Catch rate ($-weighted)", f"{m['catch_rate_dollar']*100:.1f}%")
        st.metric("Loss reduction vs no-model", f"{m['loss_reduction_pct']:.1f}%")
    with g2:
        st.markdown("**Dollar flow**")
        st.metric("Fraud $ at risk", f"${m['fraud_amount_total']:,.0f}")
        st.metric("Fraud $ caught", f"${m['fraud_amount_caught']:,.0f}")
        st.metric("Fraud $ leaked", f"${m['fraud_amount_leaked']:,.0f}")
    with g3:
        st.markdown("**Cost & benefit**")
        st.metric("Operational cost", f"${m['operational_cost_user']:,.0f}",
                  help=f"{int(m['alert_volume'])} alerts × ${cost_fp_usd:.0f}")
        st.metric("Leaked-fraud loss", f"${m['leaked_loss_user']:,.0f}",
                  help=f"${m['fraud_amount_leaked']:,.0f} × {leaked_loss_multiplier:.1f}×")
        st.metric("Net financial benefit", f"${m['net_benefit_user']:,.0f}")

    st.divider()
    st.markdown("### Confusion matrix at current threshold")
    cm_data = pd.DataFrame({
        "": ["Actual Fraud", "Actual Legit"],
        "Flagged": [int(m["tp"]), int(m["fp"])],
        "Not Flagged": [int(m["fn"]), int(m["tn"])],
    })
    st.dataframe(cm_data, hide_index=True, width="stretch")

    st.markdown(
        f"- **True positives (caught fraud):** {int(m['tp'])} cases worth ${m['fraud_amount_caught']:,.0f}\n"
        f"- **False positives (good customers flagged):** {int(m['fp'])}\n"
        f"- **False negatives (fraud leaked):** {int(m['fn'])} cases worth ${m['fraud_amount_leaked']:,.0f}\n"
        f"- **True negatives (correctly approved):** {int(m['tn']):,}"
    )

    st.markdown("### Customer experience")
    cust1, cust2 = st.columns(2)
    cust1.metric("Approval rate", f"{m['approval_rate_pct']:.3f}%")
    cust2.metric("Decline rate", f"{m['alert_rate_pct']:.3f}%")

# ============================================================================
# TAB: Operations & Decile Lift
# ============================================================================
with TAB_OPS:
    st.markdown("### Decile lift table")
    st.caption("Transactions sorted by predicted score, bucketed into 10 deciles. Industry-standard fraud table.")

    if decile_df is not None:
        d = decile_df.copy()
        display = pd.DataFrame({
            "Decile": d["decile"],
            "Txns": d["n_transactions"].map(lambda x: f"{x:,}"),
            "Score range": d.apply(lambda r: f"{r['min_score']:.4f} – {r['max_score']:.4f}", axis=1),
            "Fraud": d["fraud_count"],
            "Fraud $": d["fraud_amount"].map(lambda x: f"${x:,.0f}"),
            "% Fraud (count)": d["pct_of_total_fraud_count"].map(lambda x: f"{x:.1f}%"),
            "% Fraud ($)": d["pct_of_total_fraud_amount"].map(lambda x: f"{x:.1f}%"),
            "Cum (count)": d["cumulative_fraud_count_pct"].map(lambda x: f"{x:.1f}%"),
            "Cum ($)": d["cumulative_fraud_amount_pct"].map(lambda x: f"{x:.1f}%"),
            "Lift": d["lift"].map(lambda x: f"{x:.2f}×"),
        })
        st.dataframe(display, width="stretch", hide_index=True)

        st.markdown("### Cumulative gains")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0] + (d["decile"] * 10).tolist(),
            y=[0] + d["cumulative_fraud_count_pct"].tolist(),
            mode="lines+markers", line=dict(color="#00C896", width=3),
            name="Fraud captured (count)",
        ))
        fig.add_trace(go.Scatter(
            x=[0] + (d["decile"] * 10).tolist(),
            y=[0] + d["cumulative_fraud_amount_pct"].tolist(),
            mode="lines+markers", line=dict(color="#FFD700", width=3, dash="dash"),
            name="Fraud captured ($)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines",
            line=dict(color="#8B95A7", width=1, dash="dot"),
            name="Random",
        ))
        fig.update_layout(
            xaxis_title="% of population reviewed (top scores first)",
            yaxis_title="% of fraud captured",
            height=400, hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="#1F2937"),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Missing reports/decile_lift_table.csv")

    st.divider()
    st.markdown("### Alert volume & analyst capacity")

    alert_count = int(m["alert_volume"])
    holdout_hours = bm["holdout_summary"]["time_window_hours_approx"]
    daily_alerts = alert_count * (24 / holdout_hours) if holdout_hours > 0 else alert_count
    fte = daily_alerts / daily_alert_capacity if daily_alert_capacity > 0 else 0

    cap1, cap2, cap3, cap4 = st.columns(4)
    cap1.metric("Alerts in holdout", f"{alert_count:,}")
    cap2.metric("Daily-equiv alerts", f"{daily_alerts:.0f}")
    cap3.metric("Analyst capacity/day", f"{daily_alert_capacity:,}")
    cap4.metric("FTE required", f"{fte:.2f}")

    if fte > 1.0:
        st.markdown(f"""<div class="status-banner status-alert">
        <span class="status-icon">▲</span><div><div class="status-title">Capacity exceeded</div>
        Alert volume requires <b>{fte:.1f} FTE</b> to clear daily — raise threshold or grow team.</div></div>""", unsafe_allow_html=True)
    elif fte > 0:
        st.markdown(f"""<div class="status-banner status-good">
        <span class="status-icon">✓</span><div><div class="status-title">Capacity OK</div>
        Alert volume manageable for <b>less than 1 FTE</b>.</div></div>""", unsafe_allow_html=True)
    else:
        st.info("No alerts at this threshold.")

# ============================================================================
# TAB: Model Performance
# ============================================================================
with TAB_MODEL:
    st.markdown("### Champion vs Baseline")
    rows = []
    if baseline_metrics:
        rows.append({"Model": "Logistic Regression (baseline)",
                     "PR-AUC": f"{baseline_metrics['pr_auc']:.4f}",
                     "ROC-AUC": f"{baseline_metrics['roc_auc']:.4f}"})
    if xgb_metrics_artifact:
        rows.append({"Model": "XGBoost (champion)",
                     "PR-AUC": f"{xgb_metrics_artifact['pr_auc']:.4f}",
                     "ROC-AUC": f"{xgb_metrics_artifact['roc_auc']:.4f}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    mq = bm["model_quality"]
    q1, q2, q3 = st.columns(3)
    q1.metric("Brier Score (XGB)", f"{mq['brier_score']:.4f}", help="Lower is better. Probability calibration.")
    q2.metric("Score KS statistic", f"{mq['score_ks_statistic']:.4f}", help="Separation between fraud and legit score distributions.")
    q3.metric("KS p-value", f"{mq['score_ks_p_value']:.2e}", help="Probability the distributions are identical.")

    compare_png = FIGS / "model_comparison_auc.png"
    if compare_png.exists():
        st.image(str(compare_png), caption="AUC comparison", width="stretch")

    st.divider()
    st.markdown("### PR / ROC curves (Logistic Regression baseline)")
    c1, c2 = st.columns(2)
    pr_png = FIGS / "baseline_pr_curve.png"
    roc_png = FIGS / "baseline_roc_curve.png"
    if pr_png.exists(): c1.image(str(pr_png), width="stretch")
    if roc_png.exists(): c2.image(str(roc_png), width="stretch")

    score_png = FIGS / "baseline_score_distribution.png"
    if score_png.exists():
        st.image(str(score_png), caption="Score distribution by class", width="stretch")

    st.divider()
    st.markdown("### Threshold sweep & cost optimisation")
    c3, c4 = st.columns(2)
    sweep_png = FIGS / "baseline_threshold_sweep.png"
    cost_png = FIGS / "cost_curve_by_threshold.png"
    if sweep_png.exists(): c3.image(str(sweep_png), caption="Precision/Recall vs Threshold", width="stretch")
    if cost_png.exists(): c4.image(str(cost_png), caption="Expected cost across thresholds", width="stretch")

    if cost_opt_artifact:
        st.info(
            f"Pre-computed cost optimum: FN=${cost_opt_artifact['cost_config']['cost_fn']:.0f}, "
            f"FP=${cost_opt_artifact['cost_config']['cost_fp']:.0f} → threshold **{cost_opt_artifact['best_threshold']:.2f}**, "
            f"expected cost **${cost_opt_artifact['best_expected_cost']:,.0f}**."
        )

# ============================================================================
# TAB: Segments
# ============================================================================
with TAB_SEG:
    st.markdown("### Performance by transaction amount")
    st.caption("At threshold 0.97. Shows where the model is strong or weak by transaction size.")

    seg_amt = bm.get("segment_by_amount_at_optimum", [])
    if seg_amt:
        sa = pd.DataFrame(seg_amt)
        recall_pct = sa["recall_at_threshold"].apply(lambda x: x * 100 if x is not None else 0)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sa["segment"], y=recall_pct,
            marker_color=["#FF4D4F" if v < 50 else "#FFD700" if v < 80 else "#00C896" for v in recall_pct],
            text=recall_pct.map(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Amount segment", yaxis_title="Recall %",
            height=380, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="#1F2937"),
        )
        st.plotly_chart(fig, width="stretch")

        display = sa.copy()
        display["fraud_rate_bps"] = display["fraud_rate_bps"].map(lambda x: f"{x:.1f}")
        display["recall_at_threshold"] = display["recall_at_threshold"].map(
            lambda x: f"{x*100:.1f}%" if x is not None else "n/a")
        display["precision_at_threshold"] = display["precision_at_threshold"].map(
            lambda x: f"{x*100:.1f}%" if x is not None else "n/a")
        display["fraud_amount_caught"] = display["fraud_amount_caught"].map(lambda x: f"${x:,.0f}")
        display["fraud_amount_leaked"] = display["fraud_amount_leaked"].map(lambda x: f"${x:,.0f}")
        display.rename(columns={
            "segment": "Segment", "n_transactions": "Txns", "fraud_count": "Fraud",
            "fraud_rate_bps": "Fraud (bps)", "recall_at_threshold": "Recall",
            "precision_at_threshold": "Precision", "fraud_amount_caught": "$ Caught",
            "fraud_amount_leaked": "$ Leaked",
        }, inplace=True)
        st.dataframe(display, width="stretch", hide_index=True)

    st.divider()
    st.markdown("### Performance by hour-of-day")
    seg_hr = bm.get("segment_by_hour_at_optimum", [])
    if seg_hr:
        sh = pd.DataFrame(seg_hr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sh["hour"], y=sh["fraud_rate_bps"],
            mode="lines+markers", line=dict(color="#FFD700", width=3),
            marker=dict(size=8, color="#FFD700"),
            name="Fraud rate (bps)",
        ))
        fig.update_layout(
            xaxis_title="Hour of day", yaxis_title="Fraud rate (basis points)",
            height=340,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1F2937", dtick=2), yaxis=dict(gridcolor="#1F2937"),
        )
        st.plotly_chart(fig, width="stretch")

# ============================================================================
# TAB: Monitoring & Drift
# ============================================================================
with TAB_DRIFT:
    st.markdown("### Production drift monitoring (PSI + KS)")
    st.caption(
        "Compares the training reference distribution to a current (simulated) distribution. "
        "PSI: bin-shift magnitude. KS: distributional significance. In production, swap 'current' "
        "for a sliding window of live traffic."
    )

    if drift_df is not None:
        def severity(psi_val):
            try:
                p = float(psi_val)
            except (TypeError, ValueError):
                return "⚪ Unknown"
            if p > 0.25: return "🔴 Alert"
            if p > 0.10: return "🟡 Watch"
            return "🟢 Stable"

        d = drift_df.copy()
        d["severity"] = d["psi"].apply(severity)

        n_flagged = int(drift_df["drift_flag"].sum())
        n_total = len(drift_df)
        s1, s2, s3 = st.columns(3)
        s1.metric("Features monitored", n_total)
        s2.metric("Flagged", n_flagged)
        s3.metric("Stable", n_total - n_flagged)

        if n_flagged > 0:
            st.markdown(f"""<div class="status-banner status-warn">
            <span class="status-icon">●</span><div><div class="status-title">Drift detected</div>
            <b>{n_flagged}</b> of {n_total} features flagged. PSI: &lt;0.10 stable, 0.10–0.25 watch, &gt;0.25 alert. (Drift simulated — see Honest Limitations.)
            </div></div>""", unsafe_allow_html=True)

        d_display = d.copy()
        d_display["psi"] = d_display["psi"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
        d_display["ks_stat"] = d_display["ks_stat"].map(lambda x: f"{x:.4f}")
        d_display["ks_pvalue"] = d_display["ks_pvalue"].map(lambda x: f"{x:.2e}")
        st.dataframe(d_display[["severity", "feature", "psi", "ks_stat", "ks_pvalue", "drift_flag"]],
                     width="stretch", hide_index=True)

        st.markdown("### PSI by feature")
        chart = drift_df.copy()
        chart["psi_numeric"] = pd.to_numeric(chart["psi"], errors="coerce").fillna(0)
        chart = chart.sort_values("psi_numeric", ascending=True)
        colors = ["#FF4D4F" if v > 0.25 else "#FFD700" if v > 0.10 else "#00C896" for v in chart["psi_numeric"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart["psi_numeric"], y=chart["feature"],
            orientation="h", marker_color=colors,
        ))
        fig.add_vline(x=0.10, line_dash="dot", line_color="#FFD700")
        fig.add_vline(x=0.25, line_dash="dot", line_color="#FF4D4F")
        fig.update_layout(
            xaxis_title="Population Stability Index", yaxis_title="Feature",
            height=620, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="#1F2937"),
        )
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"FinTech Fraud Detection Platform · "
    f"[GitHub](https://github.com/fahadamjad009/fintech-fraud-detection-platform) · "
    f"Holdout: {bm['holdout_summary']['n_transactions']:,} txns · "
    f"{bm['holdout_summary']['fraud_rate_pct']:.3f}% fraud rate · "
    f"${bm['holdout_summary']['fraud_volume_usd']:,.0f} fraud volume"
)