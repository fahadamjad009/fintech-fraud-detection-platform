from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FinTech Fraud Detection Platform", layout="wide")

st.title("FinTech Fraud Detection Platform — Baseline Model Monitor")
st.caption("Interactive monitoring dashboard: dataset profile, baseline metrics, PR/ROC, threshold tuning artifacts.")

REPORTS = Path("reports")
FIGS = REPORTS / "figures"
INTERACTIVE = REPORTS / "interactive"

METRICS_JSON = REPORTS / "baseline_metrics.json"
SWEEP_JSON = REPORTS / "baseline_threshold_sweep.json"
DATA_PARQUET = Path("data/processed/creditcard.parquet")


def load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text())
    return None


metrics = load_json(METRICS_JSON)
sweep = load_json(SWEEP_JSON)

# --- Sidebar ---
st.sidebar.header("Controls")
show_data_sample = st.sidebar.checkbox("Show data sample", value=False)
show_interactive_html = st.sidebar.checkbox("Show interactive HTML embeds", value=True)

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)

if metrics:
    c1.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    c3.metric("Fraud Recall @0.5", f"{tp/(tp+fn):.3f}" if (tp+fn) else "n/a")
    c4.metric("Fraud Precision @0.5", f"{tp/(tp+fp):.3f}" if (tp+fp) else "n/a")
else:
    c1.info("Run training to populate metrics.")
    c2.info("Run training to populate metrics.")
    c3.info("Run training to populate metrics.")
    c4.info("Run training to populate metrics.")

st.divider()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🎚️ Threshold Tuning", "🧾 Dataset Overview"])

with tab1:
    st.subheader("Model Performance")
    left, right = st.columns(2)

    pr_png = FIGS / "baseline_pr_curve.png"
    roc_png = FIGS / "baseline_roc_curve.png"
    score_png = FIGS / "baseline_score_distribution.png"

    if pr_png.exists():
        left.image(str(pr_png), caption="Precision-Recall Curve", use_container_width=True)
    if roc_png.exists():
        right.image(str(roc_png), caption="ROC Curve", use_container_width=True)

    if score_png.exists():
        st.image(str(score_png), caption="Score Distribution by Class", use_container_width=True)

    if show_interactive_html:
        st.markdown("### Interactive versions")
        for name in ["baseline_pr_curve", "baseline_roc_curve", "baseline_score_distribution"]:
            html_path = INTERACTIVE / f"{name}.html"
            if html_path.exists():
                st.components.v1.html(html_path.read_text(), height=520, scrolling=True)

with tab2:
    st.subheader("Threshold Tuning")
    sweep_png = FIGS / "baseline_threshold_sweep.png"
    if sweep_png.exists():
        st.image(str(sweep_png), caption="Precision/Recall vs Threshold", use_container_width=True)

    if sweep and "sweep" in sweep:
        df_sweep = pd.DataFrame(sweep["sweep"])
        st.dataframe(df_sweep, use_container_width=True, height=350)

        st.markdown("#### Find best threshold for a target precision")
        target_precision = st.slider("Target precision", 0.0, 1.0, 0.20, 0.01)
        candidates = df_sweep[df_sweep["precision"] >= target_precision].copy()
        if len(candidates) > 0:
            best = candidates.sort_values("recall", ascending=False).iloc[0]
            st.success(
                f"Suggested threshold ≈ {best['threshold']:.2f} | "
                f"precision={best['precision']:.3f}, recall={best['recall']:.3f}, "
                f"TP={int(best['tp'])}, FP={int(best['fp'])}, FN={int(best['fn'])}"
            )
        else:
            st.warning("No thresholds meet this precision target.")
    else:
        st.info("Run plot generation to create threshold sweep JSON.")

    if show_interactive_html:
        html_path = INTERACTIVE / "baseline_threshold_sweep.html"
        if html_path.exists():
            st.markdown("### Interactive threshold plot")
            st.components.v1.html(html_path.read_text(), height=520, scrolling=True)

with tab3:
    st.subheader("Dataset Overview (Processed)")
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
        st.write("Shape:", df.shape)
        vc = df["Class"].value_counts()
        st.write("Class distribution:", vc.to_dict())
        st.write("Fraud rate:", float(vc.get(1, 0)) / float(len(df)))

        if show_data_sample:
            st.dataframe(df.sample(25, random_state=42), use_container_width=True)
    else:
        st.info("Processed parquet not found. Run CSV→Parquet conversion.")
