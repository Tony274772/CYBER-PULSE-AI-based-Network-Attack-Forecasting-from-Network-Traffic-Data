"""GT-RSSM Streamlit Dashboard (Section 7.11).

Provides a visual interface for network attack forecasting:
  1. File upload (PCAP or CSV)
  2. Infiltration probability timeline with forecast
  3. Attack-stage ribbon
  4. Network graph view with attention weights
  5. Explainability panel (SHAP + natural-language explanation)
  6. Anomaly badge for unseen techniques
  7. Export button for summary report

Usage:
    streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config — must be the first Streamlit command
st.set_page_config(
    page_title="GT-RSSM | Network Attack Forecasting",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Styling                                                                     #
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
    /* Dark glassmorphism theme */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    /* Custom metric cards */
    .metric-card {
        background: rgba(30, 40, 60, 0.7);
        border: 1px solid rgba(88, 166, 255, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, border-color 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(88, 166, 255, 0.4);
    }
    .metric-card h3 {
        font-size: 0.85rem;
        color: #8b949e;
        margin: 0 0 0.3rem 0;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .value.danger {
        background: linear-gradient(90deg, #f85149, #da3633);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Alert badge */
    .anomaly-badge {
        background: linear-gradient(135deg, #f85149 0%, #da3633 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .safe-badge {
        background: linear-gradient(135deg, #3fb950 0%, #238636 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c9d1d9;
        border-bottom: 1px solid rgba(88, 166, 255, 0.2);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    h1 {
        background: linear-gradient(90deg, #58a6ff, #3fb950, #58a6ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Hide default Streamlit styling noise */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Stage colour palette                                                        #
# --------------------------------------------------------------------------- #
STAGE_COLORS = {
    "Benign": "#3fb950",
    "Reconnaissance": "#58a6ff",
    "Initial_Access": "#d2a8ff",
    "Lateral_Movement": "#f0883e",
    "Command_And_Control": "#f778ba",
    "Exfiltration": "#ff7b72",
    "Impact_DoS": "#f85149",
}

ALERT_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Header                                                                      #
# --------------------------------------------------------------------------- #
st.markdown("# 🛡️ GT-RSSM  ·  Network Attack Forecasting")
st.markdown("##### *GAT Spatial Encoder + Recurrent State-Space Model — World-model-based forecasting for NTRO*")
st.divider()


# --------------------------------------------------------------------------- #
# Sidebar — File upload + settings                                            #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📁 Upload Network Data")
    uploaded = st.file_uploader(
        "Upload a PCAP or CSV file",
        type=["pcap", "pcapng", "csv"],
        help="CIC-IDS2017 flow CSV or any network capture PCAP"
    )
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    k_horizon = st.selectbox("Forecast Horizon (K steps)", [5, 10, 20], index=1)
    alert_thresh = st.slider("Alert Threshold", 0.0, 1.0, ALERT_THRESHOLD, 0.05)
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **GT-RSSM** is a world-model-based network attack forecaster.
    It uses graph attention to encode host-pair interactions and a
    recurrent state-space model to predict future attack states
    *before* they materialize.

    - 🔍 Dense Graph Attention (no PyG)
    - 🧠 Dreamer-v1 RSSM (Gaussian latents)
    - 📈 K-step imagination forecast
    - 🎯 MITRE ATT&CK stage mapping
    """)


# --------------------------------------------------------------------------- #
# Main area — analysis results                                                #
# --------------------------------------------------------------------------- #
if uploaded is None:
    # Landing state
    st.markdown("## 👈 Upload a file to begin analysis")
    st.markdown("""
    Upload a **CIC-IDS2017 flow CSV** or a **PCAP capture file** to run the
    GT-RSSM forecasting pipeline. The model will:

    1. **Parse** network flows and construct per-window interaction graphs
    2. **Encode** host-pair interactions via dense masked graph attention
    3. **Forecast** future infiltration probability via RSSM imagination
    4. **Explain** which features and hosts are driving the prediction

    All processing runs **fully offline** — no network calls, no external APIs.
    """)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><h3>Model</h3>
            <div class="value">GT-RSSM</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><h3>Stages</h3>
            <div class="value">7</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><h3>Window</h3>
            <div class="value">5s</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><h3>Horizon</h3>
            <div class="value">100s</div></div>""", unsafe_allow_html=True)
    st.stop()

# ---- Run inference --------------------------------------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

try:
    from src.inference.predict import predict

    with st.spinner("🔄 Running GT-RSSM inference pipeline..."):
        result = predict(tmp_path, k_steps=(k_horizon,))
except Exception as e:
    st.error(f"**Inference error:** {e}")
    st.info("Make sure the model checkpoint exists at `checkpoints/gt_rssm_v1.pt` "
            "or `checkpoints/stage1.pt`. Run training first.")
    st.stop()
finally:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

if not result.get("windows"):
    st.warning("No network windows could be extracted from the uploaded file.")
    st.stop()

windows = result["windows"]
infil_timeline = result["infiltration_timeline"]
stage_timeline = result["stage_timeline"]
forecast = result.get("forecast", {})
kl_surprise = result.get("kl_surprise", [])
stage_order = result.get("stage_order", [])
attention_by_window = result.get("attention_by_window", {})

# ---- Summary metrics -------------------------------------------------------
n_windows = len(windows)
max_prob = max(w["prob"] for w in infil_timeline) if infil_timeline else 0
mean_kl = float(np.mean([k["kl"] for k in kl_surprise])) if kl_surprise else 0
attack_windows = sum(1 for w in infil_timeline if w["prob"] >= alert_thresh)

col1, col2, col3, col4 = st.columns(4)
with col1:
    danger = ' danger' if attack_windows > 0 else ''
    st.markdown(f"""<div class="metric-card"><h3>Windows Analyzed</h3>
        <div class="value">{n_windows}</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><h3>Peak Infiltration Prob</h3>
        <div class="value{danger}">{max_prob:.3f}</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><h3>Attack Windows</h3>
        <div class="value{danger}">{attack_windows}</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card"><h3>Mean KL Surprise</h3>
        <div class="value">{mean_kl:.3f}</div></div>""", unsafe_allow_html=True)

st.markdown("")

# ---- Anomaly badge ---------------------------------------------------------
# Use top-1% of KL as threshold (from validation, or heuristic)
kl_values = [k["kl"] for k in kl_surprise]
kl_threshold = float(np.percentile(kl_values, 99)) if kl_values else 1.0
anomalous_windows = [k for k in kl_surprise if k["kl"] > kl_threshold]

if anomalous_windows:
    st.markdown(f'<div class="anomaly-badge">⚠️ Possible unseen technique detected — '
                f'{len(anomalous_windows)} window(s) with abnormally high surprise (KL > {kl_threshold:.2f})</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="safe-badge">✅ No anomalous patterns detected beyond known attack types</div>',
                unsafe_allow_html=True)

st.markdown("")

# ---- 1. Infiltration Timeline + Forecast -----------------------------------
st.markdown('<div class="section-header">📈 Infiltration Probability Timeline</div>',
            unsafe_allow_html=True)

fig_timeline = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                              shared_xaxes=True, vertical_spacing=0.05)

# Observed infiltration probability
x_obs = [w["window_id"] for w in infil_timeline]
y_obs = [w["prob"] for w in infil_timeline]
fig_timeline.add_trace(
    go.Scatter(x=x_obs, y=y_obs, mode="lines+markers", name="Observed",
               line=dict(color="#58a6ff", width=2),
               marker=dict(size=4, color="#58a6ff")),
    row=1, col=1
)

# Forecast (dashed)
if forecast:
    for K, data in forecast.items():
        forecast_probs = data.get("infiltration_probs", [])
        if forecast_probs:
            x_fc = list(range(x_obs[-1] + 1, x_obs[-1] + 1 + len(forecast_probs)))
            fig_timeline.add_trace(
                go.Scatter(x=x_fc, y=forecast_probs, mode="lines+markers",
                           name=f"Forecast K={K}",
                           line=dict(color="#d2a8ff", width=2, dash="dash"),
                           marker=dict(size=4, color="#d2a8ff")),
                row=1, col=1
            )

# Alert threshold band
fig_timeline.add_hline(y=alert_thresh, line_dash="dot", line_color="#f85149",
                        annotation_text=f"Alert threshold ({alert_thresh})",
                        annotation_font_color="#f85149", row=1, col=1)
fig_timeline.add_hrect(y0=alert_thresh, y1=1.0, fillcolor="#f85149",
                        opacity=0.08, line_width=0, row=1, col=1)

# KL surprise trace
x_kl = [k["window_id"] for k in kl_surprise]
y_kl = [k["kl"] for k in kl_surprise]
fig_timeline.add_trace(
    go.Scatter(x=x_kl, y=y_kl, mode="lines", name="KL Surprise",
               line=dict(color="#f0883e", width=1.5),
               fill="tozeroy", fillcolor="rgba(240,136,62,0.1)"),
    row=2, col=1
)

fig_timeline.update_layout(
    height=450, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=20, t=30, b=30),
)
fig_timeline.update_yaxes(title_text="P(infiltration)", row=1, col=1, range=[0, 1.05])
fig_timeline.update_yaxes(title_text="KL", row=2, col=1)
fig_timeline.update_xaxes(title_text="Window ID", row=2, col=1)
st.plotly_chart(fig_timeline, use_container_width=True)

# ---- 2. Attack-Stage Ribbon -----------------------------------------------
st.markdown('<div class="section-header">🎯 MITRE ATT&CK Stage Ribbon</div>',
            unsafe_allow_html=True)

stage_x = [s["window_id"] for s in stage_timeline]
stage_names = [s["predicted_stage"] for s in stage_timeline]
stage_colors_mapped = [STAGE_COLORS.get(s, "#8b949e") for s in stage_names]

# Add forecast stages
if forecast:
    for K, data in forecast.items():
        stage_probs_list = data.get("stage_probs", [])
        for i, sp in enumerate(stage_probs_list):
            wid = stage_x[-1] + i + 1 if stage_x else i
            best_stage = max(sp, key=sp.get) if sp else "Benign"
            stage_x.append(wid)
            stage_names.append(best_stage)
            stage_colors_mapped.append(STAGE_COLORS.get(best_stage, "#8b949e"))

fig_ribbon = go.Figure()
fig_ribbon.add_trace(go.Bar(
    x=stage_x, y=[1]*len(stage_x),
    marker_color=stage_colors_mapped,
    text=stage_names,
    textposition="inside",
    textfont=dict(size=9, color="white"),
    hovertemplate="Window %{x}: %{text}<extra></extra>",
))
fig_ribbon.update_layout(
    height=80, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=50, r=20, t=5, b=5),
    yaxis=dict(visible=False),
    xaxis=dict(visible=False),
    showlegend=False,
    bargap=0.05,
)
st.plotly_chart(fig_ribbon, use_container_width=True)

# Legend
legend_html = " ".join(
    f'<span style="display:inline-block; margin:0 8px; font-size:0.85rem;">'
    f'<span style="display:inline-block; width:12px; height:12px; '
    f'background:{color}; border-radius:3px; margin-right:4px; vertical-align:middle;"></span>'
    f'{stage}</span>'
    for stage, color in STAGE_COLORS.items()
)
st.markdown(f'<div style="text-align:center;">{legend_html}</div>', unsafe_allow_html=True)

# ---- 3. Window selector + detail panels ------------------------------------
st.markdown('<div class="section-header">🔍 Window Detail</div>',
            unsafe_allow_html=True)

selected_wid = st.slider(
    "Select window", min_value=0, max_value=max(len(windows)-1, 0), value=0,
    help="Slide to inspect individual windows"
)

col_graph, col_explain = st.columns([1, 1])

# ---- 4. Network graph view -------------------------------------------------
with col_graph:
    st.markdown("##### 🌐 Network Graph (Attention Weights)")
    attn = attention_by_window.get(selected_wid, {})
    nodes_attn = attn.get("nodes", [])
    edges_attn = attn.get("edges", [])

    if nodes_attn and edges_attn:
        # Simple spring layout approximation
        n = len(nodes_attn)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {node["ip"]: (float(np.cos(a)), float(np.sin(a))) for node, a in zip(nodes_attn, angles)}

        fig_graph = go.Figure()
        # Draw edges
        for e in edges_attn[:50]:  # limit for performance
            x0, y0 = pos.get(e["src_ip"], (0, 0))
            x1, y1 = pos.get(e["dst_ip"], (0, 0))
            w = e["weight"]
            fig_graph.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(0.5, w * 5), color=f"rgba(88,166,255,{min(0.9, w+0.1)})"),
                hoverinfo="skip", showlegend=False,
            ))
        # Draw nodes
        suspicious = {h["ip"] for h in attn.get("suspicious_hosts", [])[:3]}
        nx, ny, ntext, ncolor = [], [], [], []
        for node in nodes_attn:
            x, y = pos.get(node["ip"], (0, 0))
            nx.append(x); ny.append(y); ntext.append(node["ip"])
            ncolor.append("#f85149" if node["ip"] in suspicious else "#58a6ff")
        fig_graph.add_trace(go.Scatter(
            x=nx, y=ny, mode="markers+text",
            text=ntext, textposition="top center",
            textfont=dict(size=9, color="#c9d1d9"),
            marker=dict(size=12, color=ncolor, line=dict(width=1, color="#30363d")),
            showlegend=False,
        ))
        fig_graph.update_layout(
            height=350, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.5)",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        )
        st.plotly_chart(fig_graph, use_container_width=True)

        # Suspicious hosts
        sus_hosts = attn.get("suspicious_hosts", [])
        if sus_hosts:
            st.markdown("**Top suspicious hosts:**")
            for h in sus_hosts[:5]:
                st.markdown(f"- `{h['ip']}` — attention weight: {h['total_attention']:.3f}")
    else:
        st.info("No attention data for this window (empty graph or model not loaded).")

# ---- 5. Explainability panel -----------------------------------------------
with col_explain:
    st.markdown("##### 🧠 Explainability")
    top_feats = result.get("top_features_by_window", {}).get(selected_wid, {})
    feats_list = top_feats.get("top_features", []) if isinstance(top_feats, dict) else []

    if feats_list:
        # SHAP bar chart
        feat_names = [f["feature"] for f in feats_list]
        feat_values = [f["mean_abs_shap"] for f in feats_list]
        feat_colors = ["#f85149" if f.get("direction") == "positive" else "#58a6ff" for f in feats_list]

        fig_shap = go.Figure(go.Bar(
            x=feat_values, y=feat_names, orientation="h",
            marker_color=feat_colors,
            text=[f"{v:.3f}" for v in feat_values],
            textposition="outside",
        ))
        fig_shap.update_layout(
            height=250, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.5)",
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Mean |SHAP|",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Natural-language explanation
        try:
            from src.explainability.shap_surrogate import generate_explanation_sentence
            if stage_timeline and selected_wid < len(stage_timeline):
                pred_stage = stage_timeline[selected_wid]["predicted_stage"]
                winfo = windows[selected_wid] if selected_wid < len(windows) else {}
                top_ip = winfo.get("node_ips", [None])[0] if winfo.get("node_ips") else None
                explanation = generate_explanation_sentence(feats_list, pred_stage, top_ip)
                st.markdown(f"> 💡 {explanation}")
        except Exception:
            pass
    else:
        st.info("No SHAP explanation available for this window. "
                "Train the XGBoost surrogate to enable per-window explanations.")

    # Window detail metadata
    if selected_wid < len(windows):
        w = windows[selected_wid]
        st.markdown("**Window Info:**")
        st.markdown(f"- Nodes: **{w['n_nodes']}** | Edges: **{w['n_edges']}**")
        if selected_wid < len(infil_timeline):
            prob = infil_timeline[selected_wid]["prob"]
            st.markdown(f"- Infiltration prob: **{prob:.4f}**")
        if selected_wid < len(stage_timeline):
            stage = stage_timeline[selected_wid]["predicted_stage"]
            color = STAGE_COLORS.get(stage, "#8b949e")
            st.markdown(f"- Predicted stage: "
                        f'<span style="color:{color};font-weight:bold">{stage}</span>',
                        unsafe_allow_html=True)
        if selected_wid < len(kl_surprise):
            kl_val = kl_surprise[selected_wid]["kl"]
            st.markdown(f"- KL surprise: **{kl_val:.4f}**")

# ---- 6. Export button -------------------------------------------------------
st.markdown("---")
col_export, _ = st.columns([1, 3])
with col_export:
    if st.button("📄 Export Summary Report"):
        lines = [
            "# GT-RSSM Analysis Report",
            f"**File:** {uploaded.name}",
            f"**Windows analyzed:** {n_windows}",
            f"**Peak infiltration probability:** {max_prob:.4f}",
            f"**Attack windows (>{alert_thresh}):** {attack_windows}",
            f"**Mean KL surprise:** {mean_kl:.4f}",
            "",
            "## Infiltration Timeline",
        ]
        for w in infil_timeline:
            lines.append(f"- Window {w['window_id']}: P={w['prob']:.4f}")
        lines.append("")
        lines.append("## Stage Predictions")
        for s in stage_timeline:
            lines.append(f"- Window {s['window_id']}: {s['predicted_stage']}")
        report = "\n".join(lines)
        st.download_button("💾 Download Report", report, file_name="gt_rssm_report.md",
                           mime="text/markdown")
