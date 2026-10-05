"""Streamlit Dashboard for CyberPulse."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
import json

from src.inference.predict import predict

st.set_page_config(page_title="CyberPulse Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ CyberPulse: AI-Based Network Attack Forecasting")
st.markdown("**Learn the evolution of network state, not just the current attack label.**")

st.sidebar.header("Upload Traffic Data")
uploaded_file = st.sidebar.file_uploader("Upload CIC-IDS2017 Flow CSV or PCAP", type=["csv", "pcap"])

if uploaded_file is not None:
    with st.spinner("Processing network traffic..."):
        # Save temp
        suffix = ".csv" if uploaded_file.name.endswith(".csv") else ".pcap"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        try:
            results = predict(tmp_path, config_path="configs/random_split.yaml")
            
            st.success("Analysis Complete!")
            
            # Layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Packet Telemetry", "AVAILABLE" if results["packet_features_available"] else "FLOW-ONLY")
            with col2:
                risk = results["forecast"]["5s"]
                st.metric("+5s Risk", f"{risk*100:.1f}%")
            with col3:
                risk_30 = results["forecast"]["30s"]
                st.metric("+30s Risk", f"{risk_30*100:.1f}%", delta="High Risk" if risk_30 >= results["alert_threshold"] else "Normal")
            with col4:
                risk_60 = results["forecast"]["60s"]
                st.metric("+60s Risk", f"{risk_60*100:.1f}%")
                
            st.markdown("---")
            
            col_chart, col_explain = st.columns([2, 1])
            
            with col_chart:
                st.subheader("Forecast Probability Timeline")
                timeline = results["forecast_timeline"]
                x_vals = [f"+{i*5}s" for i in range(1, 13)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=timeline, mode='lines+markers', name='Risk Prob', line=dict(color='red', width=3)))
                fig.add_hline(y=results["alert_threshold"], line_dash="dash", line_color="orange", annotation_text="Alert Threshold")
                fig.update_layout(yaxis_range=[0, 1], title="60-Second Recursive Rollout")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Predicted Future Stage")
                st.info(f"**Stage:** {results['stage_forecast']['stage']} (Confidence: {results['stage_forecast']['probability']*100:.1f}%)")
                
            with col_explain:
                st.subheader("Explainability Drivers")
                st.write("**Top Flow Features**")
                for f in results["top_flow_features"]:
                    st.write(f"- `{f}`")
                    
                if results["packet_features_available"]:
                    st.write("**Top Packet Features**")
                    for p in results["top_packet_features"]:
                        st.write(f"- `{p}`")
                        
                st.markdown("---")
                st.subheader("Network State Summary")
                curr = results["current_state"]
                st.write(f"Total Packets: {curr.get('total_packets', 0)}")
                st.write(f"Total Flows: {curr.get('flow_count', 0)}")
                st.write(f"Unique IPs: {curr.get('unique_source_ips', 0)}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
else:
    st.info("Please upload a network traffic file from the sidebar to begin.")
