# CyberPulse: AI-Based Network Attack Forecasting

CyberPulse uses a Dual-Stream Temporal GRU World Model to learn network state evolution and forecast attacks at multiple future horizons (+5s, +30s, +60s) using both flow and packet telemetry.

## Architecture

1. **Preprocessing**: 
    - Flow records are aggregated into 5-second non-overlapping windows.
    - Streaming PCAP data is aggregated into matching windowed packet features.
2. **Features**: 28 flow features and 12 packet features define the network state.
3. **Temporal Dynamics**: A recurrent GRU Cell tracks the network state evolution.
4. **World Model**: 
    - Forecasts future state `S_(t+1)`.
    - Outputs future probabilities for +5s, +30s, +60s.
    - Predicts future MITRE ATT&CK stage.
5. **Explainability**: Integrated Gradients identifies top contributing features.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

See `scripts/run_full_pipeline.ps1` for end-to-end execution, or run individually:

1. **Data Prep**
```bash
python data_prep/build_temporal_dataset.py --clean_flow data/processed/cic2017/cic2017_clean.parquet --mapping_file mitre_mapping/attack_stage_mapping.json --output_dir data/processed/cic2017
```

2. **Train Baselines**
```bash
python src/training/train_baselines.py
```

3. **Train Temporal GRU**
```bash
python src/training/train_world_model.py
```

4. **Evaluate**
```bash
python src/evaluation/run_eval.py
```

5. **Dashboard**
```bash
streamlit run dashboard/streamlit_app.py
```

## Migration Summary

The old GT-RSSM architecture (including RSSM, GAT, focal loss, dense graph matrices) has been completely removed in favor of this lightweight temporal GRU world model.
