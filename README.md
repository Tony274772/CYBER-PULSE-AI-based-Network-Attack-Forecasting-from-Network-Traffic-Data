# GT-RSSM: Network Attack Forecasting from Network Traffic Data

> **SIH26153 — NTRO: AI-based Network Attack Forecasting**
>
> A world-model-based system that forecasts network attacks *before* a
> compromise completes, using graph attention + recurrent state-space dynamics
> to predict future infiltration probability and MITRE ATT&CK stage.

---

## Quick Start

### 1. Install Dependencies
```bash
# Python 3.11+ required
pip install -r requirements.txt
python -c "import torch, xgboost, shap, captum, scapy, streamlit"
```

### 2. Data Setup
Place datasets as described in `GT-RSSM_Build_Instructions.md` Section 2:
```
Data/raw/cic2017/csv/*.csv              # CIC-IDS2017 flow CSVs
Data/raw/cic2017/pcap/Wednesday-workingHours.pcap   # (optional) Wednesday PCAP
Data/raw/ctu13/**                        # CTU-13 extracted tarball
```

### 3. Run the Pipeline
```bash
# Step 3: Clean & ingest data
python -m data_prep.clean_cic2017
python -m data_prep.clean_ctu13

# Step 4: Parse Wednesday PCAP (optional)
python -m src.ingestion.pcap_parser

# Step 5: Build windowed feature stores
python -m src.features.build_dataset --dataset cic2017
python -m src.features.build_dataset --dataset ctu13
python -m src.features.build_dataset --dataset ctu13 --background --skip-splits

# Step 6: Train & evaluate the baseline
python -m src.baseline.logreg_baseline

# Step 7: Stage-1 unsupervised pretraining
python -m src.training.pretrain_stage1

# Step 8: Stage-2 supervised fine-tuning
python -m src.training.finetune_stage2

# Step 11: Full evaluation
python -m src.evaluation.run_eval
```

### 4. Launch the Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```
Upload a CIC-IDS2017 CSV or PCAP file to see real-time forecasting.

### 5. (Optional) FastAPI
```bash
uvicorn api.main:app --port 8000
# Then POST /predict with a file upload
```

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full 2-page
architecture document. In brief:

```
Traffic → 5s Windows → Dense Graph Attention → RSSM World Model → 4 Heads
                                                                    ↓
                                    Reconstruction | Infiltration | MITRE Stage | KL Surprise
                                                                    ↓
                                    K-step imagination → forecast probability curve
```

**Key design decisions** (deliberate engineering choices for reliability):

1. **Dense masked attention instead of PyG GATConv** — same math, zero fragile
   dependencies, trivial ONNX export path if ever needed.
2. **Diagonal Gaussian latents (Dreamer-v1/PlaNet) instead of categorical
   (Dreamer-v3)** — standard VAE reparameterization, simpler implementation,
   same "world model" lineage.
3. **In-process serving instead of ONNX runtime** — the model runs sub-second
   CPU inference per window with plain PyTorch `eval()` mode.

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── configs/default.yaml               # All hyperparameters (Section 9)
├── docs/architecture.md               # 2-page architecture document
├── mitre_mapping/
│   └── attack_stage_mapping.json      # MITRE ATT&CK stage mapping (7 stages)
├── src/
│   ├── config.py                      # Config system (Section 5)
│   ├── ingestion/                     # Data loaders (Section 7.1-7.2)
│   ├── features/                      # Feature engineering & windowing (Section 7.3-7.4)
│   ├── model/                         # GT-RSSM model (Section 7.5)
│   ├── training/                      # Stage-1 & Stage-2 training (Section 7.6)
│   ├── baseline/                      # Logistic Regression baseline (Section 7.7)
│   ├── explainability/                # Attention, IG, SHAP (Section 7.8)
│   ├── evaluation/                    # Metrics & full eval (Section 7.9)
│   └── inference/                     # Unified predict() entry point (Section 7.10)
├── dashboard/streamlit_app.py         # Streamlit UI (Section 7.11)
├── api/main.py                        # FastAPI wrapper (Section 7.12)
├── checkpoints/                       # Trained model weights
└── scripts/run_full_pipeline.ps1      # Full pipeline script
```

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| Infiltration F1 / ROC-AUC | Binary attack detection accuracy |
| MITRE Stage Macro F1 | Multi-class attack stage classification |
| **Lead Time** | How many seconds *before* compromise the model alerts |
| **K-step Brier Score** | Calibration of the imagination forecast |
| **Cross-dataset F1** | Zero-shot generalization (train CIC-IDS2017, test CTU-13) |
| **Surprise AUC** | Can KL alone detect unseen attack categories? |

---

## Datasets

| Dataset | Role | Size |
|---|---|---|
| CIC-IDS2017 | Primary (14 attack types) | ~225 MB CSVs |
| CTU-13 | Secondary (Botnet/C2) | ~1.9 GB |
| NF-UNSW-NB15-v2 | Optional (cross-dataset test) | ~200 MB CSV |

---

## Future Extensions
- ONNX export for browser/edge deployment
- Federated / privacy-preserving training
- Active-learning loop on high-KL windows
- Live Zeek/Suricata streaming ingestion
- CSE-CIC-IDS2018, CICIoT2023, LANL datasets

---

## License
This project was developed for Smart India Hackathon 2026 (SIH26153).
