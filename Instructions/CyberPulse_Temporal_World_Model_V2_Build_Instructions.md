# CyberPulse v2 — Lightweight Temporal World Model Build Instructions

## For an AI coding agent modifying the existing CyberPulse / SIH26153 repository

> **Mission:** Replace the existing GT-RSSM implementation with a lightweight, CPU-friendly temporal world model that uses **both flow-level and packet-level network telemetry**, learns temporal state transitions, forecasts attacks at multiple future horizons, predicts a future MITRE ATT&CK stage, and provides feature-level explanations.
>
> **Important:** This document supersedes `GT-RSSM_Build_Instructions.md` for the implementation. Do not preserve the GT-RSSM model as an active code path. Remove the old graph/RSSM training stack and build the new architecture described below.

---

# 0. Problem-Statement Alignment

The SIH26153 / NTRO problem asks for a system that goes beyond static malicious/benign classification. The required system should learn evolving network state, model temporal/state-transition behavior, forecast future malicious activity, map predicted activity to recognized attack stages such as MITRE ATT&CK, and provide interpretable decision support.

The new implementation must therefore preserve the following ideas:

```text
Observed traffic history
        ↓
Compact network state S_t
        ↓
Learn temporal dynamics
        ↓
Predict future network state
        ↓
Forecast attack probability
        ↓
Forecast attack stage
        ↓
Explain the forecast
```

The implementation does **not** need a graph neural network or an RSSM. A feature-vector network state plus a recurrent transition model is valid for this problem statement.

The original project specification explicitly allowed a structured feature vector or graph and allowed LSTM/Transformer/GNN/other sequence models. Preserve that compliance while dramatically reducing computation.

---

# 1. Why the Existing GT-RSSM Must Be Replaced

The old repository uses a heavy pipeline based on:

```text
5-second windows
50% overlap
        ↓
host-interaction graphs
        ↓
128-node padding
        ↓
dense graph attention
        ↓
RSSM with Gaussian latent state
        ↓
reconstruction + KL + focal + stage loss
        ↓
K-step RSSM imagination
        ↓
attention + integrated gradients + SHAP surrogate
```

The existing build specification defines `N_max=128`, `W=20`, dense edge tensors, graph attention, a 128-dimensional RSSM hidden state and 32-dimensional stochastic latent, two-stage training, and several explainability paths. These are unnecessary for a CPU-constrained hackathon implementation.

The repository's existing Stage-2 metrics also show that the old architecture was not yet producing a reliable result: the recorded best epoch is 0 and the best validation F1 is approximately 0.233. The old Logistic Regression baseline on the current per-edge CIC-IDS2017 formulation also has weak test F1, so this rebuild must fix the **data formulation and forecast target**, not simply increase the old epoch count.

The new implementation must be much simpler and must train directly on a **future-forecasting target**.

---

# 2. Non-Negotiable Design Decisions

The agent must follow these decisions unless a runtime incompatibility forces a small engineering adjustment.

## 2.1 Replace the GT-RSSM model

Do not keep these models in the active implementation:

- Dense Graph Attention
- RSSM
- Gaussian latent posterior/prior
- KL anomaly signal
- Reconstruction ELBO training
- GT-RSSM two-stage pretraining/fine-tuning
- graph padding to 128 nodes
- dense `N x N x edge_dim` tensors
- graph-attention explanation export
- XGBoost surrogate for explaining the neural model

## 2.2 New model

Use a **Dual-Stream Temporal GRU World Model**:

```text
FLOW FEATURES                    PACKET FEATURES
     │                                 │
     ↓                                 ↓
Flow MLP 27 → 32                Packet MLP 12 → 32
     │                                 │
     └──────────────┬──────────────────┘
                    ↓
             fused state x_t
                 ~64 dims
                    ↓
             GRUCell hidden=64
                    ↓
               world state h_t
                    │
       ┌────────────┼───────────────┐
       ↓            ↓               ↓
 next-state      attack         stage head
  prediction      heads          6 classes
       │            │               │
       ↓            ↓               ↓
    S(t+1)      +5/+30/+60s     future stage
                    │
                    ↓
              recursive rollout
                    ↓
           forecast probability
                timeline
```

The model is intentionally small enough to run on CPU.

## 2.3 Both flow and packet data are mandatory

The final architecture must consume both:

1. **Flow-level data** from CIC-IDS2017/CTU-13.
2. **Packet-level data** derived from the available Wednesday CIC-IDS2017 PCAP.

Do not remove packet-level processing just because it costs some preprocessing time. Reduce the packet feature set instead.

## 2.4 The primary forecast is future attack risk

Do not train the main model only on:

```text
"Is this current window malicious?"
```

Train it on:

```text
"Will an attack occur in the next H seconds?"
```

Required horizons:

- +5 seconds
- +30 seconds
- +60 seconds

The +30-second horizon is the primary headline forecast.

## 2.5 CPU-first implementation

No CUDA is required.

The model must train and infer on CPU.

Target model characteristics:

- approximately 40-dimensional raw network state
- 12 historical windows
- 64-dimensional fused state
- 1 GRUCell, hidden size 64
- small MLP heads
- batch training with early stopping

---

# 3. Repository Cleanup — Remove Old GT-RSSM Code

Before implementing v2, inspect the repository and create a clean working branch or backup.

Never delete raw datasets.

## 3.1 Delete or retire these old model files

Remove these from the active source tree:

```text
src/model/dense_graph_attention.py
src/model/rssm.py
src/model/gt_rssm.py
src/model/heads.py
```

Replace them with:

```text
src/model/temporal_world_model.py
src/model/heads.py                 # only if rewritten for the new model
```

Prefer a single main model file if that keeps the implementation clearer.

## 3.2 Delete old training stages

Remove the old training architecture:

```text
src/training/pretrain_stage1.py
src/training/finetune_stage2.py
src/training/losses.py
```

Replace with:

```text
src/training/train_world_model.py
src/training/dataset.py
src/training/losses.py             # optional; rewrite completely if retained
```

There must be **one supervised training pipeline**, not Stage 1 + Stage 2.

## 3.3 Delete old graph-specific preprocessing

The old graph/window implementation should not remain active.

Replace:

```text
src/features/windowing.py
src/features/build_dataset.py
src/features/schema.py
```

with a simple window-state pipeline.

Recommended files:

```text
src/features/flow_features.py
src/features/packet_features.py
src/features/window_state.py
src/features/forecast_targets.py
src/features/build_temporal_dataset.py
```

## 3.4 Delete/replace explainability code

Remove:

```text
src/explainability/attention_export.py
src/explainability/shap_surrogate.py
```

Rewrite:

```text
src/explainability/integrated_gradients.py
```

For the neural model, use Integrated Gradients directly on the future +30-second attack score.

For the XGBoost baseline, SHAP is allowed directly because tree-based SHAP is appropriate there.

## 3.5 Rewrite inference

Replace the old RSSM inference path:

```text
src/inference/predict.py
```

It must load the lightweight model, construct the same flow+packet state pipeline used in training, encode the most recent 12 windows, and perform recursive world-model rollout.

## 3.6 Rewrite dashboard

The old dashboard contains GT-RSSM-specific text, graph visualizations, KL-surprise displays, attention graphs, and GT-RSSM terminology.

Rewrite:

```text
dashboard/streamlit_app.py
```

It must present:

- current network state
- attack probability at +5/+30/+60 seconds
- forecast probability curve
- predicted future attack stage
- top contributing flow-level features
- top contributing packet-level features
- packet-data availability
- selected suspicious source/destination/port details from the underlying traffic
- model confidence/calibration information where applicable

## 3.7 Rewrite documentation/configuration

Rewrite:

```text
README.md
docs/architecture.md
configs/default.yaml
scripts/run_full_pipeline.ps1
```

The old `GT-RSSM_Build_Instructions.md` should no longer be treated as build instructions. It may be moved to an `archive/old_gt_rssm/` folder for historical reference or deleted if the repository should be clean.

## 3.8 Old generated artifacts

Move old generated artifacts into an archive or delete them if they are not needed:

```text
metrics/stage1_curves.png
metrics/stage1_log.json
metrics/stage2_best_metrics.json
metrics/stage2_curves.png
metrics/stage2_history.json
metrics/stage2_metrics.json
checkpoints/gt_rssm_v1.pt
checkpoints/stage1.pt
```

Do not let the new code read any old GT-RSSM checkpoint.

---

# 4. What Should Be Reused From the Existing Repository

Do not rewrite useful working ingestion code unnecessarily.

Reuse and modify:

```text
data_prep/clean_cic2017.py
data_prep/clean_ctu13.py
data_prep/extract_packet_features.py
data_prep/inspect_pcap.py
src/ingestion/flow_csv_loader.py
src/ingestion/ctu13_loader.py
mitre_mapping/attack_stage_mapping.json
```

The current repository already contains important CIC-IDS2017 cleaning logic, including:

- stripping column-name whitespace
- dropping embedded duplicate header rows
- handling `inf` / `NaN`
- parsing timestamps with the appropriate date convention
- checking the July 2017 capture range

Keep these protections.

The current packet extractor already uses `PcapReader` and streams the PCAP instead of loading the entire capture into memory. Keep this approach.

Do not use `rdpcap` for a multi-GB capture.

---

# 5. Dataset Strategy Using the Data Already in the Project

## 5.1 Primary training dataset

Use:

```text
CIC-IDS2017 flow CSVs
+
Wednesday-workingHours.pcap packet data
```

This is the primary dataset for the first complete model.

## 5.2 Secondary evaluation dataset

Use:

```text
CTU-13
```

primarily for zero-shot cross-dataset evaluation after the CIC model works.

Do not block the first successful training run on CTU-13.

## 5.3 Optional dataset

NF-UNSW-NB15-v2 remains optional.

Do not include it in the first model training pipeline unless the core model is already working.

## 5.4 No new large datasets

Do not download or introduce CIC-IDS2018, CICIoT2023, LANL, DARPA, or other large sources during the first implementation.

The goal is to build a strong, reproducible model using the data already associated with this project.

---

# 6. Raw Data Layout

Support the existing paths:

```text
data/raw/cic2017/csv/*.csv
data/raw/cic2017/pcap/Wednesday-workingHours.pcap
data/raw/ctu13/**
data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv        # optional
```

Do not require the user to reorganize raw data unless absolutely necessary.

---

# 7. Preprocessing Pipeline — Required Order

The new pipeline must be:

```text
RAW FLOW CSV
      ↓
clean + normalize schema
      ↓
raw flow table
      ↓
time-aware split boundary definition
      ↓
5-second non-overlapping windows
      ↓
flow window aggregation
      ↓
packet PCAP streaming
      ↓
packet window aggregation
      ↓
join flow + packet states by window_id
      ↓
feature cleaning / log transform
      ↓
future target creation
      ↓
6-window / 12-window historical sequences
      ↓
train-only scaler
      ↓
PyTorch dataset
```

The order is important because future information must never leak into the model features or scaling statistics.

---

# 8. Windowing Policy

Use:

```yaml
delta_t_seconds: 5
window_overlap: 0.0
history_windows: 12
```

This gives:

```text
12 × 5 seconds = 60 seconds of observed history
```

Do **not** use the old 50% overlap.

Use:

```text
00–05
05–10
10–15
...
```

Each window gets one compact network state vector.

This also keeps the existing packet-extraction implementation naturally aligned with the data.

If a previous packet parquet was created with a different window size, do not silently reuse it as if the window IDs match. Prefer re-running the packet extractor with the configured 5-second window size.

---

# 9. Flow-Level Preprocessing

## 9.1 Load and standardize

For every CIC-IDS2017 CSV:

```python
df.columns = df.columns.str.strip()
```

Drop duplicate header rows such as rows where:

```text
Label == "Label"
```

Convert:

```text
Timestamp
```

to pandas datetime using the same day-first convention used by the existing cleaner.

Validate the date range against the known CIC-IDS2017 capture period.

## 9.2 Numeric cleanup

Replace:

```text
+inf
-inf
```

with `NaN`.

Do not allow infinities into the model.

For the raw flow table:

- drop rows with invalid timestamps
- drop exact duplicate rows
- drop impossible negative numerical values where the field definition makes negativity impossible
- fill benign missing numeric values only where the feature semantics permit it
- log every major cleaning operation

Never silently discard a large percentage of the data.

## 9.3 IP and identifier handling

Do not use raw source/destination IP addresses as learned categorical features.

Do not one-hot encode millions of IP addresses.

Use IP addresses only for:

- aggregation keys
- dashboard display
- suspicious-host reporting

## 9.4 Port handling

Do not feed raw port numbers as the main numeric signal.

Derive behavior-oriented statistics such as:

- number of unique destination ports
- number of unique source ports
- destination-port entropy
- sequential-port score
- fraction of connections to common/service ports if available

---

# 10. Exact Flow-Level Window Features

Build a global network state for each time window.

The recommended flow feature vector is:

```text
flow_count
total_packets
total_fwd_packets
total_bwd_packets
total_fwd_bytes
total_bwd_bytes
total_flow_duration
mean_flow_duration
mean_iat
std_iat
min_iat
max_iat
syn_count
ack_count
fin_count
rst_count
psh_count
urg_count
mean_flow_bytes_per_sec
mean_flow_packets_per_sec
fwd_bwd_byte_ratio
fwd_bwd_packet_ratio
unique_source_ips
unique_destination_ips
unique_source_ports
unique_destination_ports
unique_protocols
destination_port_entropy
sequential_port_score
```

Target dimensionality: about **28 flow features**.

Do not add dozens of weak/redundant columns simply because the original CICFlowMeter table contains them.

## 10.1 Aggregation rules

For additive quantities:

```text
sum
```

For rates and statistics:

```text
mean / weighted mean
```

For diversity:

```text
nunique
```

For ratios:

recompute them from the aggregated numerator/denominator whenever possible instead of taking an unweighted mean of row-level ratios.

For port entropy:

calculate from the destination-port distribution within the window.

For scan score:

calculate from the ordered/distinct destination-port behavior within the window.

---

# 11. Packet-Level Preprocessing

The packet path is mandatory for the final architecture.

Use the already available:

```text
data/raw/cic2017/pcap/Wednesday-workingHours.pcap
```

The existing project already has a packet feature extractor. Modify it rather than replacing it with a completely different packet parsing framework.

## 11.1 Memory requirement

Use streaming:

```python
from scapy.utils import PcapReader
```

Never use:

```python
rdpcap(...)
```

for the full multi-GB capture.

## 11.2 Packet-level state features

Use a compact packet vector:

```text
ttl_mean
ttl_variance
tcp_window_mean
tcp_window_std
ip_fragment_ratio
payload_size_mean
payload_size_std
retransmission_count
port_scan_sequential_score
port_scan_randomised_score
packet_count
```

Target: **11 packet features**.

Add:

```text
packet_features_available
```

as a separate binary indicator.

Therefore the packet branch input is:

```text
12 dimensions
```

## 11.3 Packet aggregation

The existing packet extractor computes packet statistics per:

```text
(window_id, src_ip, dst_ip)
```

Keep this intermediate form because it is useful for dashboard host-pair explanations.

For the neural model, aggregate those per-edge packet statistics into one global network-level packet state for each window.

Suggested aggregations:

- packet_count → sum
- retransmission_count → sum
- TTL mean → packet-count-weighted mean
- TTL variance → weighted aggregate or robust approximation
- TCP-window mean/std → weighted mean/std
- fragment ratio → mean weighted by packet count
- payload mean/std → weighted aggregate
- scan sequential score → max and mean, choose one compact representation
- scan randomised score → max and mean, choose one compact representation

Keep the resulting vector fixed-length and deterministic.

## 11.4 Packet availability

For windows with no packet-level coverage:

```text
all packet features = 0
packet_features_available = 0
```

For windows with real PCAP-derived packet data:

```text
packet_features_available = 1
```

Do not fabricate packet measurements.

The model must learn that missing packet telemetry is different from measured zeros.

---

# 12. Join Flow and Packet Features

Create one row per time window:

```text
window_id
timestamp
flow_1 ... flow_28
packet_1 ... packet_11
packet_features_available
```

Approximate combined dimensionality:

```text
28 flow + 11 packet + 1 availability = 40 features
```

The final feature dimension may be 39–42 depending on the exact engineered aggregation count, but it must be fixed and written to a metadata JSON file.

Do not keep variable-width feature vectors.

Create:

```text
data/processed/cic2017/window_states.parquet
```

with one row per window.

---

# 13. Future Forecast Target Construction

This is the most important change from the old project.

## 13.1 Current state

Let:

```text
S_t = network state at time t
```

The model observes:

```text
S_(t-11), S_(t-10), ..., S_t
```

Then forecasts the future.

## 13.2 Attack target at +5 seconds

```text
y_5(t) = 1
```

if the next 5-second window contains any non-Benign attack label.

## 13.3 Attack target at +30 seconds

```text
y_30(t) = 1
```

if any attack occurs in the next six 5-second windows.

Formally:

```text
[t+1, t+6]
```

## 13.4 Attack target at +60 seconds

```text
y_60(t) = 1
```

if any attack occurs in the next twelve 5-second windows.

## 13.5 Future stage target

For the +30-second horizon, assign:

```text
future_stage_30
```

to the most severe attack stage present in the next six windows.

If there is no attack:

```text
Benign
```

Stage mapping should reuse:

```text
mitre_mapping/attack_stage_mapping.json
```

The expected stage set is:

```text
Benign
Reconnaissance
Initial_Access
Lateral_Movement
Command_And_Control
Exfiltration
Impact_DoS
```

The existing project correctly includes `Impact_DoS` because DoS/DDoS traffic is prominent in CIC-IDS2017 and should not be incorrectly forced into a different MITRE tactic.

## 13.6 Next-state target

For each training example also retain:

```text
S_(t+1)
```

as the supervised transition target for the world-model component.

This lets the model learn:

```text
S_t → S_(t+1)
```

rather than being only an attack classifier.

---

# 14. Data Leakage Prevention

This section is mandatory.

## 14.1 Never use future windows as input features

For a sample at time `t`:

```text
INPUT:
S_(t-11) ... S_t

TARGET:
S_(t+1)
y_5
y_30
y_60
future_stage_30
```

The input must never include any feature derived from windows after `t`.

## 14.2 Time-aware split

Do not use random `train_test_split` for the final temporal dataset.

Preferred development split:

```text
oldest 70% → train
next 15% → validation
latest 15% → test
```

The split should be done separately within each CIC capture segment/source file where necessary to preserve enough labeled coverage, following the project's existing approach, but with a purge gap.

## 14.3 Purge gap

Because each sample looks back 60 seconds and labels look forward 60 seconds, place at least a:

```text
60-second purge gap
```

around train/validation/test boundaries.

No sample whose history or future target overlaps another split should survive the split boundary.

## 14.4 Fit preprocessing only on training data

Anything learned from data must be fitted on training only:

- scaler mean/std
- clipping thresholds if data-dependent
- class weights
- threshold optimization
- calibration parameters

Save them to:

```text
checkpoints/preprocessing_stats.json
```

or an equivalent versioned artifact.

---

# 15. Heavy-Tailed Feature Processing

Network quantities are highly skewed.

Apply `log1p` to appropriate nonnegative magnitude/count features such as:

```text
bytes
packets
flow_count
duration sums
retransmissions
payload sizes
```

Do not log-transform bounded ratios or binary flags.

Then apply:

```text
StandardScaler
```

fit on the training set only.

Optional robust clipping:

```text
clip to training 0.5th–99.5th percentile
```

if extreme values are observed.

Write a data report showing:

- raw row count
- cleaned row count
- window count
- positive rate for +5s/+30s/+60s
- packet coverage percentage
- missing-value counts
- final feature count

---

# 16. Sequence Construction

After window-level feature creation:

```text
history = 12 windows
```

Construct samples:

```text
X_i = [S_(i-11), ..., S_i]
```

Shape:

```text
[batch, 12, feature_dim]
```

Targets:

```text
next_state:      [batch, feature_dim]
y_5:             [batch]
y_30:            [batch]
y_60:            [batch]
stage_30:         [batch]
```

No graph tensors.

No host-node padding.

No adjacency matrices.

---

# 17. Final Model Architecture

## 17.1 Flow branch

```text
28 flow features
       ↓
Linear(28, 32)
ReLU
LayerNorm
Dropout(0.10)
       ↓
32-dim flow embedding
```

## 17.2 Packet branch

```text
12 packet features
       ↓
Linear(12, 32)
ReLU
LayerNorm
Dropout(0.10)
       ↓
32-dim packet embedding
```

## 17.3 Fusion

```text
flow_embedding (32)
        ‖
packet_embedding (32)
        ↓
64-dim fused network state x_t
```

## 17.4 Temporal dynamics

Use a single:

```text
GRUCell(input_size=64, hidden_size=64)
```

Process the 12 observed windows sequentially:

```python
h = zeros(64)
for t in history:
    h = gru_cell(x_t, h)
```

The final:

```text
h_t
```

is the learned latent/deterministic network state.

This is intentionally a very small temporal world model.

---

# 18. Prediction Heads

## 18.1 Next-state transition head

```text
h_t
 ↓
Linear(64, 64)
ReLU
Linear(64, feature_dim)
 ↓
predicted S_(t+1)
```

Loss:

```text
SmoothL1 / Huber loss
```

This head provides explicit learned state-transition dynamics.

## 18.2 Multi-horizon attack head

Use one shared head producing three logits:

```text
h_t
 ↓
Linear(64, 32)
ReLU
Linear(32, 3)
 ↓
logits for +5s, +30s, +60s
```

Apply sigmoid independently.

Output:

```text
p_attack_5
p_attack_30
p_attack_60
```

The +30s score is the primary KPI.

## 18.3 Future-stage head

```text
h_t
 ↓
Linear(64, 32)
ReLU
Linear(32, 7)
 ↓
Benign / Recon / Initial_Access /
Lateral_Movement / C2 / Exfiltration / Impact_DoS
```

During training, apply the stage loss with a mask:

```text
mask = 1 when future_attack_30 == 1
mask = 0 when future_attack_30 == 0
```

Do not force the stage classifier to learn attack subtypes from benign examples.

---

# 19. World-Model Rollout

The model must support future simulation from the last observed state.

At inference:

```text
observed 12 windows
       ↓
h_t
       ↓
next-state head → predicted state S_(t+1)
       ↓
encode predicted state
       ↓
GRUCell
       ↓
h_(t+1)
       ↓
predict next state again
       ↓
repeat
```

Perform at least:

```text
12 rollout steps
```

corresponding to:

```text
60 seconds
```

At each step read the attack head and stage head.

The dashboard should show:

```text
+5s
+10s
+15s
...
+60s
```

as a forecast curve.

## 19.1 Avoid stochastic RSSM behavior

Do not introduce Gaussian latent sampling, KL divergence, VAE machinery, or stochastic rollouts.

The deterministic GRU state-transition model is sufficient for this implementation.

---

# 20. Training Loss

Use a simple weighted objective:

```text
L_total =
    0.3 * L_state
  + 1.0 * L_attack
  + 0.5 * L_stage
```

where:

```text
L_state  = SmoothL1(predicted_next_state, actual_next_state)
```

```text
L_attack = weighted BCE across +5s, +30s, +60s
```

```text
L_stage  = masked weighted CrossEntropy
```

The exact loss weights may be tuned on validation data, but do not create a complicated multi-stage training recipe.

---

# 21. Class Imbalance Strategy

Do not reproduce the old extreme inverse-frequency weights.

The previous GT-RSSM run produced very large stage weights, including an extremely large weight for the rare Exfiltration bucket. Avoid this because it can destabilize learning.

## 21.1 Attack forecast

For each forecast horizon calculate:

```text
negative / positive
```

on the training set.

Use moderate `pos_weight` for BCE.

Cap an obviously extreme weight if necessary and document the cap.

## 21.2 Stage prediction

Use:

```text
class-weighted CrossEntropy
```

with weights based on the attack-only training subset.

If a stage has too few examples to support meaningful training, do not invent synthetic examples. Report it as a low-support class.

Do not use SMOTE on temporal windows.

---

# 22. Training Configuration

Recommended starting values:

```yaml
training:
  epochs: 40
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 7
  gradient_clip_norm: 1.0
  num_workers: 0
  seed: 42
```

Use:

```text
AdamW
```

with a small model and CPU.

Use early stopping on:

```text
validation PR-AUC at the +30-second horizon
```

not accuracy.

Save:

```text
checkpoints/temporal_world_model.pt
```

The checkpoint must include:

- model state dict
- feature list
- feature dimension
- scaler statistics
- configuration
- class weights
- chosen alert threshold
- training seed
- dataset metadata

---

# 23. Required Baselines

Build the baselines before tuning the GRU.

## Baseline 1 — Logistic Regression

Use the same window-level features and the same future target.

Do not reuse the old per-edge baseline as the main comparison.

The baseline must predict:

```text
y_30
```

from the current window state.

Use:

```text
LogisticRegression(class_weight="balanced")
```

## Baseline 2 — XGBoost

Train a small XGBoost model on the same current-window + temporal summary features.

Recommended starting configuration:

```text
n_estimators = 300
max_depth = 5
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
tree_method = hist
```

The XGBoost model is not the main world model. It is a strong tabular benchmark.

## Final model

```text
Dual-Stream Temporal GRU World Model
```

The final model should beat or meaningfully complement the baselines on the forecasting metrics.

---

# 24. Feature Abulation Required

Run and report at least:

### Experiment A — Flow only

```text
flow features
→ GRU
```

### Experiment B — Packet + Flow

```text
flow features + packet features
→ GRU
```

This ablation is important because the problem statement explicitly requires both traffic levels.

The final presentation should be able to say whether packet-level information improves:

- +30s PR-AUC
- F1
- recall at fixed FPR
- lead time

Do not claim packet-level improvement unless the experiment actually demonstrates it.

---

# 25. Main Evaluation Metrics

Do not optimize for raw accuracy.

The required evaluation table should contain:

| Metric | Purpose |
|---|---|
| PR-AUC (+5s) | Short-horizon discrimination |
| **PR-AUC (+30s)** | **Primary forecasting KPI** |
| PR-AUC (+60s) | Longer-horizon discrimination |
| F1 (+30s) | Alert-quality summary |
| Precision (+30s) | Alert usefulness |
| Recall (+30s) | Attack coverage |
| False Positive Rate | Operational false alarms |
| ROC-AUC | Secondary discrimination metric |
| Brier Score (+30s) | Probability calibration |
| Median Lead Time | How early the model warns |
| Mean Lead Time | Overall anticipatory behavior |
| Stage Macro-F1 | Future MITRE-stage quality |
| Cross-dataset F1 | Generalization to CTU-13 |

---

# 26. Alert Threshold Selection

Do not hard-code `0.5` as the final alert threshold without validation.

Tune the +30-second threshold on validation data.

Preferred approach:

```text
choose threshold giving a target validation FPR around 5%
```

Then freeze that threshold.

Use that exact threshold once on the test set.

Do not tune on the test set.

Save the chosen threshold in the checkpoint metadata.

---

# 27. Lead-Time Evaluation

This metric is particularly important because the PS is about forecasting rather than only detecting an attack after it begins.

For each attack episode:

1. Find the first window where the ground-truth label becomes non-Benign.
2. Find the first forecast window before that point where:

```text
p_attack_30 >= alert_threshold
```

3. Compute:

```text
lead_time = compromise_time - alert_time
```

Positive values mean the system warned before the attack-labelled window.

Also report:

- median lead time
- mean lead time
- percentage of attack episodes with positive lead time
- percentage of attacks never anticipated before onset

Do not hide negative/zero lead times.

A real forecasting system should show both successful anticipation and late/no-warning cases.

---

# 28. Brier Score

For each future horizon:

```text
Brier = mean((probability - outcome)^2)
```

Report at minimum:

```text
Brier +30s
```

The model should output probabilities that are meaningful, not only rankings.

---

# 29. Cross-Dataset Evaluation

After the CIC-IDS2017 model is stable:

```text
Train: CIC-IDS2017
Test: CTU-13
```

without retraining.

Because CTU-13 does not have the same packet coverage, the packet branch must support:

```text
packet_features_available = 0
```

and zero-filled packet features.

Report:

```text
CIC test F1 / PR-AUC
CTU-13 F1 / PR-AUC
performance drop
```

Do not claim strong cross-dataset generalization unless the numbers justify it.

---

# 30. Explainability

## 30.1 Neural model

Use Integrated Gradients on the +30-second attack probability.

Input baseline:

```text
all-zero normalized state history
```

Compute attributions over:

```text
12 history windows × feature_dim
```

Aggregate absolute attribution across the 12 windows to obtain feature importance.

Report:

```text
top 5 flow features
top 5 packet features
top 3 historical windows
```

## 30.2 Natural-language explanation

Create a deterministic explanation template such as:

```text
High SYN activity, rising destination-port diversity, and increased
retransmission behavior are the strongest contributors to the +30s risk forecast.
```

Do not use an external LLM for this.

## 30.3 XGBoost baseline

SHAP TreeExplainer may be used for XGBoost.

Do not build an XGBoost surrogate to explain the neural GRU.

---

# 31. Dashboard Requirements

Rewrite the Streamlit dashboard around the new model.

## 31.1 Top-level cards

Show:

```text
Current Risk
+5s Risk
+30s Risk
+60s Risk
Predicted Future Stage
```

## 31.2 Forecast chart

Plot:

```text
Time / horizon
        vs
Attack probability
```

Mark the alert threshold.

Example x-axis:

```text
Now → +5s → +10s → ... → +60s
```

## 31.3 Stage forecast

Show the dominant predicted future stage and confidence.

Example:

```text
Predicted Stage: Command & Control
Confidence: 82%
```

Never invent a confidence number; derive it from the model probability.

## 31.4 Explainability panel

Show:

```text
Top flow drivers
Top packet drivers
Historical windows that mattered most
```

## 31.5 Packet coverage indicator

Show:

```text
Packet telemetry: AVAILABLE
```

or:

```text
Packet telemetry: FLOW-ONLY / UNAVAILABLE
```

## 31.6 Suspicious traffic details

Use the underlying flow records for the selected window to show:

- top source IPs by traffic volume
- top destination IPs
- top destination ports
- SYN-heavy sources
- high scan-score sources
- retransmission-heavy sources

These details are for defender visualization and are not required to be direct model inputs.

---

# 32. Inference API Contract

Implement:

```python
predict(input_path: str) -> dict
```

The returned dictionary should contain at least:

```python
{
    "windows": [...],
    "current_state": {...},
    "forecast": {
        "5s": float,
        "30s": float,
        "60s": float
    },
    "forecast_timeline": [...],
    "stage_forecast": {
        "stage": str,
        "probability": float
    },
    "top_flow_features": [...],
    "top_packet_features": [...],
    "packet_features_available": bool,
    "alert_threshold": float
}
```

Support both:

```text
CSV input
PCAP input
```

when the corresponding parser exists.

The same preprocessing functions used by training must be used for inference. Do not create a second incompatible preprocessing implementation.

---

# 33. Inference Behavior for CSV

If a user uploads a CIC-IDS2017-style flow CSV:

```text
CSV
 ↓
clean
 ↓
5-sec windows
 ↓
flow state
 ↓
packet features = zero unless a matching packet file is explicitly provided
 ↓
packet_features_available = 0
 ↓
12-window history
 ↓
model
```

The model must still work in flow-only degraded mode.

---

# 34. Inference Behavior for PCAP

If a PCAP is uploaded:

```text
PCAP
 ↓
packet feature extraction
 ↓
5-sec windows
 ↓
packet state
```

If a corresponding flow CSV is also supplied by the UI or an inference directory contains a matched flow file, join them.

If no flow features are available, the application may support a packet-only fallback only if the model has been explicitly trained for that mode. Otherwise display a clear unsupported-input message rather than silently producing a malformed prediction.

---

# 35. Recommended Final Project Structure

Use approximately:

```text
CyberPulse/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── docs/
│   └── architecture.md
│
├── data/
│   ├── raw/
│   │   ├── cic2017/
│   │   │   ├── csv/
│   │   │   └── pcap/
│   │   └── ctu13/
│   └── processed/
│       ├── cic2017/
│       │   ├── cleaned_flow.parquet
│       │   ├── packet_features.parquet
│       │   ├── window_states.parquet
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   └── test.parquet
│       └── ctu13/
│           └── ...
│
├── data_prep/
│   ├── clean_cic2017.py
│   ├── clean_ctu13.py
│   ├── inspect_pcap.py
│   ├── extract_packet_features.py
│   ├── build_temporal_dataset.py
│   └── data_quality_report.py
│
├── mitre_mapping/
│   └── attack_stage_mapping.json
│
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── flow_csv_loader.py
│   │   └── ctu13_loader.py
│   │
│   ├── features/
│   │   ├── flow_features.py
│   │   ├── packet_features.py
│   │   ├── window_state.py
│   │   ├── forecast_targets.py
│   │   └── build_temporal_dataset.py
│   │
│   ├── model/
│   │   └── temporal_world_model.py
│   │
│   ├── training/
│   │   ├── dataset.py
│   │   ├── train_world_model.py
│   │   └── train_baselines.py
│   │
│   ├── explainability/
│   │   └── integrated_gradients.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── lead_time.py
│   │   └── run_eval.py
│   │
│   └── inference/
│       └── predict.py
│
├── baseline_results/
│   └── ...
│
├── checkpoints/
│   ├── temporal_world_model.pt
│   └── preprocessing_stats.json
│
├── metrics/
│   ├── results.md
│   ├── forecast_curves.png
│   └── feature_attributions.json
│
├── dashboard/
│   └── streamlit_app.py
│
├── api/
│   └── main.py
│
└── scripts/
    └── run_full_pipeline.ps1
```

The exact number of modules may be reduced if the code is cleaner, but responsibilities must remain separated.

---

# 36. Configuration File

Replace the old GT-RSSM config with something like:

```yaml
data:
  delta_t_seconds: 5
  window_overlap: 0.0
  history_windows: 12
  horizon_5_windows: 1
  horizon_30_windows: 6
  horizon_60_windows: 12

model:
  flow_input_dim: 28
  packet_input_dim: 12
  branch_dim: 32
  fused_dim: 64
  hidden_dim: 64

training:
  epochs: 40
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 7
  gradient_clip_norm: 1.0
  seed: 42

loss:
  state_weight: 0.3
  attack_weight: 1.0
  stage_weight: 0.5

thresholding:
  target_validation_fpr: 0.05

paths:
  cic2017_csv_dir: data/raw/cic2017/csv
  cic2017_pcap: data/raw/cic2017/pcap/Wednesday-workingHours.pcap
  ctu13_dir: data/raw/ctu13
  processed_dir: data/processed
  checkpoint_dir: checkpoints
  metrics_dir: metrics
  mitre_mapping: mitre_mapping/attack_stage_mapping.json
```

Do not hard-code these values throughout the project.

---

# 37. Requirements

Reduce the dependency list where possible.

Required core dependencies:

```text
numpy
pandas
pyarrow
scikit-learn
xgboost
shap
scapy
torch
pyyaml
tqdm
matplotlib
plotly
streamlit
fastapi
uvicorn
python-multipart
captum
```

No PyTorch Geometric.

No ONNX runtime.

No graph-specific dependency.

---

# 38. Build Order — Agent Must Follow This Sequence

## Step 1 — Repository audit

Inspect all current source files.

Create a short migration report listing:

- files to delete
- files to rewrite
- files to reuse
- old generated artifacts

Do not modify raw data in this step.

## Step 2 — Remove GT-RSSM active code

Delete/retire all old model and Stage-1/Stage-2 training modules.

Confirm that importing the project does not import `GTRSSM`, `RSSM`, graph attention, or old losses.

## Step 3 — Fix data preprocessing

Run:

```text
clean CIC
inspect PCAP
extract packet features
```

Verify all reports before model work.

## Step 4 — Build flow window states

Generate 5-second non-overlapping network-level states.

Verify feature count and no NaNs/infinities.

## Step 5 — Build packet window states

Use streaming PCAP extraction.

Generate the compact packet state vector.

Verify:

```text
packet_features_available = 1
```

for windows with PCAP coverage.

## Step 6 — Join flow + packet state

Generate:

```text
window_states.parquet
```

## Step 7 — Build future labels

Generate:

```text
y_5
y_30
y_60
future_stage_30
next_state
```

Print the positive rates.

## Step 8 — Split with purge

Create:

```text
train
val
test
```

with temporal ordering and a 60-second purge gap.

## Step 9 — Train Logistic Regression

Use the future +30-second target.

Save baseline metrics.

## Step 10 — Train XGBoost

Use the same data split and +30-second target.

Save baseline metrics and SHAP feature importance.

## Step 11 — Train the GRU world model

Train one supervised model with:

- state transition loss
- +5s attack loss
- +30s attack loss
- +60s attack loss
- masked future stage loss

## Step 12 — Add recursive rollout

From the final observed hidden state, recursively predict future state and future risk for 60 seconds.

## Step 13 — Evaluate

Produce the complete metrics table.

## Step 14 — Run the packet-vs-flow ablation

Compare:

```text
flow-only
vs
flow+packet
```

## Step 15 — Add Integrated Gradients

Verify non-zero sensible attributions.

## Step 16 — Rewrite Streamlit

Use only the new model terminology.

## Step 17 — Rewrite API

Make FastAPI call the same `predict()` function used by Streamlit.

## Step 18 — Rewrite README and architecture document

Document the new architecture, data flow, metrics and limitations.

---

# 39. Required Data-Quality Checks

The preprocessing script must print a report similar to:

```text
=== CIC-IDS2017 DATA QUALITY ===
Raw flow rows                 : ...
Clean flow rows               : ...
Duplicate rows removed        : ...
Invalid timestamp rows        : ...
Inf/NaN rows removed          : ...
Attack rows                   : ...
Benign rows                   : ...

=== WINDOWING ===
Window size                   : 5s
Overlap                       : 0s
Window count                  : ...

=== PACKET DATA ===
PCAP packets processed        : ...
Packet windows                : ...
Packet-covered windows        : ...
Packet coverage percentage    : ...

=== FORECAST TARGETS ===
+5s positive rate             : ...
+30s positive rate            : ...
+60s positive rate            : ...

=== FINAL DATASET ===
Feature dimension             : ...
Train samples                 : ...
Validation samples            : ...
Test samples                  : ...
```

Stop and investigate if:

- positive rate is 0%
- positive rate is 100%
- packet coverage is unexpectedly 0%
- feature columns contain NaN/inf
- timestamps are out of order
- train/val/test boundaries overlap
- stage labels are entirely one class when attack data should contain multiple stages

---

# 40. Required Unit Tests / Sanity Tests

Create tests or equivalent assertions for:

## Test 1 — No future leakage

Verify every sample's feature timestamps are `<= t` and every future target timestamp is `> t`.

## Test 2 — Split isolation

Verify no timestamp or window ID is shared across train/val/test.

## Test 3 — Scaler isolation

Verify scaler statistics are computed from train only.

## Test 4 — Packet availability

Verify packet features are never treated as real measurements when availability is 0.

## Test 5 — Sequence shape

Expected:

```text
[batch, 12, feature_dim]
```

## Test 6 — Model output

Verify:

```text
attack_logits.shape == [batch, 3]
stage_logits.shape  == [batch, 7]
next_state.shape    == [batch, feature_dim]
```

## Test 7 — Rollout

Verify 12 forecast steps can be generated without NaNs.

## Test 8 — Explanation

Verify Integrated Gradients produces non-zero attributions on a non-trivial sample.

---

# 41. What the Agent Must NOT Do

Do not:

- keep GAT/RSSM code and call it the new model
- train only for one epoch and report final metrics
- use random row splitting for the final result
- create graphs or 128-node padded tensors
- use SMOTE on temporal rows
- use future labels as features
- tune the test threshold
- fit the scaler on the full dataset
- use raw IP strings as model inputs
- download huge new datasets during the core implementation
- claim packet-level modeling when the packet branch is disabled
- report accuracy alone
- claim a specific target F1 before experiments are actually run
- keep old GT-RSSM checkpoints as fallback inference weights
- build an XGBoost surrogate just to explain the GRU
- create a second incompatible preprocessing implementation for the dashboard

---

# 42. Required Final Results Table

The final `metrics/results.md` should contain something like:

| Model | Horizon | PR-AUC | F1 | Precision | Recall | FPR | Brier | Lead Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 30s | ... | ... | ... | ... | ... | ... | ... |
| XGBoost | 30s | ... | ... | ... | ... | ... | ... | ... |
| Temporal GRU | 30s | ... | ... | ... | ... | ... | ... | ... |

Then a second table:

| Model | Stage Macro-F1 | CTU-13 F1 | CTU-13 PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | ... | ... | ... |
| XGBoost | ... | ... | ... |
| Temporal GRU | ... | ... | ... |

And an ablation:

| Configuration | +30s PR-AUC | +30s F1 | Median Lead Time |
|---|---:|---:|---:|
| Flow-only | ... | ... | ... |
| Flow + Packet | ... | ... | ... |

Do not fabricate numbers.

---

# 43. Final Demo Story

The demo should communicate:

```text
1. CyberPulse receives recent network traffic.

2. Flow-level telemetry captures traffic volume, connections,
   flags, timing and port behavior.

3. Packet-level telemetry captures TTL, TCP window,
   fragmentation, payload, retransmissions and scan behavior.

4. The system compresses both into a compact network state.

5. A lightweight GRU learns how network state evolves over time.

6. The world model predicts the next network state and forecasts
   attack risk at +5s, +30s and +60s.

7. The system predicts the likely future MITRE ATT&CK stage.

8. Integrated Gradients identifies the traffic features and
   historical windows driving the warning.
```

The central phrase for the project is:

> **Learn the evolution of network state, not just the current attack label.**

---

# 44. What Counts as a Successful Implementation

The implementation is complete only when all of the following are true:

### Architecture

- GT-RSSM is removed from the active path.
- No graph tensors are used.
- No RSSM/KL machinery remains.
- The new model uses both flow and packet branches.
- A recurrent state-transition model is implemented.
- Future attack probability is explicitly supervised.

### Data

- CIC-IDS2017 flow data is cleaned correctly.
- Wednesday PCAP is streamed and converted into packet features.
- Flow and packet features are joined by time window.
- Future labels are constructed correctly.
- Temporal leakage tests pass.

### Training

- LR baseline exists.
- XGBoost baseline exists.
- GRU world model trains for multiple epochs with early stopping.
- +30s PR-AUC is the main selection metric.
- No test-set threshold tuning occurs.

### Evaluation

- PR-AUC, F1, precision, recall and FPR are reported.
- Brier score is reported.
- Lead time is reported.
- Future MITRE stage Macro-F1 is reported.
- Cross-dataset evaluation is attempted on CTU-13 when available.
- Flow-only vs flow+packet ablation is reported.

### Explainability

- Neural model produces Integrated Gradients attributions.
- Dashboard displays top contributing flow and packet features.

### Demo

- Streamlit works offline.
- Same inference function is used by Streamlit and FastAPI.
- Dashboard shows the future probability curve rather than only current classification.

---

# 45. Final Implementation Principle

The goal is **not** to make the smallest possible model.

The goal is to make the **simplest model that still satisfies the NTRO problem statement and produces trustworthy forecasting experiments**.

The final architecture should therefore retain exactly these capabilities:

```text
FLOW + PACKET TELEMETRY
          ↓
COMPACT NETWORK STATE
          ↓
12-WINDOW TEMPORAL HISTORY
          ↓
64-D GRU WORLD STATE
          ↓
STATE TRANSITION PREDICTION
          ↓
+5s / +30s / +60s ATTACK FORECAST
          ↓
FUTURE MITRE STAGE
          ↓
RECURSIVE 60s ROLLOUT
          ↓
INTEGRATED-GRADIENT EXPLANATION
          ↓
STREAMLIT DEFENDER DASHBOARD
```

This replaces the computationally expensive GT-RSSM implementation while preserving the strongest requirements of SIH26153: temporal modeling, future attack forecasting, progression/stage prediction, use of both flow and packet telemetry, explainability, and an offline working prototype.
