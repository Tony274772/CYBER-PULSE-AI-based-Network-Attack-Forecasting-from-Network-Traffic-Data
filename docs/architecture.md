# GT-RSSM Architecture — Network Attack Forecasting

## Problem Statement (SIH26153)
Develop an AI-based system that forecasts network attacks from traffic data
**before** a compromise completes, mapping to recognised attack stages
(e.g. MITRE ATT&CK), with explainability and an interactive dashboard.

## Architecture: GT-RSSM (GAT + RSSM World Model)

### Core Idea
Instead of classifying past traffic, we treat the network as a **world**: each
5-second window produces a graph of host interactions. A Recurrent State-Space
Model (RSSM) learns the *dynamics* of this world — how the network evolves —
and can then **imagine forward K steps** without new observations, generating a
probability curve of future infiltration.

### Pipeline
```
Network Traffic → Windowed Graphs → Dense Graph Attention → RSSM → Belief State
                                                                    ↓
                    ┌─────────────┬─────────────┬──────────┬────────┘
                    ↓             ↓             ↓          ↓
              Reconstruction  Infiltration   MITRE     KL Surprise
              (ELBO, MSE)     prob (sigmoid) (7-way)   (anomaly signal)
```

### Dense Graph Attention Encoder (replaces PyG GATConv)
- 2 layers, 4 heads, 64-dim embeddings
- Pure PyTorch dense attention with adjacency + edge-feature bias
- Mathematically equivalent to GAT; zero external dependency
- **Rationale**: PyG has brittle version-locked wheels and export issues.
  Dense attention over ≤128 nodes per window is equally expressive and
  exports trivially.

### RSSM (Dreamer-v1 / PlaNet lineage)
- GRUCell deterministic path (h_dim=128)
- Diagonal Gaussian stochastic latent (z_dim=32) with reparameterization
- Posterior q(z|h,e) uses graph embedding; Prior p(z|h) does not
- **Rationale**: Gaussian latents (not categorical Dreamer-v3) avoid
  straight-through estimator complexity at this project's scale.

### Read-out Heads
| Head | Output | Loss |
|---|---|---|
| Reconstruction | Pooled edge-feature vector | MSE |
| Infiltration | Sigmoid scalar (binary) | Focal loss (γ=2) |
| MITRE Stage | 7-way softmax | Class-weighted CE |
| Anomaly | KL(q‖p) — no trainable parameters | Direct read-out |

### MITRE ATT&CK Mapping (7 stages)
Benign, Reconnaissance (TA0043), Initial Access (TA0001), Lateral Movement
(TA0008), Command & Control (TA0011), Exfiltration (TA0010), **Impact/DoS
(TA0040)**. The sixth stage is added because CIC-IDS2017's dominant attack
type (DoS/DDoS) maps to MITRE Impact, not to any of the five originally listed
example stages. This is documented transparently as an engineering decision.

### Two-Stage Training
1. **Stage 1 (unsupervised)**: reconstruction + KL on all traffic (no labels).
   β-annealing 0→1 over first 10% of steps to prevent posterior collapse.
2. **Stage 2 (supervised)**: full loss on labeled CIC-IDS2017 + CTU-13 windows.
   Encoder+RSSM at 1e-4 lr (preserve dynamics), heads at 1e-3.

### K-Step Imagination Forecast
At inference, unroll the prior-only RSSM forward K steps (K=5/10/20 ≈
25/50/100s ahead), reading the infiltration and stage heads at each step.
This generates a probability curve *before* the attack materializes —
the "lead time" property.

### Explainability Stack
- **Attention weights** → which host-pairs mattered (edge highlighting)
- **Integrated Gradients** (captum) → which raw features mattered
- **XGBoost surrogate + SHAP** → tabular feature attribution
- **KL surprise** → possible unseen attack technique

### Engineering Decisions (Section 0)
| Original | Replacement | Rationale |
|---|---|---|
| PyG GATConv | Dense masked attention | Removes fragile dependency |
| Categorical Dreamer-v3 latents | Diagonal Gaussian (Dreamer-v1) | Simpler, same lineage |
| ONNX export | PyTorch state_dict + CPU eval | No dynamic control-flow ONNX pain |
| FastAPI + Streamlit over HTTP | Streamlit imports inference in-process | One fewer failure point |

### Datasets
- **CIC-IDS2017** (primary): ~225 MB flow CSVs, Wednesday PCAP for packet features
- **CTU-13** (secondary): Botnet/C2 coverage, cross-dataset generalization
- NF-UNSW-NB15-v2 (optional): cross-dataset surprise AUC test

### Serving
Streamlit app imports ``src/inference/predict.py`` directly (in-process, no
HTTP hop). Optional FastAPI wrapper available at ``api/main.py``.
