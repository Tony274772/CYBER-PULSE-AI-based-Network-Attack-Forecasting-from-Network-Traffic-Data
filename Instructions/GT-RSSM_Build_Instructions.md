# GT-RSSM: Network Attack Forecasting — Full Build Specification
### For SIH26153 (NTRO — AI-based Network Attack Forecasting from Network Traffic Data)
### Target reader: an AI coding agent implementing this repository from an empty folder

This document is self-contained. Follow it top to bottom. Where a design choice is
not obvious, this document tells you exactly what to do and why — do not
substitute your own judgement on architecture-level decisions (model shapes,
loss terms, windowing, dataset choice); you may use your own judgement for
routine engineering (variable names, exact library call signatures, refactors)
as long as behavior matches this spec.

The Data folder will be populated manually by the project owner following
Section 2. Do not attempt to download datasets yourself from inside a
sandboxed build step unless a step explicitly says to fetch a specific small
file — assume `data/raw/...` already contains what Section 2 describes.

---

## 0. What this revision changes vs. the original proposal, and why

The original architecture document (`network_attack_forecasting_architecture.md`,
codenamed **GT-RSSM**: GAT spatial encoder + Dreamer-style Recurrent
State-Space Model + three read-out heads + KL-based anomaly signal) is
conceptually correct and well-aligned to the problem statement — keep the
narrative, keep the name, keep the "world model, not a classifier" pitch.
Three implementation-level substitutions are made below, purely to remove
fragile dependencies and failure points, without weakening the technical
story you present to judges:

| Original choice | Problem with it under a real build | Replacement in this spec |
|---|---|---|
| PyTorch Geometric `GATConv` over per-window graphs | PyG has brittle version-locked wheels against a given torch/CUDA build, and its sparse/scatter ops frequently fail to trace/export cleanly — a real risk for a small team with limited debugging time | A **dense, masked multi-head attention layer written in plain PyTorch** over the (small, padded) set of active hosts per window. Mathematically the same idea as GAT (attention over graph neighbours); implemented with ordinary batched matmul + softmax + an adjacency mask. Per-window host counts in these datasets are small (tens, not millions), so dense attention is not a performance concern, and it removes an entire dependency + a whole class of "why won't this export/install" bugs |
| Categorical Dreamer-v3-style latents with straight-through gradients | Correct but adds real implementation complexity (straight-through estimator, unimix, free-bits) for marginal benefit at this project's scale | **Diagonal Gaussian latents** with the standard VAE reparameterization trick. Same posterior/prior/KL machinery, same "world model" lineage (this is exactly the original Dreamer-v1/PlaNet RSSM, which is *also* citable prior art — cite Hafner et al.'s "Learning Latent Dynamics for Planning from Pixels" / PlaNet in your architecture doc, in addition to Dreamer) |
| ONNX export as the serving path | Exporting a model that combines a custom attention layer and a recurrent cell that is *unrolled a variable number of steps* is a known ONNX pain point (dynamic control flow, opset gaps) — do not spend build time on this | **Plain PyTorch checkpoint (`state_dict`), loaded once, run in `eval()` mode on CPU.** A model this small does sub-second CPU inference per window with no export step at all. Mention ONNX export as a documented future extension, not a build blocker |
| Two networked services (FastAPI inference server + Streamlit calling it over HTTP) for the demo | Two coordinated processes is two things that can fail to start correctly on a judge's machine at demo time | **The Streamlit app imports the inference module directly and calls it in-process.** No network hop, no second process. A thin FastAPI wrapper is still built (it's an easy, low-risk add and gives you an "API" bullet point for the README) but it is *not* on the critical path of the live demo |
| CSE-CIC-IDS2018 (AWS-hosted, ~220 GB) as primary dataset | Requires AWS CLI + registry access and is far larger than a hackathon team can practically download and iterate on | **CIC-IDS2017** as primary (flow CSVs ≈ 225 MB total, well-documented, 14 human-readable attack labels, single day-PCAPs are a few GB each so packet-level extraction stays tractable). CTU-13 (≈1.9 GB total, no registration) as the second dataset for Botnet/C2 coverage and cross-dataset generalization. NF-UNSW-NB15-v2 (a few hundred MB CSV, no PCAP needed) as an optional third dataset purely for the cross-dataset "surprise AUC" generalization test |

Everything else in the original proposal — the GAT→RSSM→three-heads shape,
the two-stage (unsupervised pretrain → supervised fine-tune) training recipe,
the KL-as-anomaly-signal novelty claim, the explainability stack (attention +
integrated gradients + SHAP surrogate), the evaluation protocol (lead-time,
Brier score, cross-dataset F1, held-out-category surprise AUC), and the
deliverables mapping — is preserved and specified precisely below.

One additional honest correction: the problem statement lists five MITRE
ATT&CK stages (Reconnaissance, Initial Access, Lateral Movement, Command &
Control, Exfiltration) as an **example** mapping ("map to recognised attack
stages, e.g. MITRE ATT&CK"), but CIC-IDS2017 is dominated by DoS/DDoS traffic,
which MITRE ATT&CK classifies under the **Impact** tactic (TA0040) — not one
of the five listed. Forcing DoS into one of the five would misrepresent the
data. This spec adds an explicit sixth stage bucket, **Impact / DoS**, and
documents this openly in `mitre_mapping/attack_stage_mapping.json` — this
kind of transparent scope decision reads as engineering maturity to judges,
not a weakness. State this explicitly in your architecture document.

---

## 1. Final Architecture — GT-RSSM (build-ready revision)

```
DATA INGESTION
  CIC-IDS2017 flow CSVs ──┐
  CIC-IDS2017 PCAP subset ┼──► Unified per-window Edge/Node Feature Tables
  CTU-13 binetflow files ─┤        (one row per (src_ip,dst_ip,window) and
  NF-UNSW-NB15-v2 CSV ────┘         one row per (ip,window))
                 │
                 ▼
FEATURE ENGINEERING & GRAPH CONSTRUCTION
  Sliding window Δt (default 5s, 50% overlap)
  Per window t: build G_t = (V_t active IPs, E_t flows between them)
  Edge feature = flow-level vector ⊕ packet-level vector ⊕ availability mask
  Node feature = degree, byte volume in/out, distinct-dst-port count
  Pad/mask to N_max = 128 nodes per window (small networks ⇒ dense attn is fine)
  Sequence of W consecutive windows = one training sequence (default W=20 ⇒ 100s)
                 │
                 ▼
GT-RSSM WORLD MODEL
  G_t ──► Dense Masked Graph-Attention Encoder ──► graph embedding e_t
                                 │
              ┌──────────────────────────────────────┐
              │  h_t = GRU(h_{t-1}, [z_{t-1}, e_t])    (deterministic path)
              │  z_t ~ q(z_t | h_t, e_t) = N(μ_q,σ_q)   (posterior, training)
              │  ẑ_t ~ p(ẑ_t | h_t)      = N(μ_p,σ_p)   (prior, imagination)
              │  belief state s_t = [h_t ; z_t]
              └──────────────────────────────────────┘
                                 │
     ┌───────────────┬──────────────────┬───────────────────┬──────────────┐
     ▼               ▼                  ▼                   ▼              │
 Reconstruction  Infiltration      MITRE stage head    Surprise/Anomaly ◄───┘
 head (ELBO)     probability head  (7-way softmax:     KL(q‖p), read
 self-supervised (sigmoid)         Benign, Recon,       directly off the
                                    Initial Access,      ELBO term, no
                                    Lateral Movement,    extra parameters
                                    C2, Exfiltration,
                                    Impact/DoS)
                 │
                 ▼
K-STEP FORWARD SIMULATION ("imagination", prior-only rollout)
  for k in 1..K: h=GRU(h, ẑ); ẑ~prior(h); read infiltration_head, mitre_head
  → infiltration probability curve for t+1..t+K
  → predicted MITRE-stage trajectory for t+1..t+K
                 │
                 ▼
EXPLAINABILITY
  Dense-attention weights → which host-pairs mattered (edge highlighting)
  Integrated Gradients (captum) on edge feature vector → which raw features mattered
  XGBoost surrogate (trained to imitate infiltration_head output) + SHAP → tabular attribution
  Temporal attention pooling over the W-window input → which past windows mattered
                 │
                 ▼
SERVING (offline, single process)
  Streamlit app imports src/inference directly (in-process call, no HTTP hop)
  Optional thin FastAPI wrapper for an "API" deliverable (not on demo critical path)
```

---

## 2. Data Acquisition (manual — the project owner does this, then places files as below)

Download the following. No dataset here requires payment; CIC-IDS2017
requires filling a short access-request form on the official page (approved
quickly / automatically in practice); CTU-13 and the NetFlow dataset are
direct downloads.

### 2.1 CIC-IDS2017 (primary dataset)
- Official source (fills a short form, then emails a download link):
  `https://www.unb.ca/cic/datasets/ids-2017.html`
- What to get:
  - **`GeneratedLabelledFlows.zip`** (a.k.a. `MachineLearningCVE.zip` in some
    mirrors) — 8 CSV files, one per capture segment (Monday through Friday,
    with Tuesday/Wednesday/Thursday/Friday split into morning/afternoon
    segments). This is the flow-level source, ~225 MB unzipped total.
  - **One raw PCAP file**: `Wednesday-workingHours.pcap` only (a few GB — do
    **not** download all five days' PCAPs, they total ~50+ GB and you only
    need one day for the packet-level feature module). Wednesday contains
    the DoS Hulk / DoS GoldenEye / DoS Slowloris / DoS Slowhttptest /
    Heartbleed attacks, which is enough to demonstrate the packet-level
    pipeline end to end.
  - If the official form is slow to respond, equivalent mirrors exist on
    Kaggle (search "CIC-IDS2017", e.g. datasets published by `cicdataset`,
    `dhoogla`, or `biprobarai`) and on Hugging Face (`bencorn/CICIDS2017`
    mirrors the raw PCAPs and zips). Verify the mirror's file list matches
    the official one before trusting it.
- Place files at:
  ```
  data/raw/cic2017/csv/*.csv            # all 8 files from GeneratedLabelledFlows.zip, unzipped, flat
  data/raw/cic2017/pcap/Wednesday-workingHours.pcap
  ```

### 2.2 CTU-13 (secondary dataset — Botnet C2/exfiltration coverage + cross-dataset eval)
- Direct download, no registration:
  `https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2`
  (~1.9 GB; also mirrored at `https://www.stratosphereips.org/datasets-ctu13`)
- Extract the archive. It contains one subfolder per scenario (13 total),
  each with a bidirectional NetFlow file (extension `.binetflow`, which is a
  CSV with columns `StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr,
  Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label`) and,
  for most scenarios, the original PCAP.
- Place the **entire extracted tarball contents** at:
  ```
  data/raw/ctu13/**            # keep the original folder structure from the tarball
  ```
  Do not rename or flatten anything — the loader (Section 7.2) discovers
  files by recursive glob (`**/*.binetflow`), not by hardcoded path, because
  exact subfolder names differ slightly across mirrors.
- The `Label` field is a free-text string, not a clean category — e.g. it
  contains substrings like `Botnet`, `Normal`, `Background`. Section 6.2
  specifies the exact substring rules used to bucket it.

### 2.3 NF-UNSW-NB15-v2 (optional — cross-dataset generalization test only)
- Source: University of Queensland NIDS datasets page,
  `https://staff.itee.uq.edu.au/marius/NIDS_datasets/` → NetFlow v2 Datasets
  → `NF-UNSW-NB15-v2` (a single CSV, already flow-level, already labelled,
  a few hundred MB — no PCAP parsing needed for this one at all).
- Place at:
  ```
  data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv
  ```
- If time runs short, this dataset is the first thing to cut — it is only
  used for the optional cross-dataset surprise-AUC ablation (Section 8.7),
  not for core training. CIC-IDS2017 + CTU-13 alone satisfy every mandatory
  deliverable.

### 2.4 What NOT to fetch (explicitly out of scope for this build)
CSE-CIC-IDS2018, CICIoT2023, and the LANL Authentication Dataset are **not**
part of this build. They were in the original brief's dataset list as
options, not requirements, and adding them multiplies data-engineering time
for marginal narrative benefit given two datasets already cover Recon,
Brute-Force, Web-Attack, Infiltration, Botnet/C2, and DoS/Impact. Mention them
only in the "Future Extensions" section of your submission as stretch goals.

---

## 3. Repository Structure

```
network-attack-forecasting/
├── README.md
├── requirements.txt
├── docs/
│   └── architecture.md                # condensed 2-page version of Section 1 + 0 for submission
├── configs/
│   └── default.yaml                    # Section 9
├── data/
│   ├── raw/                            # as populated per Section 2 (gitignored)
│   └── processed/                      # pipeline output: windowed parquet feature store (gitignored)
├── mitre_mapping/
│   └── attack_stage_mapping.json       # Section 6.2, verbatim content given below
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── flow_csv_loader.py          # 7.1
│   │   ├── ctu13_loader.py             # 7.1
│   │   └── pcap_parser.py              # 7.2
│   ├── features/
│   │   ├── flow_features.py            # 7.3
│   │   ├── packet_features.py          # 7.3
│   │   ├── windowing.py                # 7.4 — the core, most detailed spec section
│   │   └── build_dataset.py            # CLI entrypoint, runs the full pipeline to parquet
│   ├── model/
│   │   ├── dense_graph_attention.py    # 7.5.1
│   │   ├── rssm.py                     # 7.5.2
│   │   ├── heads.py                    # 7.5.3
│   │   └── gt_rssm.py                  # 7.5.4 — assembles the full model
│   ├── training/
│   │   ├── dataset.py                  # PyTorch Dataset/DataLoader over the parquet store
│   │   ├── losses.py                   # 7.6.1
│   │   ├── pretrain_stage1.py          # 7.6.2 CLI entrypoint
│   │   └── finetune_stage2.py          # 7.6.3 CLI entrypoint
│   ├── baseline/
│   │   └── logreg_baseline.py          # 7.7
│   ├── explainability/
│   │   ├── attention_export.py         # 7.8.1
│   │   ├── integrated_gradients.py     # 7.8.2
│   │   └── shap_surrogate.py           # 7.8.3
│   ├── evaluation/
│   │   ├── metrics.py                  # 7.9
│   │   └── run_eval.py                 # CLI entrypoint, produces the final results table
│   └── inference/
│       └── predict.py                  # 7.10 — single unified entrypoint used by both UIs
├── dashboard/
│   └── streamlit_app.py                # 7.11
├── api/
│   └── main.py                         # 7.12 (optional, not demo-critical)
├── checkpoints/
│   └── gt_rssm_v1.pt                   # trained weights (deliverable)
└── scripts/
    └── run_full_pipeline.sh            # convenience script chaining every step in Section 8
```

---

## 4. Environment

Python 3.11. Deliberately dependency-light (no PyTorch Geometric, no ONNX
runtime required for the core build):

```
torch>=2.2
numpy
pandas
pyarrow            # parquet I/O
scikit-learn
xgboost
shap
captum             # integrated gradients
scapy              # PCAP parsing — use PcapReader (streaming), never rdpcap (loads whole file into RAM)
pyyaml
tqdm
matplotlib
plotly
streamlit
fastapi
uvicorn
python-multipart
```

Install with `pip install -r requirements.txt`, then re-freeze exact resolved
versions into `requirements.txt` once the environment installs cleanly, for
reproducibility (the README deliverable should state the exact versions that
were actually used).

---

## 5. Config System (`src/config.py`)

A single dataclass or pydantic model loaded from `configs/default.yaml`,
covering: window size `delta_t_seconds` (default 5), window overlap fraction
(default 0.5), sequence length `W` in windows (default 20), max nodes per
graph `N_max` (default 128), model dims (Section 7.5), training
hyperparameters (Section 7.6), and dataset paths. Every script in Section 8
takes `--config configs/default.yaml` plus optional CLI overrides
(`--delta-t 10`) for the ablation runs in Section 8.6. Do not hardcode any of
these numbers inside model/feature code — read them from config so the
Δt-ablation (2s/5s/10s) required by the evaluation protocol is a one-flag
change, not a code edit.

---

## 6. Data Schema Reference

### 6.1 Known gotchas — handle these explicitly, they will silently corrupt results otherwise

- **CIC-IDS2017 CSV column headers have leading whitespace** (e.g.
  `' Destination Port'`, `' Flow Duration'`). First line of every loader:
  `df.columns = df.columns.str.strip()`.
- Some CIC-IDS2017 CSVs contain **duplicated header rows embedded mid-file**
  (an artifact of how the files were concatenated). After loading, drop any
  row where the `Label` column literally equals the string `"Label"`.
- The `Flow Bytes/s` and `Flow Packets/s` columns contain `Infinity` and
  `NaN` for zero-duration flows. Replace `inf`/`-inf` with `NaN`, then either
  drop those rows or impute with the column median — do not leave `inf` in
  any tensor fed to the model, it will silently propagate to `NaN` loss.
- The `Timestamp` column's day/month order is ambiguous from the string
  alone. Parse with `pandas.to_datetime(..., dayfirst=True, errors="coerce")`
  and sanity-check the resulting dates fall inside July 3–7, 2017 (the
  documented capture window); log a warning and drop any row that fails this
  check rather than silently mis-ordering the sequence.
- **CTU-13 `Label` field** is not a clean category — see 6.2 for exact
  substring rules.
- CTU-13's `StartTime` format is `YYYY/MM/DD HH:MM:SS.ffffff` — parse
  directly with `pd.to_datetime`, no ambiguity here.
- When only flow-level data is available for a window (no matching PCAP —
  true for every CTU-13 window and every CIC-IDS2017 day except Wednesday),
  **do not attempt to synthesize packet-level features.** Zero-fill the
  packet-level portion of the edge vector and set an explicit
  `packet_features_available` binary flag to 0 for that edge. This flag is
  part of the input vector, not a post-hoc filter — the model must be
  trained on a mix of flag=0 and flag=1 edges from the start (this is what
  gives you the "degraded mode still works" claim called out in the original
  feasibility section).

### 6.2 MITRE ATT&CK stage mapping — save verbatim as `mitre_mapping/attack_stage_mapping.json`

```json
{
  "stage_order": ["Benign", "Reconnaissance", "Initial_Access", "Lateral_Movement",
                   "Command_And_Control", "Exfiltration", "Impact_DoS"],
  "mitre_tactic_ids": {
    "Reconnaissance": "TA0043",
    "Initial_Access": "TA0001",
    "Lateral_Movement": "TA0008",
    "Command_And_Control": "TA0011",
    "Exfiltration": "TA0010",
    "Impact_DoS": "TA0040"
  },
  "cic_ids2017_label_map": {
    "BENIGN": "Benign",
    "PortScan": "Reconnaissance",
    "FTP-Patator": "Initial_Access",
    "SSH-Patator": "Initial_Access",
    "Web Attack \u2013 Brute Force": "Initial_Access",
    "Web Attack \u2013 XSS": "Initial_Access",
    "Web Attack \u2013 Sql Injection": "Initial_Access",
    "Heartbleed": "Initial_Access",
    "Infiltration": "Lateral_Movement",
    "Bot": "Command_And_Control",
    "DoS slowloris": "Impact_DoS",
    "DoS Slowhttptest": "Impact_DoS",
    "DoS Hulk": "Impact_DoS",
    "DoS GoldenEye": "Impact_DoS",
    "DDoS": "Impact_DoS"
  },
  "ctu13_label_substring_rules": [
    {"contains": "Normal", "stage": "Benign"},
    {"contains": "Background", "stage": null, "note": "exclude from supervised heads; usable only in Stage-1 unsupervised pretraining pool"},
    {"contains_any": ["CC", "Botnet"], "excludes": ["Scan"], "stage": "Command_And_Control"},
    {"contains_any": ["Scan", "PortScan"], "stage": "Reconnaissance"}
  ],
  "nf_unsw_nb15_v2_label_map": {
    "Benign": "Benign",
    "Reconnaissance": "Reconnaissance",
    "Exploits": "Initial_Access",
    "Backdoor": "Initial_Access",
    "Shellcode": "Initial_Access",
    "Analysis": "Initial_Access",
    "Fuzzers": "Initial_Access",
    "Worms": "Lateral_Movement",
    "DoS": "Impact_DoS",
    "Generic": "Impact_DoS"
  },
  "note_on_impact_dos": "The problem statement lists five example stages; Impact/DoS (MITRE TA0040) is added as a sixth because CIC-IDS2017 is DoS/DDoS-heavy and forcing that traffic into one of the five would misrepresent it. Document this decision explicitly in the architecture doc.",
  "note_on_ctu13_rules": "CTU-13 Label strings are free text (e.g. 'flow=From-Botnet-V42-TCP-CC107-N1-HTTP'); apply the substring rules in order and take the first match. This is an approximation and should be described as such."
}
```

The CTU-13 rule engine: implement `map_ctu13_label(label: str) -> Optional[str]`
that lower-cases the label, checks rules **in the array order given above**,
and returns the first matching `stage`, or `None` for `Background` (meaning:
usable for Stage-1 pretraining, excluded from Stage-2 supervised loss).

### 6.3 Flow-level feature vector (per host-pair, per window)

```
flow_duration_sum, tot_fwd_pkts, tot_bwd_pkts, tot_fwd_bytes, tot_bwd_bytes,
flag_syn_count, flag_ack_count, flag_fin_count, flag_rst_count, flag_psh_count, flag_urg_count,
iat_mean, iat_std, iat_max, iat_min,
fwd_bwd_byte_ratio, fwd_bwd_pkt_ratio,
flow_bytes_per_sec, flow_pkts_per_sec
```
For CIC-IDS2017: pull the equivalent already-computed CICFlowMeter columns
directly and re-aggregate any flows that fall in the same window for the
same host pair (sum counts/bytes, re-derive ratios and rates, mean the IAT
stats weighted by flow duration). For CTU-13: derive the same fields from
raw `binetflow` rows (protocol/flag information there is coarser — TCP state
strings like `S_RA` — parse what's available, zero-fill flag counts that
cannot be derived and flag it in a per-field availability mask analogous to
6.1's packet-level flag).

### 6.4 Packet-level feature vector (per host-pair, per window — Wednesday PCAP only)

```
ttl_mean, ttl_var,
tcp_window_mean, tcp_window_std,
ip_frag_flag_ratio,
payload_size_mean, payload_size_std,
retransmission_count,
port_scan_sequential_score, port_scan_randomised_score
```
`retransmission_count`: count packets whose (src,dst,sport,dport,seq) has
been seen before within the same TCP stream. `port_scan_*_score`: over the
set of distinct destination ports contacted by this src within the window,
compute (a) the fraction of ports that are numerically sequential
consecutive values (`sequential_score`) and (b) the normalized entropy of
the destination-port distribution (`randomised_score`) — a slow sequential
scan drives `sequential_score` up even when volume is too low to trip a
flow-count threshold, a randomised scan drives entropy/`randomised_score` up
instead. Both are cheap to compute per window with a `Counter` over ports.

### 6.5 Node feature vector (per active IP, per window)

```
degree (distinct peers contacted, in+out),
in_byte_volume, out_byte_volume,
distinct_dst_port_count
```
A rising `distinct_dst_port_count` on one host, held across consecutive
windows, is literally the reconnaissance signature — this is why it's a node
feature rather than folded only into edges.

---

## 7. Module Specifications

### 7.1 `ingestion/flow_csv_loader.py`, `ingestion/ctu13_loader.py`
Load raw CSVs → apply the cleaning rules in 6.1 → return a single tidy
DataFrame with columns `[timestamp, src_ip, dst_ip, src_port, dst_port,
protocol, <flow feature columns from 6.3>, raw_label]`. `ctu13_loader.py`
additionally applies `map_ctu13_label` from 6.2 and drops `Background` rows
before returning (they are re-added separately as an unlabeled pool for
Stage-1 — keep them in a second return value, not silently discarded).
Discover input files with `glob.glob(path, recursive=True)`, never hardcoded
filenames (see Section 2.2 rationale).

### 7.2 `ingestion/pcap_parser.py`
Use `scapy.utils.PcapReader` as a streaming iterator over
`Wednesday-workingHours.pcap` — **do not use `rdpcap`**, which loads the
entire file into memory and will exhaust RAM on a multi-GB capture. For each
packet, extract (if present) the fields listed in 6.4's raw ingredients
(TTL, TCP window size, IP fragment flag, TCP seq for retransmission
detection, payload length, dst port) keyed by `(src_ip, dst_ip, sport, dport,
timestamp)`. Aggregate into the per-(host-pair, window) vector of Section
6.4 as a separate pass after the full file has been streamed once into a
temporary per-packet parquet file (stream-parse once, aggregate with
vectorized pandas groupby afterward — do not build the window aggregation
inside the packet-by-packet Python loop, it will be extremely slow at
multi-million-packet scale).

### 7.3 `features/flow_features.py`, `features/packet_features.py`
Pure functions: given a tidy per-flow (or per-packet) DataFrame and a window
boundary, return the aggregated feature vector of 6.3 / 6.4 for each
`(src_ip, dst_ip)` group present in that window. Implement with
`groupby([...]).agg({...})`, vectorized — no per-row Python loops over
millions of rows.

### 7.4 `features/windowing.py` — core logic, specified precisely

**Windowing algorithm:**
1. Sort all events (flows for CIC/CTU, packets for the PCAP pass) by
   timestamp.
2. Generate window boundaries: `window_starts = arange(t_min, t_max,
   delta_t_seconds * (1 - overlap))`; each window `t` spans
   `[window_starts[t], window_starts[t] + delta_t_seconds)`.
3. For each window `t`: filter events whose timestamp falls in the window
   (an event that spans a window boundary — a flow with nonzero duration —
   is assigned to the window containing its **start** time; this is a
   documented simplification, state it in the architecture doc).
4. Within each window, group by `(src_ip, dst_ip)` → one edge feature row
   per pair present (flow vector 6.3 ⊕ packet vector 6.4, zero-filled with
   `packet_features_available=0` where no PCAP coverage exists ⊕ raw label
   of any flow in that group, majority vote if multiple labels collide,
   mapped through Section 6.2's tables to get the target stage).
5. Group by `ip` (as either src or dst) → one node feature row per IP
   present (Section 6.5).
6. **Graph assembly:** `V_t` = the set of distinct IPs across all edges in
   window `t`, capped at `N_max=128` by keeping the `N_max` highest-degree
   nodes if more are present (rare — most windows will have far fewer than
   128 distinct IPs) and dropping edges touching a dropped node. Build a
   dense adjacency mask `A_t ∈ {0,1}^{N_max × N_max}` (1 where an edge
   exists), pad node/edge feature tensors with zeros up to `N_max`, and keep
   a boolean `node_valid_mask ∈ {0,1}^{N_max}` so the model can ignore
   padding.
7. **Window label:** a window is "infiltration-positive" if **any** edge in
   it carries a non-Benign label; the window's MITRE stage target is the
   **most severe** stage present (severity order = the `stage_order` array
   in the mapping JSON, later = more severe, `Benign` least severe),
   ties broken by majority vote.
8. **Sequencing:** chunk the ordered list of windows into overlapping
   sequences of length `W` (default 20) with stride `W/2`, for use as
   training examples — each sequence is one `(graph_1, ..., graph_W)` input
   to the RSSM, unrolled window-by-window.
9. Persist everything to `data/processed/<dataset_name>/*.parquet`
   (one file per sequence or per fixed-size shard — do not keep the whole
   feature store in memory at once for CIC-IDS2017's multi-million-row
   scale; write incrementally).

`features/build_dataset.py` is the CLI that runs steps 1–9 end to end for a
given `--dataset {cic2017,ctu13,nf_unsw_nb15_v2}` and config, and prints a
short data-quality report (row counts before/after cleaning, % positive
windows, % edges with packet coverage) — use that report as your first
sanity checkpoint (Section 8).

### 7.5 Model (`src/model/`)

All tensors are batched: batch dim `B`, sequence length `W`, node cap
`N_max`, edge feature dim `d_e`, node feature dim `d_n`.

#### 7.5.1 `dense_graph_attention.py`
```
class DenseGraphAttention(nn.Module):
    """
    2 layers, 4 heads, embed_dim=64 (config-driven).
    Input per window: node_feats [B,N,d_n], edge_feats [B,N,N,d_e] (0 where no edge),
                       adjacency_mask [B,N,N] in {0,1}, node_valid_mask [B,N] in {0,1}.
    For each layer:
      q = W_q(node_feats)          # [B,N,heads,dh]
      k = W_k(node_feats)          # [B,N,heads,dh]
      edge_bias = W_e(edge_feats)  # [B,N,N,heads]  -- lets edge features bias attention scores
      scores[b,i,j,h] = (q[b,i,h]·k[b,j,h])/sqrt(dh) + edge_bias[b,i,j,h]
      mask out where adjacency_mask==0 or node_valid_mask==0 with -inf before softmax
      attn = softmax(scores, dim=j)
      v = W_v(node_feats)          # [B,N,heads,dh]
      node_feats = node_feats + concat_heads(attn @ v)   # residual, then a 2-layer MLP + LayerNorm
    Store `attn` from the last layer as self.last_attention for explainability export.
    Final graph embedding e_t = masked mean-pool of node_feats over valid nodes -> [B, embed_dim]
    """
```
This is the direct dense-attention replacement for the original PyG `GATConv`
stack — same attention-over-neighbours idea, zero PyG dependency, exports
trivially if you ever want ONNX later since it's pure matmul/softmax.

#### 7.5.2 `rssm.py`
```
class RSSM(nn.Module):
    """
    h_dim=128 (GRUCell hidden), z_dim=32 (Gaussian latent).
    posterior_net:  MLP([h_t, e_t]) -> (mu_q, logvar_q)      # used when e_t is observed
    prior_net:      MLP([h_t])      -> (mu_p, logvar_p)      # used for imagination
    gru: nn.GRUCell(input_size=z_dim, hidden_size=h_dim)     # driven by z, not raw e_t directly
         call as h_t = gru(z_{t-1}, h_{t-1}) each step; feed e_t only into posterior_net
    reparameterize(mu, logvar): mu + exp(0.5*logvar) * randn_like(mu)

    def observe_step(h_prev, z_prev, e_t):
        h_t = gru(z_prev, h_prev)
        mu_q, logvar_q = posterior_net(h_t, e_t)
        z_t = reparameterize(mu_q, logvar_q)
        mu_p, logvar_p = prior_net(h_t)               # prior computed too, for the KL term
        kl_t = kl_divergence(N(mu_q,logvar_q), N(mu_p,logvar_p))   # analytic Gaussian KL, closed form
        return h_t, z_t, kl_t, (mu_q, logvar_q, mu_p, logvar_p)

    def imagine_step(h_prev, z_prev):                  # no e_t argument at all -- this is the point
        h_t = gru(z_prev, h_prev)
        mu_p, logvar_p = prior_net(h_t)
        z_t = reparameterize(mu_p, logvar_p)
        return h_t, z_t
    """
```
Belief state `s_t = concat(h_t, z_t)` (dim 160) feeds every head.

#### 7.5.3 `heads.py`
Four small MLPs sharing input `s_t` (dim 160):
- `reconstruction_head`: 2-layer MLP → predicts the **pooled edge feature
  vector summary of the window** (mean edge feature vector over valid edges,
  not the full N×N tensor — reconstructing the full padded tensor is wasted
  capacity). Loss: MSE.
- `infiltration_head`: 2-layer MLP → sigmoid scalar. Loss: focal loss
  (γ=2), binary target = "any non-Benign edge in this window" (7.4 step 7).
- `mitre_head`: 2-layer MLP → 7-way softmax over `stage_order` in the
  mapping JSON. Loss: class-weighted cross-entropy, weights = inverse
  class frequency computed once from the Stage-2 training split.
- No separate anomaly head — `kl_t` from `RSSM.observe_step` **is** the
  surprise/anomaly signal, read out directly, never trained against a
  target.

#### 7.5.4 `gt_rssm.py`
Assembles the above. `forward(sequence_of_graphs)` runs `DenseGraphAttention`
per window to get `e_1..e_W`, then unrolls `RSSM.observe_step` across the
sequence (teacher-forced on real `e_t` at every step during training),
collecting `s_t`, `kl_t`, and all four head outputs per step. Also exposes
`imagine(h_t, z_t, K)`:
```python
def imagine(self, h_t, z_t, K):
    trajectory = []
    h, z = h_t, z_t
    for k in range(1, K + 1):
        h, z = self.rssm.imagine_step(h, z)
        s = torch.cat([h, z], dim=-1)
        trajectory.append({
            "infiltration_prob": torch.sigmoid(self.infiltration_head(s)),
            "stage_probs": torch.softmax(self.mitre_head(s), dim=-1),
        })
    return trajectory
```
This is the literal "roll out K steps ahead, no new observations" forecasting
deliverable — run it at inference time for K ∈ {5, 10, 20} (25s/50s/100s
ahead at Δt=5s).

### 7.6 Training

#### 7.6.1 `losses.py` — full objective
```
L_total = L_reconstruction (MSE, mean over sequence)
        + β_t · mean_over_sequence(KL_t)          # β follows the annealing schedule below
        + λ1 · L_infiltration (focal loss, γ=2)   # only active in Stage 2
        + λ2 · L_mitre (class-weighted CE)        # only active in Stage 2
```
`β` schedule: linear ramp from 0 to 1 over the first 10% of Stage-1 training
steps, then held at 1 for the remainder of Stage 1 and all of Stage 2 — this
is the standard KL-annealing fix for posterior collapse, mention it if asked
about training stability. `λ1, λ2` default to 1.0 each in Stage 2; if the
unsupervised reconstruction/KL loss visibly degrades once Stage 2 starts
(watch for it — Section 8's checkpoints), lower both to 0.5 and/or drop the
encoder+RSSM learning rate (see 7.6.3) rather than raising them, since the
whole point of the two-stage recipe is that the anomaly signal must **not**
depend on supervised labels to keep the "generalizes to unseen attacks"
claim honest.

#### 7.6.2 `pretrain_stage1.py`
Train `DenseGraphAttention + RSSM + reconstruction_head` only, on **all**
traffic (CIC-IDS2017 Monday, which is 100% benign, plus every window from
every dataset regardless of label, plus CTU-13's excluded `Background` pool)
using only `L_reconstruction + β·KL`. Optimizer: Adam, lr=1e-3, batch
size=64 sequences, ~15–30 epochs or until reconstruction loss plateaus on a
held-out slice. Checkpoint to `checkpoints/stage1.pt`.

#### 7.6.3 `finetune_stage2.py`
Load `stage1.pt`, add `infiltration_head` + `mitre_head`, train on the
labeled Stage-2 mix (CIC-IDS2017 Tue–Fri + CIC-IDS2017 Wednesday PCAP-joined
windows + CTU-13 non-Background windows) using the full `L_total`.
Two parameter groups: encoder+RSSM at a **lower** learning rate (1e-4) so
Stage-1 dynamics aren't destroyed, heads at 1e-3. ~20–40 epochs, early-stop
on validation infiltration F1. Class imbalance beyond focal loss / class
weights: oversample whole **windows** belonging to rare stages
(Exfiltration-mapped, Command_And_Control) at the sequence level — never
row-level SMOTE on flow rows, it breaks temporal structure and was correctly
flagged as a pitfall in the original proposal. Checkpoint to
`checkpoints/gt_rssm_v1.pt`.

### 7.7 `baseline/logreg_baseline.py`
Logistic Regression (`sklearn.linear_model.LogisticRegression`, class_weight
`balanced`) on the **flattened edge feature vectors** (no graph, no temporal
structure — one row per (host-pair, window), exactly as the brief specifies
for the baseline), predicting the same binary infiltration target. Train
this **first**, before the world model (Section 8) — it de-risks the
"benchmark against a baseline" deliverable early and gives you a sanity
floor to compare against while GT-RSSM is still being debugged.

### 7.8 Explainability

#### 7.8.1 `attention_export.py`
Pull `DenseGraphAttention.last_attention` for a chosen window, average over
heads, and export as an edge-weight overlay (node positions from a simple
spring layout, edge thickness/opacity = attention weight) — used by the
dashboard's network graph view.

#### 7.8.2 `integrated_gradients.py`
Use `captum.attr.IntegratedGradients` with the model's `infiltration_head`
output (composed with the frozen encoder/RSSM up to `s_t`) as the target
function, baseline = all-zeros edge feature vector, attributing back to the
input edge feature vector for a chosen window. Report top-5 features by
absolute attribution.

#### 7.8.3 `shap_surrogate.py`
Train a small `xgboost.XGBClassifier` on (flattened window-level tabular
features → GT-RSSM's `infiltration_head` sigmoid output as a soft
regression-style target, or the hard label — either is defensible, prefer
soft target so the surrogate imitates the actual model rather than the
ground truth). Compute `shap.TreeExplainer` values on this surrogate. This
is the standard, defensible workaround for "SHAP directly on the RSSM is
impractical" — state that explicitly if asked in Q&A.

### 7.9 `evaluation/metrics.py`
Implement, for both GT-RSSM and the LR baseline, on a common held-out test
split:
- Precision / Recall / F1 (macro across the 7 MITRE stages, and separately
  the binary infiltration F1), ROC-AUC, PR-AUC, False Positive Rate.
- **Lead time to detection**: for each true attack window sequence, find the
  first window where the ground-truth label turns non-Benign (`t_compromise`)
  and the first window where the model's infiltration probability crosses a
  fixed threshold (tuned on validation data to a target FPR, e.g. 5%)
  (`t_alert`). Lead time = `(t_compromise - t_alert) * delta_t_seconds`
  (can be negative for a baseline that only fires after the fact — report
  mean and median, and explicitly report the LR baseline's lead time too,
  which should be at or near zero by construction).
- **Brier score** on the K-step-ahead forecast: for each `k` in {5,10,20},
  compare `imagine(...)`'s `infiltration_prob` at horizon `k` against the
  actual realized label `k` windows later; report per-horizon Brier score.
- **Cross-dataset generalization**: train on CIC-IDS2017 only, evaluate
  zero-shot on CTU-13 (and NF-UNSW-NB15-v2 if present) without
  re-fine-tuning; report the F1 drop, and do the same for the LR baseline
  for comparison.
- **Held-out-category surprise AUC**: retrain Stage 2 (Stage 1 is already
  label-free, so no change needed there) with one entire mapped stage
  excluded from the supervised loss (recommended: exclude
  `Command_And_Control`, since CTU-13 provides enough of it to have a
  meaningful held-out test); at evaluation time, compute `kl_t` (the
  surprise signal) on held-out `Command_And_Control` windows vs. benign
  validation windows, and report the ROC-AUC of using `kl_t` alone to
  separate the two groups. This is the single number that answers "what
  about attacks you've never seen" — make sure it appears in the
  architecture doc, the slide deck, and the demo video narration.

`run_eval.py` runs all of the above and writes one comparison table
(GT-RSSM vs. LR baseline, every metric above) to
`data/processed/eval_results.md`.

### 7.10 `inference/predict.py`
Single function `predict(input_path: str, k_steps=(5,10,20)) -> dict` that:
loads `checkpoints/gt_rssm_v1.pt` once (cache the loaded model at module
level so repeated calls in the Streamlit session don't reload weights),
accepts a PCAP or CSV path, runs it through the same feature pipeline as
Section 7.4 (reusing `build_dataset`'s functions directly, not a
reimplementation), runs the model forward pass to get per-window belief
states, runs `imagine()` from the last observed belief state, and returns a
single dict: `{"windows": [...], "infiltration_timeline": [...],
"stage_timeline": [...], "forecast": {...imagine() output...},
"kl_surprise": [...], "attention_by_window": {...}, "top_features_by_window":
{...}}`. **Both the Streamlit dashboard and the optional FastAPI wrapper call
this exact same function** — there is exactly one inference code path in the
whole repository, so results can never disagree between the two UIs.

### 7.11 `dashboard/streamlit_app.py`
1. File-upload widget (PCAP or CSV).
2. On upload, call `inference.predict.predict(...)` directly (in-process,
   no HTTP) with a progress spinner.
3. Infiltration probability timeline: line chart, observed history solid,
   the `imagine()` forecast extending past the last real window as a dashed
   line, red horizontal band above the alert threshold.
4. Attack-stage ribbon: colour-coded strip beneath the timeline, one colour
   per stage in `stage_order`, including the forecast region.
5. Network graph view for the currently-selected window: nodes = IPs, edges
   thickened/coloured by exported attention weight (7.8.1), suspicious
   high-attention hosts highlighted.
6. Explainability panel: SHAP bar chart (7.8.3, top-5 features) for the
   selected window, plus one templated natural-language sentence built from
   the top feature name and its direction, e.g. "High SYN-flag ratio and a
   rising distinct-port count on `<ip>` are driving this
   `<predicted_stage>` prediction."
7. Anomaly badge: shows "possible unseen technique" whenever `kl_surprise`
   for the selected window exceeds the top-1%-of-benign-validation threshold
   computed once at training time and stored alongside the checkpoint.
8. Export button: render the above into a one-page Markdown/PDF summary
   (reuse the `pdf` skill available in this environment if/when you build
   this specific piece, rather than hand-rolling PDF generation).

### 7.12 `api/main.py` (optional, not on the critical demo path)
A single FastAPI endpoint `POST /predict` accepting a file upload, calling
the same `inference.predict.predict(...)`, returning its dict as JSON. Build
this last, only after the Streamlit path works end to end — it exists to
tick the "or a Flask/API-style interface" box in the brief, not to be
load-bearing for the live demo.

---

## 8. Build Order — follow this sequence, verifying each checkpoint before moving on

1. **Environment**: install requirements, `python -c "import torch, xgboost,
   shap, captum, scapy, streamlit"` succeeds with no errors.
2. **Data presence check**: write a tiny script that asserts the files from
   Section 2 exist at the expected paths and prints row/packet counts. Do
   not proceed until this passes.
3. **Ingestion + cleaning** (7.1): load CIC-IDS2017 CSVs and CTU-13
   binetflow files, apply every rule in 6.1, print before/after row counts
   and label distributions. *Checkpoint*: label distribution roughly matches
   the known CIC-IDS2017 counts (~2.27M Benign, 14 attack types, heavily
   imbalanced — if your counts are wildly different, a cleaning rule was
   applied wrong).
4. **PCAP streaming pass** (7.2): stream `Wednesday-workingHours.pcap` to a
   temporary packet-level parquet file. *Checkpoint*: total packet count is
   plausible for a single working day of a testbed with ~10–25 hosts (tens
   of millions, not billions; if scapy chokes on RAM here, confirm you used
   `PcapReader`, not `rdpcap`).
5. **Windowing + graph construction** (7.4): run `build_dataset.py` for
   `cic2017`, then `ctu13`. *Checkpoint*: read back one parquet shard,
   confirm shapes (`node_feats`, `edge_feats`, `adjacency_mask`,
   `node_valid_mask` all have the `N_max` dimension you configured), confirm
   `packet_features_available` is 1 only for Wednesday CIC-IDS2017 windows
   and 0 everywhere else, confirm window-level labels are non-degenerate
   (not 100% one class).
6. **Baseline** (7.7): train and evaluate the Logistic Regression baseline
   end to end, write its numbers into `eval_results.md`. Do this now, not
   later — it is your fallback demonstrable result if the world model needs
   more debugging time.
7. **Stage-1 pretraining** (7.6.2): train on benign-heavy + unlabeled data.
   *Checkpoint*: reconstruction loss decreases and KL term does not collapse
   to exactly 0 (posterior collapse) nor blow up unboundedly — plot both
   curves, this is the single most important training-stability check for
   this whole project.
8. **Stage-2 fine-tuning** (7.6.3). *Checkpoint*: infiltration validation F1
   clearly exceeds the LR baseline's F1 from step 6 — if it does not, first
   suspect the class-weighting/oversampling before touching model
   architecture.
9. **K-step imagination** (7.5.4 `imagine()`): sanity-run it on a known
   attack sequence and visually confirm the forecast probability rises
   *before* the labeled compromise window in at least some cases — this is
   the "lead time" property, check it qualitatively before trusting the
   quantitative metric in step 11.
10. **Explainability** (7.8): wire all three methods, confirm they produce
    non-degenerate output on a real example (attention weights not uniform,
    SHAP values not all zero).
11. **Full evaluation** (7.9): run `run_eval.py`, producing the final
    comparison table including lead time, Brier score, cross-dataset F1,
    and the held-out-category surprise AUC.
12. **Streamlit demo** (7.11): run end to end on a fresh terminal
    (`streamlit run dashboard/streamlit_app.py`), upload a fresh CSV/PCAP
    slice not used in training, confirm no crashes and no network calls out.
13. **FastAPI wrapper** (7.12) — only after 12 is solid.
14. **Docs & deliverables** (Section 10): condense architecture doc to 2
    pages, record the demo video, build the 5-slide deck, finalize README.

---

## 9. `configs/default.yaml` (starting defaults — tune from here)

```yaml
data:
  delta_t_seconds: 5
  window_overlap: 0.5
  sequence_length_W: 20
  n_max_nodes: 128
model:
  gat_embed_dim: 64
  gat_heads: 4
  gat_layers: 2
  rssm_h_dim: 128
  rssm_z_dim: 32
training:
  stage1_lr: 1.0e-3
  stage1_epochs: 25
  stage1_batch_size: 64
  stage2_encoder_lr: 1.0e-4
  stage2_head_lr: 1.0e-3
  stage2_epochs: 30
  focal_loss_gamma: 2.0
  kl_anneal_fraction: 0.10
  lambda_infiltration: 1.0
  lambda_mitre: 1.0
paths:
  cic2017_csv_dir: data/raw/cic2017/csv
  cic2017_pcap: data/raw/cic2017/pcap/Wednesday-workingHours.pcap
  ctu13_dir: data/raw/ctu13
  nf_unsw_nb15_v2_csv: data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv
  processed_dir: data/processed
  checkpoint_dir: checkpoints
```

---

## 10. Deliverables Mapping (explicit checklist against the brief)

| Required deliverable | Where it comes from |
|---|---|
| Source Code Link | This repository |
| Readme with Setup Instructions | `README.md` — `pip install -r requirements.txt`, then the exact command sequence from Section 8 steps 3–13, ending in `streamlit run dashboard/streamlit_app.py` |
| Architecture Document (max 2 pages) | Condense Sections 0, 1, and the deliverables/feasibility tables into two pages, keeping the Section 1 diagram |
| Demo Video (max 2 minutes) | 0:00–0:15 problem framing ("forecast before compromete completes"); 0:15–0:45 upload a CSV/PCAP, show the pipeline running fully offline; 0:45–1:30 walk the infiltration timeline (note the dashed forecast region), the MITRE stage ribbon, the attention/SHAP explainability panel; 1:30–2:00 show the anomaly/surprise badge and the baseline-comparison numbers from `eval_results.md` |
| Technical Presentation (5 slides) | 1: problem + world-model framing; 2: GT-RSSM architecture diagram (Section 1); 3: explainability + MITRE mapping (note the honest Impact/DoS addition); 4: evaluation table incl. lead time and held-out-category surprise AUC; 5: feasibility/limitations (Section 0's substitution table doubles as this) + future extensions |

`README.md` should also state plainly, in one paragraph, the three
substitutions from Section 0 (dense attention instead of PyG, Gaussian
latents instead of categorical, in-process serving instead of ONNX) framed
as deliberate engineering decisions for reliability — this reads well to
technical judges precisely because it is true and because naming your own
constraints is a credibility signal, exactly as the original proposal's
feasibility section already argued.

---

## 11. Troubleshooting / Known Pitfalls (read before debugging blindly)

- **NaN loss appearing after a few Stage-1 steps** → almost always leftover
  `inf`/`NaN` from `Flow Bytes/s`/`Flow Packets/s` (see 6.1) that slipped
  past cleaning, or a `logvar` exploding because the KL term isn't yet
  annealed in (check the β schedule is actually being applied, not just
  defined).
- **Posterior collapse** (`KL → 0` early and the reconstruction loss stops
  improving) → increase `kl_anneal_fraction`, or reduce `rssm_z_dim`
  learning-rate implicitly by capping `logvar` outputs to a sane range
  (e.g. clamp to `[-10, 10]` before `exp`) — an unclamped `logvar` head is a
  common silent cause of this.
- **Training extremely slow on CIC-IDS2017 windowing** → confirm feature
  aggregation is vectorized `groupby` calls (7.3), not per-row Python loops;
  a per-row loop over CIC-IDS2017's ~2.8M rows will make this step the
  dominant cost of the entire project by orders of magnitude.
- **Scapy PCAP parsing OOMs or hangs** → confirm `PcapReader` streaming is
  used, not `rdpcap`; if still slow, consider `dpkt` as a faster drop-in for
  the raw packet iteration, keeping the same aggregation logic.
- **Streamlit demo works locally but "hangs" when judges try it** → confirm
  nothing in the demo path makes an outbound network call (no
  `pip install` at runtime, no external font/CDN fetch, no accidental
  telemetry) — the brief explicitly requires the interface to run fully
  offline.
- **GT-RSSM underperforms the LR baseline** → before touching the
  architecture, check (in order): class weighting in `L_mitre` is actually
  active, `lambda_infiltration`/`lambda_mitre` aren't so high they're
  fighting the KL term (watch the individual loss components separately,
  not just `L_total`), and the train/val split doesn't leak windows from the
  same attack episode across the split (split by contiguous time ranges,
  never by random row shuffling, or you will get leakage-inflated numbers
  that collapse under an honest cross-dataset test).

---

## 12. Timeline (adjust to your actual available days)

| Phase | Deliverable |
|---|---|
| Data & feature pipeline | Windowed feature store from CIC-IDS2017 + CTU-13; data-quality report passes checkpoint 5 |
| Baseline | LR baseline trained + logged (do this early, Section 7.7's rationale) |
| Stage-1 pretraining | Reconstruction + KL converging, no posterior collapse |
| Stage-2 fine-tuning | Infiltration/MITRE heads beat the LR baseline; K-step imagination runs end to end |
| Explainability | All three methods wired into `inference.predict` |
| Full evaluation | Lead-time, Brier, cross-dataset F1, held-out-category surprise AUC all computed and tabulated |
| Streamlit demo | Fresh-machine end-to-end run with zero network calls |
| Docs, video, deck | README, 2-page architecture doc, demo video, 5-slide deck |

---

## 13. Future Extensions (mention briefly in the deck, do not build now)

- ONNX export of the (now dense-op-only, export-friendly) model for
  browser/edge deployment.
- Federated / privacy-preserving training across multiple CII operators.
- Active-learning loop queuing high-`kl_surprise` windows for analyst
  labeling.
- Live Zeek/Suricata streaming ingestion instead of file upload.
- Adding CSE-CIC-IDS2018, CICIoT2023, and the LANL Authentication Dataset
  for broader cross-dataset and lateral-movement-via-auth-logs coverage.
