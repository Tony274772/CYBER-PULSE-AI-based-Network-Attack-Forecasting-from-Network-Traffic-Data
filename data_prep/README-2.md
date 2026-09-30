# Data Preparation Toolkit

Run this before handing the repo to the build agent. It downloads, cleans,
and splits the raw datasets. It stops **before** windowing/graph
construction — that step needs the model config and belongs to the agent's
build (Section 7.4 of `GT-RSSM_Build_Instructions.md`).

## 1. Datasets needed

| # | Dataset | Required? | Why |
|---|---|---|---|
| 1 | CIC-IDS2017 | Yes (primary) | Flow-level labels for 14 attack types (CSV) **and** packet-level features (TTL, TCP window, retransmission, port-scan signature — from one day's raw PCAP) |
| 2 | CTU-13 | Yes (secondary) | Botnet C2 traffic + cross-dataset generalization test |
| 3 | NF-UNSW-NB15-v2 | Optional | Only used for the extra cross-dataset surprise-AUC ablation — skip under time pressure |

## 2. Download

### CIC-IDS2017
1. Go to https://www.unb.ca/cic/datasets/ids-2017.html and fill the short
   access-request form (usually approved fast).
2. Download `GeneratedLabelledFlows.zip`, unzip it — 8 CSVs come out. Put
   them flat at `data/raw/cic2017/csv/`.
3. Download **only** `Wednesday-workingHours.pcap` (don't fetch all 5 days —
   that's 50+GB and one day is enough). Put it at
   `data/raw/cic2017/pcap/Wednesday-workingHours.pcap`.
   (If the form is slow to respond, Kaggle mirrors named "CIC-IDS2017" from
   users `cicdataset` / `dhoogla` / `biprobarai` carry the same files —
   check the file list matches the official one before trusting a mirror.)

### CTU-13
No registration needed:
```bash
bash download_ctu13.sh data/raw/ctu13
```

### NF-UNSW-NB15-v2 (optional)
Go to https://staff.itee.uq.edu.au/marius/NIDS_datasets/ → "NetFlow v2
Datasets" → `NF-UNSW-NB15-v2` → download the CSV to
`data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv`.

## 3. Preprocess

```bash
pip install -r requirements.txt

python clean_cic2017.py --input_dir data/raw/cic2017/csv \
    --output_dir data/processed/cic2017

python inspect_pcap.py --pcap data/raw/cic2017/pcap/Wednesday-workingHours.pcap

python extract_packet_features.py --pcap data/raw/cic2017/pcap/Wednesday-workingHours.pcap \
    --output_dir data/processed/cic2017 --delta_t_seconds 5

python clean_ctu13.py --input_dir data/raw/ctu13 \
    --output_dir data/processed/ctu13

# optional
python clean_nf_unsw_nb15_v2.py --input_csv data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv \
    --output_dir data/processed/nf_unsw_nb15_v2
```

Each script prints a data-quality report — read it before moving on:
- `clean_cic2017.py`: row counts before/after cleaning, label distribution,
  date-range sanity check.
- `inspect_pcap.py`: packet count, time span, distinct source IPs — confirms
  the PCAP is readable and roughly matches the working-hours capture you
  expect (don't skip this; a silently truncated download is a common failure
  mode and this catches it before you're debugging the model instead).
- `extract_packet_features.py`: **this is the actual packet-level feature
  extraction** — TTL variance, TCP window size stats, IP fragmentation
  ratio, payload size stats, retransmission counts, and port-scan
  sequential/randomised scores, aggregated per `(src_ip, dst_ip,
  window_id)` at the same `delta_t_seconds` the build agent's config uses
  (default 5s). It streams the PCAP once (never loads it whole into RAM)
  and can take a while on a multi-GB file — that's expected. Output is
  `cic2017_wednesday_packet_features.parquet`, which is a **separate
  table from the flow-level parquet**: the two are meant to be joined by
  the build agent on `(src_ip, dst_ip, window_id)` once flow-level
  windows are constructed, not merged by this toolkit (the flow-level
  windowing itself is still Section 7.4's job).
- `clean_ctu13.py`: if it reports a meaningful number of `UNMATCHED` labels,
  open the script, look at the sample raw labels it prints, and extend
  `CTU13_RULES` before proceeding — don't let `UNMATCHED` rows silently drop.

## 4. Split (train/val/test — time-aware, no leakage)

```bash
python split_data.py --dataset cic2017 \
    --input data/processed/cic2017/cic2017_clean.parquet \
    --output_dir data/processed/cic2017

python split_data.py --dataset ctu13 \
    --input data/processed/ctu13/ctu13_labeled.parquet \
    --output_dir data/processed/ctu13
```
- **CIC-IDS2017** splits *within each day-file* by time order (first 70% →
  train, next 15% → val, last 15% → test) — every split still sees every
  attack type, but no minute of traffic is shared across splits.
- **CTU-13** splits by *whole scenario* — each of the 13 botnet runs is
  assigned entirely to one split, never divided across splits.
- `ctu13_background_pool.parquet` (unlabeled `Background` traffic) is **not**
  split — hand it to the agent as-is; it's Stage-1-only, and train/val/test
  doesn't apply to unsupervised pretraining data.
- `Monday-WorkingHours.pcap_ISCX.csv` in CIC-IDS2017 is 100% benign traffic.
  It's fine for it to land in all three splits (there's no attack label to
  leak), and it also doubles as extra Stage-1 pretraining material alongside
  the `train` split.

## 5. What you hand off to the build agent

```
data/raw/...                                          (as downloaded above)
data/processed/cic2017/cic2017_{train,val,test}.parquet          (flow-level)
data/processed/cic2017/cic2017_wednesday_packet_features.parquet (packet-level)
data/processed/ctu13/ctu13_{train,val,test}.parquet
data/processed/ctu13/ctu13_background_pool.parquet
data/processed/nf_unsw_nb15_v2/...                    (optional)
GT-RSSM_Build_Instructions.md
```
The agent's job starts at windowing/graph construction (Section 7.4 of the
build spec) — everything upstream of that is already done.
