"""
Run this FIRST, before doing anything else. Reports the current state of
every raw/processed data file the pipeline expects, so you (agent or human)
know exactly what's already done and what's still missing -- instead of
blindly re-running every step from scratch.

Usage:
    python check_status.py --data_root .

Exit code 0 = every mandatory item is present and looks valid.
Exit code 1 = at least one mandatory item is missing or looks wrong --
              see the [MISSING] / [INCOMPLETE] / [SUSPICIOUSLY SMALL] lines.
"""
import argparse
import glob
import os
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def parquet_row_count(path):
    if pq is None:
        return None
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def check_file(path, label, min_size_mb=None):
    if not os.path.isfile(path):
        print(f"  [MISSING] {label}: {path}")
        return False
    size_mb = os.path.getsize(path) / 1e6
    note = f"{size_mb:.1f} MB"
    if path.endswith(".parquet"):
        n = parquet_row_count(path)
        if n is not None:
            note += f", {n:,} rows"
    if min_size_mb is not None and size_mb < min_size_mb:
        print(f"  [SUSPICIOUSLY SMALL] {label}: {path} ({note}, expected >= {min_size_mb}MB -- "
              f"likely a truncated download, re-fetch it)")
        return False
    print(f"  [OK] {label}: {path} ({note})")
    return True


def check_glob_count(pattern_dir, suffix, expected_count, label):
    files = glob.glob(os.path.join(pattern_dir, "**", f"*{suffix}"), recursive=True)
    ok = len(files) >= expected_count
    status = "OK" if ok else "INCOMPLETE"
    print(f"  [{status}] {label}: found {len(files)} (expected >= {expected_count}) under {pattern_dir}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".", help="Repo root containing data/")
    args = ap.parse_args()
    root = args.data_root
    all_ok = True

    print("\n=== CIC-IDS2017 ===")
    print(" -- raw --")
    all_ok &= check_glob_count(os.path.join(root, "data/raw/cic2017/csv"), ".csv", 8, "flow CSVs")
    all_ok &= check_file(os.path.join(root, "data/raw/cic2017/pcap/Wednesday-workingHours.pcap"),
                          "Wednesday PCAP", min_size_mb=10000)
    print(" -- processed --")
    all_ok &= check_file(os.path.join(root, "data/processed/cic2017/cic2017_clean.parquet"), "cleaned flow parquet")
    all_ok &= check_file(os.path.join(root, "data/processed/cic2017/cic2017_wednesday_packet_features.parquet"),
                          "packet-level features")
    for split in ("train", "val", "test"):
        all_ok &= check_file(os.path.join(root, f"data/processed/cic2017/cic2017_{split}.parquet"), f"{split} split")

    print("\n=== CTU-13 ===")
    print(" -- raw --")
    all_ok &= check_glob_count(os.path.join(root, "data/raw/ctu13"), ".binetflow", 13, "binetflow files")
    print(" -- processed --")
    all_ok &= check_file(os.path.join(root, "data/processed/ctu13/ctu13_labeled.parquet"), "labeled parquet")
    all_ok &= check_file(os.path.join(root, "data/processed/ctu13/ctu13_background_pool.parquet"), "background pool")
    for split in ("train", "val", "test"):
        all_ok &= check_file(os.path.join(root, f"data/processed/ctu13/ctu13_{split}.parquet"), f"{split} split")

    print("\n=== NF-UNSW-NB15-v2 (optional) ===")
    optional_csv = os.path.join(root, "data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv")
    if os.path.isfile(optional_csv):
        print(" -- raw --")
        check_file(optional_csv, "raw CSV")
        print(" -- processed --")
        check_file(os.path.join(root, "data/processed/nf_unsw_nb15_v2/nf_unsw_nb15_v2_clean.parquet"), "cleaned parquet")
        for split in ("train", "val", "test"):
            check_file(os.path.join(root, f"data/processed/nf_unsw_nb15_v2/nf_unsw_nb15_v2_{split}.parquet"),
                       f"{split} split")
    else:
        print("  (skipped entirely -- not downloaded; this is fine, this dataset is optional)")

    print()
    if all_ok:
        print("=== ALL MANDATORY DATA READY -- proceed to GT-RSSM_Build_Instructions.md ===")
    else:
        print("=== SOME MANDATORY STEPS STILL MISSING -- see [MISSING]/[INCOMPLETE] lines above; "
              "run the matching command from DATA_PREP_AGENT_CHECKLIST.md, then re-run this script ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
