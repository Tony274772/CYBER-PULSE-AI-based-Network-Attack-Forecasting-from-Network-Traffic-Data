"""
One-time cleanup for the CTU-13 scenario folders you already fully extracted
with plain `tar -xjf` (before switching to extract_binetflow_only.py) -- walks
the folder tree and deletes every file EXCEPT .binetflow and README, freeing
the disk space wasted on pcaps/exes the pipeline never reads.

Run with --dry_run first to see exactly what would be deleted before it
actually deletes anything.

Usage:
    python cleanup_extracted_extras.py --root data/raw/ctu13/CTU-13-Dataset --dry_run
    python cleanup_extracted_extras.py --root data/raw/ctu13/CTU-13-Dataset
"""
import argparse
import os

KEEP_SUFFIXES = (".binetflow",)
KEEP_BASENAME_PREFIX = "readme"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. data/raw/ctu13/CTU-13-Dataset")
    ap.add_argument("--dry_run", action="store_true",
                     help="List what would be deleted without deleting anything -- run this first")
    args = ap.parse_args()

    freed_bytes = 0
    removed = 0

    for dirpath, _, filenames in os.walk(args.root):
        for fname in filenames:
            lower = fname.lower()
            keep = lower.endswith(KEEP_SUFFIXES) or lower.startswith(KEEP_BASENAME_PREFIX)
            if keep:
                continue
            fpath = os.path.join(dirpath, fname)
            size = os.path.getsize(fpath)
            freed_bytes += size
            removed += 1
            if args.dry_run:
                print(f"  would delete: {fpath}  ({size / 1e6:.1f} MB)")
            else:
                os.remove(fpath)
                print(f"  deleted: {fpath}  ({size / 1e6:.1f} MB)")

    action = "Would free" if args.dry_run else "Freed"
    print(f"\n{action} {freed_bytes / 1e9:.2f} GB across {removed} file(s).")


if __name__ == "__main__":
    main()
