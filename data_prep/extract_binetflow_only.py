"""
Extract ONLY the .binetflow (and README) files from CTU-13-Dataset.tar.bz2 --
every .pcap and .exe entry is skipped entirely, never written to disk. This
matters a lot here: scenario 10 alone has a 66GB pcap, and the pipeline
(clean_ctu13.py) only ever reads .binetflow files anyway.

Does a single sequential decompression pass over the archive (required for
.tar.bz2 -- it can't be randomly seeked), extracting matches as it goes and
skipping everything else without ever loading pcap/exe content into memory.

Safe to re-run: already-extracted files (same size) are skipped, so you can
run this once now covering ALL 13 scenarios (already-extracted ones and the
remaining ones alike) rather than tracking which ones still need it.

Usage:
    python extract_binetflow_only.py --archive data/raw/ctu13/CTU-13-Dataset.tar.bz2 \
        --output_dir data/raw/ctu13/CTU-13-Dataset
"""
import argparse
import os
import tarfile
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="Path to CTU-13-Dataset.tar.bz2")
    ap.add_argument("--output_dir", required=True,
                     help="Where to extract to -- use the SAME folder your earlier "
                          "tar extraction already wrote into, so paths line up")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    extracted = 0
    skipped_existing = 0
    n_inspected = 0
    start_time = time.time()
    last_heartbeat = start_time

    with tarfile.open(args.archive, mode="r:bz2") as tar:
        # Iterating the TarFile object directly (not calling getmembers() first)
        # is what keeps this a single forward pass over the decompressed stream --
        # calling getmembers() first would force a full scan, then re-seeking
        # backward through a bz2 stream is expensive. This way each member's
        # data is read exactly once, either extracted or skipped.
        for member in tar:
            n_inspected += 1
            now = time.time()
            if now - last_heartbeat > 5:
                print(f"  ...still scanning ({n_inspected} archive entries checked so far, "
                      f"{now - start_time:.0f}s elapsed) -- currently passing: {member.name} "
                      f"({member.size / 1e6:.1f} MB, this one is skipped if it's not .binetflow)")
                last_heartbeat = now

            if not member.isfile():
                continue
            name_lower = member.name.lower()
            base_lower = os.path.basename(member.name).lower()
            wanted = name_lower.endswith(".binetflow") or base_lower.startswith("readme")
            if not wanted:
                continue

            target_path = os.path.join(args.output_dir, member.name)
            if os.path.exists(target_path) and os.path.getsize(target_path) == member.size:
                skipped_existing += 1
                continue

            tar.extract(member, path=args.output_dir)
            extracted += 1
            print(f"  extracted: {member.name}  ({member.size / 1e6:.1f} MB)")

    print(f"\nDone. Extracted {extracted} new file(s), "
          f"skipped {skipped_existing} already-present file(s).")
    print("Every .pcap and .exe in the archive was skipped entirely -- never written to disk.")


if __name__ == "__main__":
    main()