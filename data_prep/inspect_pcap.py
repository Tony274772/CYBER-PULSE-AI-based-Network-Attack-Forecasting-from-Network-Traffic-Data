"""
Lightweight sanity check on the Wednesday-workingHours.pcap file — confirms it's
readable and reports basic stats, WITHOUT doing the full packet-level feature
extraction (that step belongs to the modeling pipeline the build agent implements,
per Section 7.2 of GT-RSSM_Build_Instructions.md).

Streams the file with scapy's PcapReader (not rdpcap) so it never loads the
whole multi-GB capture into RAM.

Usage:
    python inspect_pcap.py --pcap data/raw/cic2017/pcap/Wednesday-workingHours.pcap
"""
import argparse
from collections import Counter

from scapy.layers.inet import IP
from scapy.utils import PcapReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--max_packets", type=int, default=None,
                     help="Optional cap, for a quick check on a huge file")
    args = ap.parse_args()

    n_packets = 0
    ip_counter = Counter()
    first_ts, last_ts = None, None

    with PcapReader(args.pcap) as reader:
        for pkt in reader:
            n_packets += 1
            if args.max_packets and n_packets > args.max_packets:
                break
            ts = float(pkt.time)
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
            if IP in pkt:
                ip_counter[pkt[IP].src] += 1
            if n_packets % 1_000_000 == 0:
                print(f"  ...{n_packets:,} packets streamed so far")

    print("\n=== PCAP sanity report ===")
    print(f"File            : {args.pcap}")
    print(f"Total packets   : {n_packets:,}")
    if first_ts is not None:
        print(f"Time span (s)   : {last_ts - first_ts:,.1f}  ({(last_ts - first_ts) / 3600:.2f} hours)")
    print(f"Distinct src IPs: {len(ip_counter)}")
    print("Top 10 talkers  :")
    for ip, count in ip_counter.most_common(10):
        print(f"  {ip:>15}  {count:,} packets")


if __name__ == "__main__":
    main()
