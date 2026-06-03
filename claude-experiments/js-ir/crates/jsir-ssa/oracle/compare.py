#!/usr/bin/env python3
"""Fast local gate proxy: bucket fixtures by comparing our structure (cc_batch
output on stdin or arg) against the stable React reference (/tmp/react_ref.txt),
reproducing the corpus harness buckets WITHOUT re-running the Node oracle.

Usage:
  target/release/examples/cc_batch <fixtures> > /tmp/ours.txt
  python3 compare.py /tmp/react_ref.txt /tmp/ours.txt [prev_ours.txt]

If prev_ours.txt is given, also prints fixtures whose bucket changed (regressions
first): the agree->mismatch / agree->react_only flips that the slow gate catches.
"""
import sys

def load(path):
    d = {}
    for line in open(path):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        name, s = line.split("\t", 1)
        d[name] = s
    return d

def memo(s):  # does this side memoize? returns (yes, struct-or-None)
    if s in ("none", "fail", "err", ""):
        return (False, None)
    return (True, s)

def bucket(react, ours):
    rm, rs = memo(react)
    om, os = memo(ours)
    if rm and om:
        return "agree" if rs == os else "mismatch"
    if rm and not om:
        return "react_only"
    if om and not rm:
        return "ours_only"
    return "neither"

def main():
    react = load(sys.argv[1])
    ours = load(sys.argv[2])
    counts = {}
    buckets = {}
    for name, rs in react.items():
        os_ = ours.get(name, "none")
        b = bucket(rs, os_)
        counts[b] = counts.get(b, 0) + 1
        buckets[name] = b
    universe = counts.get("agree", 0) + counts.get("mismatch", 0) + counts.get("react_only", 0)
    print(f"total={len(react)} universe={universe} "
          f"agree={counts.get('agree',0)} mismatch={counts.get('mismatch',0)} "
          f"react_only={counts.get('react_only',0)} ours_only={counts.get('ours_only',0)} "
          f"neither={counts.get('neither',0)} "
          f"agree%={100*counts.get('agree',0)/universe:.2f}")
    if len(sys.argv) > 3:
        prev = load(sys.argv[3])
        prev_buckets = {n: bucket(react.get(n, "none"), s) for n, s in prev.items()}
        regr, gain = [], []
        for name, b in buckets.items():
            pb = prev_buckets.get(name)
            if pb is None or pb == b:
                continue
            if pb == "agree" and b != "agree":
                regr.append(f"  REGRESS {name}: agree -> {b} (ours {prev.get(name)} -> {ours.get(name)})")
            elif b == "agree" and pb != "agree":
                gain.append(f"  GAIN    {name}: {pb} -> agree")
            else:
                gain.append(f"  move    {name}: {pb} -> {b}")
        for l in regr: print(l)
        for l in gain: print(l)
        if not regr and not gain:
            print("  (no bucket changes)")

if __name__ == "__main__":
    main()
