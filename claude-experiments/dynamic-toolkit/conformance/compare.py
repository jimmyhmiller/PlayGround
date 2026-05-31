#!/usr/bin/env python3
"""Join /tmp/ours.tsv against conformance/oracle.tsv and report matches.

ours.tsv  lines: <index>\tOK|ERR|ABORT\t<pr-str>
oracle.tsv lines: <index>\t<pr-str>
"""
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(ROOT, "conformance", "oracle.tsv")
OURS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ours.tsv"
CORPUS = os.path.join(ROOT, "conformance", "corpus.edn")


def load_oracle(p):
    d = {}
    for line in open(p):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            d[int(parts[0])] = parts[1]
    return d


def load_ours(p):
    d = {}
    for line in open(p):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[0].isdigit():
            d[int(parts[0])] = (parts[1], parts[2])
    return d


corpus = [l.rstrip("\n") for l in open(CORPUS)]
o = load_oracle(ORACLE)
u = load_ours(OURS)
match = 0
fails = []
for i in sorted(o):
    ov = o[i]
    status, uv = u.get(i, ("MISSING", ""))
    if uv == ov:
        match += 1
    else:
        expr = corpus[i] if i < len(corpus) else "?"
        fails.append((i, expr, ov, status, uv))
print(f"match: {match}/{len(o)}")
print("--- fails ---")
for i, expr, ov, st, uv in fails:
    print(f"{i}: {expr}\n     oracle={ov!r}  ours={st}:{uv!r}")
