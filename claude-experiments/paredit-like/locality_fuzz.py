#!/usr/bin/env python3
"""Single-paren-drop locality fuzz for `paredit-like balance`.

For each line ending in a closing delimiter, drop one closer, balance the
perturbed file, and compare the output to the ORIGINAL file. Classify:
  exact     - output == original (perfect repair)
  local     - well-formed, idempotent, differences confined to the top-level
              form holding the perturbation
  nonlocal  - a changed line outside that form
  unbalanced- output still not well-formed
  nonidem   - balance(balance(x)) != balance(x)
Every classification is logged with its diff; nothing is capped.
"""
import subprocess, sys, os, tempfile
from concurrent.futures import ThreadPoolExecutor

BIN, SRC, LOG = sys.argv[1], sys.argv[2], sys.argv[3]
WORKERS = int(os.environ.get("FUZZ_WORKERS", "12"))

PAIRS = {")": "(", "]": "[", "}": "{"}


def well_formed(text):
    stack, in_str, esc, in_com = [], False, False, False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == "\\" and not in_com:
            esc = True
            continue
        if ch == '"' and not in_com:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == ";":
            in_com = True
        if ch == "\n":
            in_com = False
        if in_com:
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in PAIRS:
            if not stack or stack.pop() != PAIRS[ch]:
                return False
    return not in_str and not stack


orig = open(SRC).read()
lines = orig.split("\n")
starts = [i for i, l in enumerate(lines) if l[:1] in "([{"]


def form_range(idx):
    lo = 0
    for s in starts:
        if s <= idx:
            lo = s
        else:
            return (lo, s - 1)
    return (lo, len(lines) - 1)


def run_balance(text):
    with tempfile.NamedTemporaryFile("w", suffix=".coil", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        r = subprocess.run([BIN, "balance", path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    return r


def check(i):
    line = lines[i]
    r = line.rstrip()
    perturbed = lines[:]
    perturbed[i] = r[:-1] + line[len(r):]
    out = run_balance("\n".join(perturbed))
    if out.returncode != 0:
        return ("failed", f"line {i+1}: rc={out.returncode}\n{out.stderr}\n")
    got = out.stdout
    if not well_formed(got):
        return ("unbalanced", f"line {i+1}: output not well-formed\n")
    again = run_balance(got).stdout
    if again.rstrip("\n") != got.rstrip("\n"):
        return ("nonidem", f"line {i+1}: balance not idempotent\n")
    if got.rstrip("\n") == orig.rstrip("\n"):
        return ("exact", "")
    got_lines = got.split("\n")
    lo, hi = form_range(i)
    bad = []
    for j in range(max(len(got_lines), len(lines))):
        a = lines[j] if j < len(lines) else "<missing>"
        b = got_lines[j] if j < len(got_lines) else "<missing>"
        if a != b and not (lo <= j <= hi):
            bad.append((j + 1, a, b))
    if bad:
        msg = [f"\n=== drop closer line {i+1} (form {lo+1}-{hi+1}): "
               f"{len(bad)} non-local changed lines ===\n"]
        for j, a, b in bad:
            msg.append(f"  line {j}:\n    - {a}\n    + {b}\n")
        return ("nonlocal", "".join(msg))
    # local: record which lines inside the form moved
    inside = [(j + 1, lines[j] if j < len(lines) else "<missing>",
               got_lines[j] if j < len(got_lines) else "<missing>")
              for j in range(max(len(got_lines), len(lines)))
              if (lines[j] if j < len(lines) else "<missing>") !=
                 (got_lines[j] if j < len(got_lines) else "<missing>")]
    msg = [f"\nlocal-only, drop closer line {i+1} (form {lo+1}-{hi+1}):\n"]
    for j, a, b in inside:
        msg.append(f"  line {j}:\n    - {a}\n    + {b}\n")
    return ("local", "".join(msg))


targets = [i for i, l in enumerate(lines) if l.rstrip().endswith((")", "]", "}"))]
counts = {}
with open(LOG, "w") as log, ThreadPoolExecutor(WORKERS) as pool:
    for kind, msg in pool.map(check, targets):
        counts[kind] = counts.get(kind, 0) + 1
        if msg:
            log.write(msg)

order = ["exact", "local", "nonlocal", "unbalanced", "nonidem", "failed"]
print(f"{os.path.basename(SRC):18s} n={len(targets):5d}  " +
      "  ".join(f"{k}={counts.get(k,0)}" for k in order))
sys.exit(1 if any(counts.get(k) for k in ("nonlocal", "unbalanced", "nonidem", "failed")) else 0)
