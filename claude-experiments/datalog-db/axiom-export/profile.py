import json
from collections import Counter

fields = {}
n = 0
with open("events.ndjson") as f:
    for line in f:
        n += 1
        if n > 2000:
            break
        e = json.loads(line)
        for k, v in e.items():
            d = fields.setdefault(k, {"nonnull": 0, "vals": Counter(), "types": set()})
            if v is not None:
                d["nonnull"] += 1
                d["types"].add(type(v).__name__)
                if len(d["vals"]) < 200:
                    d["vals"][str(v)[:30]] += 1

for k in sorted(fields):
    d = fields[k]
    distinct = len(d["vals"])
    types = "/".join(sorted(d["types"])) or "-"
    ex = ", ".join(list(d["vals"])[:3])
    print(f"{k:26s} {types:12s} nonnull {d['nonnull']:4d}/{n}  distinct~{distinct:3d}  e.g. {ex[:55]}")
