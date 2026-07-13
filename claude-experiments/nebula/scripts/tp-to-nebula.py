#!/usr/bin/env python3
"""Convert a Turbopack task-graph dump to nebula's typed-edge JSON.

Usage:  python3 tp-to-nebula.py IN.json OUT.json

Each node's `children`, `deps`, and `cell_deps` arrays become three separately
colored edge sets. Every scalar node field is carried through as an attribute
(so color-by-attribute / filter / inspect work); `owned_cells` is reduced to a
count. Missing arrays are just skipped, so dumps with only some of these work.
"""
import json
import sys

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: tp-to-nebula.py IN.json OUT.json")
    src, dst = sys.argv[1], sys.argv[2]
    nodes = json.load(open(src))["nodes"]

    children, deps, cell_deps = [], [], []
    out_nodes = []
    for n in nodes:
        u = n["id"]
        for c in n.get("children", []):
            children.append([u, c])
        for c in n.get("deps", []):
            deps.append([u, c])
        seen = set()
        for cd in n.get("cell_deps", []):
            t = cd.get("task") if isinstance(cd, dict) else cd
            if t is not None and t not in seen:
                seen.add(t)
                cell_deps.append([u, t])
        # Keep scalar fields; drop the edge arrays; summarize owned_cells.
        nn = {k: v for k, v in n.items()
              if k not in ("children", "deps", "cell_deps", "owned_cells")}
        if "owned_cells" in n:
            nn["owned_cells"] = len(n["owned_cells"])
        out_nodes.append(nn)

    edge_types = []
    if children:
        edge_types.append({"name": "children", "color": [90, 150, 255], "edges": children})
    if deps:
        edge_types.append({"name": "deps", "color": [255, 150, 60], "edges": deps})
    if cell_deps:
        edge_types.append({"name": "cell_deps", "color": [120, 215, 130], "edges": cell_deps})

    json.dump({"nodes": out_nodes, "edge_types": edge_types}, open(dst, "w"))
    print(f"nodes {len(out_nodes)}  "
          + "  ".join(f"{t['name']} {len(t['edges'])}" for t in edge_types))
    print("wrote", dst)

if __name__ == "__main__":
    main()
