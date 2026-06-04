#!/bin/sh
# launch_ide.sh — open the datalog IDE as a tiled set of widget panes in the
# terminal-bevy "Datalog" project: schema + history sidebar (left), results
# table (top-right, large), query editor (bottom-right). Re-run anytime; it
# spawns fresh panes (there is no close-pane IPC, so close old ones by hand).
#
# Usage:  ./launch_ide.sh [PROJECT]      (default project: Datalog)
PROJECT="${1:-Datalog}"

python3 - "$PROJECT" <<'PY'
import socket, json, os, sys

project = sys.argv[1]

# Logical window size = physical (window.json) / scale factor (2 on retina).
# The pane rect coordinate space is logical px (probe-calibrated).
geo = json.load(open(os.path.expanduser("~/.terminal-bevy/window.json")))
W = geo["w"] / 2.0
H = geo["h"] / 2.0

SIDE = 300.0
rx, rw = SIDE, W - SIDE
results_h = round(H * 0.68)
editor_h  = H - results_h
schema_h  = round(H * 0.62)
hist_h    = H - schema_h

panes = [
    ("dlide_schema.rhai",  "datalog · schema",  0.0,      0.0,        SIDE, schema_h),
    ("dlide_history.rhai", "datalog · history", 0.0,      schema_h,   SIDE, hist_h),
    ("dlide_results.rhai", "datalog · results", rx,       0.0,        rw,   results_h),
    ("dlide_editor.rhai",  "datalog · query",   rx,       results_h,  rw,   editor_h),
]

sock = os.path.expanduser("~/.terminal-bevy/socket")
for script, title, x, y, w, h in panes:
    req = {"action": "spawn_widget", "command": script, "kind": "rhai_widget",
           "project": project, "title": title,
           "position": [float(x), float(y)], "size": [float(w), float(h)]}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock)
    s.sendall(json.dumps(req).encode())
    s.shutdown(socket.SHUT_WR)
    s.close()
    print(f"  {title:20s} pos=({x:.0f},{y:.0f}) size=({w:.0f}x{h:.0f})")
print(f"datalog IDE launched in project '{project}' (window {W:.0f}x{H:.0f} logical)")
PY
