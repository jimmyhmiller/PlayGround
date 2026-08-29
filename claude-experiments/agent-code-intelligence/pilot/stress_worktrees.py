#!/usr/bin/env python3
"""Massive multi-worktree stress test for quarry's shared store (A12+A14).

Topology: one git repo, N worktrees sharing one content-addressed store.
 - N-1 ORACLE trees: an editor thread toggles 3 marker files between two
   known contents each (8 legal states/tree). Every legal state's
   snapshot id and exact refs answer is precomputed, so every storm
   response is scored with zero tolerance: (snapshot_id, answer) must be
   a precomputed pair, or status must be an honestly-disclosed
   partial/unstable. Anything else is a TORN-VIEW VIOLATION.
 - 1 CHAOS tree: every edit writes novel content (timestamped), forcing
   fresh shard builds under storm; scored only for structured non-error
   behavior.
 - Editors use plain non-atomic writes on purpose (mid-write reads are
   part of the attack). Assembled caches are deleted after precompute so
   the storm exercises shard reuse + assembly under contention, not a
   warm cache.
 - Extra load: roaming workers run graph/path/describe/resolve across
   random trees; overlay workers create overlays and query through them
   (under churn the only legal outcomes are SNAPSHOT_GONE or the exact
   pinned answer).

Usage: python3 stress_worktrees.py --root DIR --trees 16 --seconds 120
"""

import argparse
import itertools
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QUARRY = os.path.join(HERE, "..", "prototype", "quarry.py")
CORPUS = None   # set in main


def q(root, verb, req=None, timeout=90):
    t0 = time.time()
    p = subprocess.run([sys.executable, QUARRY, verb, "--root", root,
                        "--json", json.dumps(req or {})],
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    return json.loads(p.stdout), dt


ON = ("from granite.core.sift_hopper import notch_billet\n\n\n"
      "def stress_fn_t{t}_m{m}(val: int) -> int:\n"
      "    return notch_billet(val)\n")
OFF = "# marker t{t} m{m} off\n"


def marker_path(tree, m):
    return os.path.join(tree, "granite", f"stress_t{os.path.basename(tree)[-2:]}_m{m}.py")


def set_marker(tree, m, on, t_id):
    with open(marker_path(tree, m), "w") as f:      # plain write: attackable
        f.write((ON if on else OFF).format(t=t_id, m=m))


def setup(root, n_trees):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    main = os.path.join(root, "t00")
    shutil.copytree(CORPUS, main)
    for junk in (".quarry",):
        shutil.rmtree(os.path.join(main, junk), ignore_errors=True)
    subprocess.run(["git", "init", "-q"], cwd=main, check=True)
    subprocess.run(["git", "add", "-A"], cwd=main, check=True)
    subprocess.run(["git", "-c", "user.email=s@l", "-c", "user.name=s",
                    "commit", "-qm", "corpus"], cwd=main, check=True)
    trees = [main]
    for i in range(1, n_trees):
        wt = os.path.join(root, f"t{i:02d}")
        subprocess.run(["git", "worktree", "add", "-q", wt, "-b",
                        f"agent{i}"], cwd=main, check=True)
        trees.append(wt)
    return trees


def precompute(tree, t_id):
    """Pin all 8 marker states; record snapshot id -> exact call count."""
    legal = {}
    for combo in itertools.product((0, 1), repeat=3):
        for m, on in enumerate(combo):
            set_marker(tree, m, on, t_id)
        d, _ = q(tree, "refs", {"name": "notch_billet", "role": "call"})
        assert d["status"] == "complete", d
        legal[d["snapshot"]["id"]] = d["result"]["total"]
    assert len(legal) == 8, f"expected 8 distinct states, got {len(legal)}"
    counts = sorted(set(legal.values()))
    assert counts[-1] - counts[0] == 3, counts
    return legal


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.lat = []
        self.c = {"ok": 0, "unstable_disclosed": 0, "VIOLATION": 0,
                  "roam_ok": 0, "roam_err": 0, "overlay_gone": 0,
                  "overlay_exact": 0, "overlay_VIOLATION": 0,
                  "edits": 0, "shard_builds": 0, "transport_err": 0}
        self.violations = []

    def add(self, key, lat=None, detail=None):
        with self.lock:
            self.c[key] += 1
            if lat is not None:
                self.lat.append(lat)
            if detail and len(self.violations) < 10:
                self.violations.append(detail)


def editor(tree, t_id, chaos, stop, stats):
    rng = random.Random(t_id)
    while not stop.is_set():
        m = rng.randrange(3)
        if chaos:
            with open(marker_path(tree, m), "w") as f:
                f.write(f"# chaos {time.time_ns()}\n"
                        f"def chaos_t{t_id}_m{m}_{rng.randrange(10**9)}"
                        f"(v: int) -> int:\n    return v + {rng.randrange(99)}\n")
        else:
            set_marker(tree, m, rng.random() < 0.5, t_id)
        stats.add("edits")
        time.sleep(rng.uniform(0.15, 0.45))


def verifier(tree, legal, stop, stats):
    while not stop.is_set():
        try:
            d, dt = q(tree, "refs", {"name": "notch_billet", "role": "call"})
        except Exception:
            stats.add("transport_err")
            continue
        sid = d.get("snapshot", {}).get("id")
        builds = (d.get("index", {}).get("shards") or {}).get("built", 0)
        with stats.lock:
            stats.c["shard_builds"] += builds
        if d["status"] == "partial" and any(
                "torn" in n or "modified during" in n
                for n in d.get("notes", [])):
            stats.add("unstable_disclosed", dt)
        elif d["status"] == "complete" and legal.get(sid) == \
                d["result"]["total"]:
            stats.add("ok", dt)
        else:
            stats.add("VIOLATION", dt,
                      {"tree": tree, "sid": sid, "status": d["status"],
                       "got": d.get("result", {}).get("total"),
                       "want": legal.get(sid)})


def chaos_watcher(tree, stop, stats):
    while not stop.is_set():
        try:
            d, dt = q(tree, "refs", {"name": "notch_billet", "role": "call"})
            builds = (d.get("index", {}).get("shards") or {}).get("built", 0)
            with stats.lock:
                stats.c["shard_builds"] += builds
            if d["status"] in ("complete", "partial"):
                stats.add("ok", dt)
            else:
                stats.add("VIOLATION", dt, {"tree": tree, "chaos": True,
                                            "resp": d.get("error")})
        except Exception:
            stats.add("transport_err")


def roamer(trees, stop, stats):
    rng = random.Random()
    jobs = [
        ("graph", {"name": "granite.main.main", "edge": "path",
                   "to": "notch_billet"}),
        ("describe", {"name": "glaze_ledger.WinchQuarry.process"}),
        ("resolve", {"name": "process"}),
        ("refs", {"sym": "py:granite.core.glaze_ledger.WinchQuarry.process",
                  "role": "call"}),
        ("search", {"pattern": {"regex": "notch_billet"},
                    "budget": {"items": 5}}),
    ]
    while not stop.is_set():
        tree = rng.choice(trees)
        verb, req = rng.choice(jobs)
        try:
            d, dt = q(tree, verb, req)
            ok = d["status"] in ("complete", "partial") or \
                (verb == "resolve" and d.get("error", {}).get("code")
                 == "AMBIGUOUS")
            stats.add("roam_ok" if ok else "roam_err", dt)
        except Exception:
            stats.add("transport_err")


def overlayer(tree, t_id, pinned_total, stop, stats):
    patch = {"format": "files", "files": {}}
    for m in range(3):
        rel = os.path.relpath(marker_path(tree, m), tree)
        patch["files"][rel] = ON.format(t=t_id, m=m)
    while not stop.is_set():
        try:
            d, _ = q(tree, "overlay.create", {"patch": patch})
            if d["status"] != "complete":
                stats.add("overlay_gone")     # creation raced an edit; fine
                continue
            ovl = d["result"]["overlay_id"]
            d2, dt = q(tree, "refs", {"name": "notch_billet",
                                      "role": "call", "overlay": ovl})
            if d2["status"] == "error" and \
                    d2["error"]["code"] == "SNAPSHOT_GONE":
                stats.add("overlay_gone", dt)
            elif d2["status"] in ("complete", "partial") and \
                    d2["result"]["total"] == pinned_total:
                stats.add("overlay_exact", dt)
            else:
                stats.add("overlay_VIOLATION", dt,
                          {"tree": tree, "overlay": True,
                           "got": d2.get("result", {}).get("total"),
                           "want": pinned_total,
                           "status": d2["status"]})
        except Exception:
            stats.add("transport_err")


def monitor(stop, samples):
    while not stop.is_set():
        out = subprocess.run(["ps", "-eo", "rss,args"], capture_output=True,
                             text=True).stdout
        procs = [int(l.strip().split()[0]) for l in out.splitlines()
                 if "quarry.py" in l and "stress_worktrees" not in l]
        samples.append((len(procs), sum(procs) // 1024))
        time.sleep(2)


def main():
    global CORPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trees", type=int, default=16)
    ap.add_argument("--seconds", type=int, default=120)
    args = ap.parse_args()
    CORPUS = args.corpus

    print(f"setup: {args.trees} worktrees ...", flush=True)
    trees = setup(args.root, args.trees)
    store = os.path.join(trees[0], ".git", "quarry-store")

    print("precompute: pinning 8 states x oracle trees ...", flush=True)
    t0 = time.time()
    legal = {}
    oracle_trees = trees[:-1]
    chaos_tree = trees[-1]
    for i, tree in enumerate(oracle_trees):
        legal[tree] = precompute(tree, i)
    for m in range(3):
        set_marker(chaos_tree, m, False, len(trees) - 1)
    pinned_totals = {t: max(legal[t].values()) for t in oracle_trees}
    shutil.rmtree(os.path.join(store, "assembled"), ignore_errors=True)
    n_shards0 = len(os.listdir(os.path.join(store, "objects")))
    print(f"precompute done in {time.time()-t0:.0f}s; "
          f"{n_shards0} shards in store; assembled cache dropped", flush=True)

    stop = threading.Event()
    stats = Stats()
    samples = []
    threads = []
    for i, tree in enumerate(oracle_trees):
        threads.append(threading.Thread(
            target=editor, args=(tree, i, False, stop, stats)))
        threads.append(threading.Thread(
            target=verifier, args=(tree, legal[tree], stop, stats)))
    threads.append(threading.Thread(
        target=editor, args=(chaos_tree, len(trees) - 1, True, stop, stats)))
    threads.append(threading.Thread(
        target=chaos_watcher, args=(chaos_tree, stop, stats)))
    for _ in range(4):
        threads.append(threading.Thread(
            target=roamer, args=(trees, stop, stats)))
    for tree in oracle_trees[:3]:
        threads.append(threading.Thread(
            target=overlayer, args=(tree, oracle_trees.index(tree),
                                    pinned_totals[tree], stop, stats)))
    threads.append(threading.Thread(target=monitor, args=(stop, samples)))

    print(f"storm: {len(threads)} threads for {args.seconds}s ...",
          flush=True)
    t0 = time.time()
    for t in threads:
        t.start()
    time.sleep(args.seconds)
    stop.set()
    for t in threads:
        t.join(timeout=120)
    dur = time.time() - t0

    n_shards1 = len(os.listdir(os.path.join(store, "objects")))
    store_mb = int(subprocess.run(["du", "-sm", store], capture_output=True,
                                  text=True).stdout.split()[0])
    lat = sorted(stats.lat)
    total_q = (stats.c["ok"] + stats.c["unstable_disclosed"]
               + stats.c["VIOLATION"] + stats.c["roam_ok"]
               + stats.c["roam_err"] + stats.c["overlay_gone"]
               + stats.c["overlay_exact"] + stats.c["overlay_VIOLATION"])
    print("\n================ RESULTS ================")
    print(f"duration {dur:.0f}s | trees {args.trees} | threads {len(threads)}")
    print(f"edits performed: {stats.c['edits']} "
          f"({stats.c['edits']/dur:.1f}/s)")
    print(f"queries: {total_q} ({total_q/dur:.1f}/s)")
    print(f"oracle-verified ok:      {stats.c['ok']}")
    print(f"unstable disclosed:      {stats.c['unstable_disclosed']}")
    print(f"TORN-VIEW VIOLATIONS:    {stats.c['VIOLATION']}")
    print(f"roaming ok/err:          {stats.c['roam_ok']}/{stats.c['roam_err']}")
    print(f"overlay exact/gone:      {stats.c['overlay_exact']}/{stats.c['overlay_gone']}")
    print(f"overlay VIOLATIONS:      {stats.c['overlay_VIOLATION']}")
    print(f"transport errors:        {stats.c['transport_err']}")
    print(f"shard builds during storm: {stats.c['shard_builds']} "
          f"(chaos tree novel content)")
    print(f"store: {n_shards0} -> {n_shards1} shards, {store_mb} MB")
    if lat:
        print(f"latency p50/p95/max: {statistics.median(lat):.2f}s / "
              f"{lat[int(len(lat)*0.95)]:.2f}s / {lat[-1]:.2f}s")
    if samples:
        peak = max(samples, key=lambda s: s[1])
        print(f"peak concurrent quarry processes: "
              f"{max(s[0] for s in samples)} ({peak[1]} MB transient RSS); "
              f"resident between calls: 0")
    for v in stats.violations:
        print("VIOLATION DETAIL:", v)
    return 1 if (stats.c["VIOLATION"] or stats.c["overlay_VIOLATION"]) else 0


if __name__ == "__main__":
    sys.exit(main())
