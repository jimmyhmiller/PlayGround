#!/usr/bin/env python3
"""
gc-rust benchmark harness: gc-rust vs Rust vs Go vs JVM.

For each benchmark it compiles all four languages, verifies they produce the
same numeric result (so the comparison is honest), then uses `hyperfine` to
measure wall-clock runtime, and records compile time + artifact size.

Competitor sources (Rust / Go / Java) are taken verbatim (single-threaded,
std-only variants) from public benchmark suites and credited in their headers;
see bench/suite/*/. gc-rust versions are faithful ports of the same algorithm.

Output: bench/results.json
"""
import json, os, re, subprocess, sys, time, shutil, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUITE = ROOT / "bench" / "suite"
BUILD = ROOT / "bench" / "build"
GCR = ROOT / "target" / "release" / "gcr"
RESULTS = ROOT / "bench" / "results.json"

# benchmark -> {arg (for rust/go/java CLI; gc-rust has it hardcoded to match),
#               java_main class name}
BENCHES = {
    "nbody":         {"arg": "5000000",  "java_main": "app"},
    "spectralnorm":  {"arg": "3000",     "java_main": "spectralnorm"},
    "fannkuchredux": {"arg": "11",       "java_main": "fannkuchredux"},
    "binarytrees":   {"arg": "16",       "java_main": "app"},
}

# hyperfine tuning per benchmark (slow allocation-heavy ones get fewer runs)
RUNS = {"nbody": 10, "spectralnorm": 10, "fannkuchredux": 10, "binarytrees": 8}

LANGS = ["gcrust", "rust", "go", "java"]
LANG_LABEL = {"gcrust": "gc-rust", "rust": "Rust", "go": "Go", "java": "JVM (Java)"}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def numeric_signature(stdout):
    """Comparable signature: ints exact, floats rounded to 6 significant decimals."""
    toks = re.findall(r"-?\d+\.\d+|-?\d+", stdout)
    sig = []
    for t in toks:
        if "." in t:
            sig.append(f"{float(t):.6f}")
        else:
            sig.append(t)
    return sig


def float_set(stdout):
    return [round(float(t), 6) for t in re.findall(r"-?\d+\.\d+", stdout)]


def int_set(stdout):
    # tokenize floats-first so fractional digits aren't mistaken for integers
    toks = re.findall(r"-?\d+\.\d+|-?\d+", stdout)
    return [int(t) for t in toks if "." not in t]


def compile_all(bench):
    d = SUITE / bench
    arg = BENCHES[bench]["arg"]
    bdir = BUILD / bench
    bdir.mkdir(parents=True, exist_ok=True)
    artifacts = {}  # lang -> {cmd, compile_ms, size_bytes}

    # --- Rust ---
    rs = d / f"{bench}.rs"
    out = bdir / f"{bench}_rust"
    t = time.perf_counter()
    r = run(["rustc", "-O", "-C", "panic=abort", str(rs), "-o", str(out)])
    cms = (time.perf_counter() - t) * 1000
    if r.returncode != 0:
        print(f"  [rust] COMPILE FAIL:\n{r.stderr}"); sys.exit(1)
    artifacts["rust"] = {"cmd": [str(out), arg], "compile_ms": cms,
                         "size_bytes": out.stat().st_size}

    # --- Go ---
    go = d / f"{bench}.go"
    out = bdir / f"{bench}_go"
    env = dict(os.environ, CGO_ENABLED="0")
    t = time.perf_counter()
    r = run(["go", "build", "-o", str(out), str(go)], env=env)
    cms = (time.perf_counter() - t) * 1000
    if r.returncode != 0:
        print(f"  [go] COMPILE FAIL:\n{r.stderr}"); sys.exit(1)
    artifacts["go"] = {"cmd": [str(out), arg], "compile_ms": cms,
                       "size_bytes": out.stat().st_size}

    # --- Java ---
    jdir = bdir / "java"
    if jdir.exists():
        shutil.rmtree(jdir)
    jdir.mkdir()
    t = time.perf_counter()
    r = run(["javac", "-d", str(jdir)] + [str(p) for p in d.glob("*.java")])
    cms = (time.perf_counter() - t) * 1000
    if r.returncode != 0:
        print(f"  [java] COMPILE FAIL:\n{r.stderr}"); sys.exit(1)
    jsize = sum(p.stat().st_size for p in jdir.glob("*.class"))
    artifacts["java"] = {
        "cmd": ["java", "-cp", str(jdir), BENCHES[bench]["java_main"], arg],
        "compile_ms": cms, "size_bytes": jsize}

    # --- gc-rust (AOT) ---
    gcr_src = d / f"{bench}.gcr"
    t = time.perf_counter()
    # gcr build writes a binary named after the file stem into CWD; run from bdir
    r = run([str(GCR), "build", str(gcr_src)], cwd=str(bdir))
    cms = (time.perf_counter() - t) * 1000
    out = bdir / bench  # stem
    if r.returncode != 0 or not out.exists():
        print(f"  [gcrust] BUILD FAIL:\n{r.stdout}\n{r.stderr}"); sys.exit(1)
    gout = bdir / f"{bench}_gcrust"
    out.rename(gout)
    artifacts["gcrust"] = {"cmd": [str(gout)], "compile_ms": cms,
                           "size_bytes": gout.stat().st_size}
    return artifacts


def verify(bench, artifacts):
    """Run each once; confirm all four agree numerically."""
    outs = {}
    for lang in LANGS:
        r = run(artifacts[lang]["cmd"])
        if r.returncode != 0:
            print(f"  [{lang}] RUN FAIL rc={r.returncode}:\n{r.stderr[:400]}"); sys.exit(1)
        outs[lang] = r.stdout
    # floats compared with tolerance, ints exact
    ref_floats = float_set(outs["rust"])
    ref_ints = int_set(outs["rust"])
    ok = True
    detail = {}
    for lang in LANGS:
        f = float_set(outs[lang])
        i = int_set(outs[lang])
        # match the meaningful values: every reference float must appear (±1e-5)
        fmatch = all(any(abs(a - b) <= 1e-5 for b in f) for a in ref_floats)
        imatch = all(v in i for v in ref_ints) if ref_ints else True
        detail[lang] = {"floats": f, "ints_sample": i[:6]}
        if not (fmatch and imatch):
            ok = False
            print(f"  [{lang}] MISMATCH floats={f} ints={i}")
    status = "VERIFIED-MATCH" if ok else "MISMATCH"
    print(f"  verify: {status}  (ref floats={ref_floats}, ref ints[:6]={ref_ints[:6]})")
    return status, detail


def hyperfine(bench, artifacts):
    tmp = BUILD / bench / "hf.json"
    cmds = []
    names = []
    for lang in LANGS:
        cmds.append(" ".join(artifacts[lang]["cmd"]))
        names.append(LANG_LABEL[lang])
    args = ["hyperfine", "--warmup", "2", "--min-runs", str(RUNS[bench]),
            "--export-json", str(tmp), "--shell", "none"]
    for n, c in zip(names, cmds):
        args += ["-n", n, c]
    r = run(args)
    if r.returncode != 0:
        print(f"  [hyperfine] FAIL:\n{r.stderr}"); sys.exit(1)
    data = json.loads(tmp.read_text())
    res = {}
    for lang, entry in zip(LANGS, data["results"]):
        res[lang] = {
            "mean_s": entry["mean"], "stddev_s": entry.get("stddev", 0.0),
            "min_s": entry["min"], "max_s": entry["max"],
            "median_s": entry.get("median", entry["mean"]),
        }
    return res


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    results = {"benchmarks": {}, "langs": LANG_LABEL, "lang_order": LANGS,
               "meta": {"args": {b: BENCHES[b]["arg"] for b in BENCHES}}}
    for bench in BENCHES:
        print(f"\n=== {bench} (N={BENCHES[bench]['arg']}) ===")
        art = compile_all(bench)
        status, detail = verify(bench, art)
        timing = hyperfine(bench, art)
        entry = {"arg": BENCHES[bench]["arg"], "verify": status, "langs": {}}
        for lang in LANGS:
            entry["langs"][lang] = {
                "time": timing[lang],
                "compile_ms": art[lang]["compile_ms"],
                "size_bytes": art[lang]["size_bytes"],
                "output": detail[lang],
            }
            t = timing[lang]
            print(f"  {LANG_LABEL[lang]:12s} {t['mean_s']*1000:8.1f} ms "
                  f"± {t['stddev_s']*1000:6.1f}  "
                  f"(compile {art[lang]['compile_ms']:6.0f} ms, "
                  f"{art[lang]['size_bytes']/1024:7.1f} KiB)")
        # ratios vs rust
        base = timing["rust"]["mean_s"]
        for lang in LANGS:
            entry["langs"][lang]["ratio_vs_rust"] = timing[lang]["mean_s"] / base
        results["benchmarks"][bench] = entry
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
