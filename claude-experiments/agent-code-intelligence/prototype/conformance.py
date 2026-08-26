#!/usr/bin/env python3
"""Quarry protocol conformance suite (SPEC.md v0.1.1).

Black-box: exercises any implementation through its CLI. Provisions its own
fixture workspace, then checks the envelope laws (SPEC §2.5-§2.6) and the
amendments A1-A11 (§11) as named, mechanical tests. The point of the suite is
the point of the protocol: honesty must be a checkable property, not per-verb
discipline.

Usage:
    python3 conformance.py --cmd "python3 quarry.py" [-v] [--keep]

Exit code 0 iff every non-skipped check passes.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

FIXTURE = {
    "pkg/__init__.py": "",
    "pkg/alpha.py": '''\
class Alpha:
    def __init__(self) -> None:
        self.rate = 2

    def go(self, x: int) -> int:
        return x * self.rate


def helper() -> int:
    return Alpha().go(3)


def top() -> int:
    return mid() + 1


def mid() -> int:
    return leaf() * 2


def leaf() -> int:
    return 7
''',
    "pkg/beta.py": '''\
from .alpha import Alpha


class Beta:
    def go(self, x: int) -> int:
        return x - 1


def use_beta() -> int:
    b = Beta()
    return b.go(Alpha().go(2))


def dead_fn() -> None:
    """Never referenced anywhere."""
''',
    "data.yml": "rate: 5\n",
    "notes.md": "The go method multiplies by rate.\n",
}

VOLATILE = ("build_ms", "cache", "age_s")


class Suite:
    def __init__(self, cmd, ws, verbose=False):
        self.cmd = shlex.split(cmd)
        self.ws = ws
        self.verbose = verbose
        self.results = []           # (name, "pass"|"fail"|"skip", detail)

    # ------------------------------------------------------------ plumbing
    def call(self, verb, req=None, root=None, stdin_open=False, timeout=120):
        argv = self.cmd + [verb, "--root", root or self.ws,
                           "--json", json.dumps(req or {})]
        stdin = subprocess.PIPE if stdin_open else subprocess.DEVNULL
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, stdin=stdin)
        if proc.returncode != 0:
            raise AssertionError(
                f"transport failure rc={proc.returncode}: "
                f"{proc.stderr.strip()[:200]}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise AssertionError(
                f"stdout is not JSON: {proc.stdout[:200]!r}")

    def check(self, name, fn):
        try:
            fn()
            self.results.append((name, "pass", ""))
            if self.verbose:
                print(f"ok   {name}")
        except SkipCheck as e:
            self.results.append((name, "skip", str(e)))
            print(f"skip {name} — {e}")
        except AssertionError as e:
            self.results.append((name, "fail", str(e)))
            print(f"FAIL {name} — {e}")
        except Exception as e:
            self.results.append((name, "fail", f"{type(e).__name__}: {e}"))
            print(f"FAIL {name} — {type(e).__name__}: {e}")

    def write(self, rel, content):
        full = os.path.join(self.ws, rel)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    def read(self, rel):
        with open(os.path.join(self.ws, rel), encoding="utf-8") as f:
            return f.read()

    def remove(self, rel):
        os.remove(os.path.join(self.ws, rel))

    # ------------------------------------------------------------ helpers
    @staticmethod
    def is_error(env, code=None):
        assert env.get("status") == "error", \
            f"expected status=error, got {env.get('status')!r} " \
            f"(result: {str(env.get('result'))[:120]})"
        if code:
            actual = env.get("error", {}).get("code")
            assert actual == code, f"expected code={code}, got {actual!r}"

    @staticmethod
    def is_ok(env, status="complete"):
        assert env.get("status") == status, \
            f"expected status={status}, got {env.get('status')!r} " \
            f"({str(env.get('error') or env.get('notes'))[:150]})"

    @staticmethod
    def strip_volatile(env):
        env = json.loads(json.dumps(env))
        idx = env.get("index") or {}
        for k in VOLATILE:
            idx.pop(k, None)
        return env


class SkipCheck(Exception):
    pass


def build_checks(s):
    C = s.check

    # ---------------------------------------------------------- envelope
    def env_shape():
        env = s.call("resolve", {"name": "leaf"})
        for field in ("status", "confidence", "index", "coverage",
                      "snapshot", "result"):
            assert field in env, f"envelope missing {field!r}"
        assert env["confidence"] in ("compiler", "index", "syntactic",
                                     "textual")
    C("ENV-1 envelope has all mandatory fields", env_shape)

    C("ENV-2 unknown verb is UNSUPPORTED with supported list",
      lambda: (lambda e: (s.is_error(e, "UNSUPPORTED"),
                          [][0] if not e["error"].get("supported") else None)
               )(s.call("frobnicate")))

    def env_search_ast():
        e = s.call("search", {"pattern": {"ast": "$X", "lang": "python"}})
        s.is_error(e, "UNSUPPORTED")
        assert e["error"].get("fallback"), "UNSUPPORTED without a fallback"
    C("ENV-3 unimplemented pattern type is UNSUPPORTED with fallback",
      env_search_ast)

    # ---------------------------------------------------------- A1 locators
    C("A1-1 occurrence=0 is an error, never clamped",
      lambda: s.is_error(s.call("describe", {"at": {
          "path": "pkg/alpha.py", "line": 3, "anchor": "self.rate = 2",
          "occurrence": 0}})))
    C("A1-2 occurrence out of range is an error, never clamped",
      lambda: s.is_error(s.call("describe", {"at": {
          "path": "pkg/alpha.py", "line": 3, "anchor": "self.rate = 2",
          "occurrence": 99}})))

    def a1_anchor():
        e = s.call("describe", {"at": {"path": "pkg/alpha.py", "line": 3,
                                       "anchor": "text that is not there"}})
        s.is_error(e, "ANCHOR_MISMATCH")
        assert "actual_line_text" in e["error"], \
            "ANCHOR_MISMATCH must include the actual line text"
    C("A1-3 wrong anchor is ANCHOR_MISMATCH with actual line", a1_anchor)

    C("A1-4 out-of-range line is ANCHOR_MISMATCH",
      lambda: s.is_error(s.call("describe", {"at": {
          "path": "pkg/alpha.py", "line": 9999, "anchor": "x"}}),
          "ANCHOR_MISMATCH"))

    C("A1-5 anchor naming no symbol is an error, never line-fallback",
      lambda: s.is_error(s.call("describe", {"at": {
          "path": "pkg/alpha.py", "line": 3, "anchor": "= 2"}})))

    # ---------------------------------------------------------- A2 resolve
    def a2_fuzzy_refused():
        e = s.call("refs", {"name": "lea"})
        s.is_error(e)
        assert e["error"].get("nearest") or e["error"].get("candidates"), \
            "fuzzy near-miss must name the nearest symbols"
    C("A2-1 lone fuzzy hit is never silently substituted", a2_fuzzy_refused)

    def a2_fuzzy_optin():
        e = s.call("describe", {"name": "lea", "match": "fuzzy"})
        s.is_ok(e)
        assert e["result"].get("match") == "fuzzy", \
            "opt-in fuzzy resolution must be disclosed via match field"
    C("A2-2 opted-in fuzzy match is disclosed", a2_fuzzy_optin)

    # ---------------------------------------------------------- A3 blind spots
    def a3_ctor():
        e = s.call("refs", {"name": "Alpha.__init__"})
        s.is_ok(e)
        assert e["result"]["total"] >= 2, \
            f"Alpha() instantiations must count as __init__ refs " \
            f"(got {e['result']['total']})"
    C("A3-1 instantiations are references to __init__", a3_ctor)

    def a3_zero():
        e = s.call("refs", {"name": "dead_fn"})
        s.is_ok(e)
        assert e["result"]["total"] == 0
        assert e.get("notes"), "a zero at syntactic confidence needs notes"
    C("A3-2 an honest zero: complete + 0 + caveat notes", a3_zero)

    # ---------------------------------------------------------- A4 narrowing
    def a4_depth():
        e1 = s.call("graph", {"name": "top", "edge": "callees", "depth": 1})
        e2 = s.call("graph", {"name": "top", "edge": "callees", "depth": 2})
        s.is_ok(e1); s.is_ok(e2)
        n1, n2 = len(e1["result"]["edges"]), len(e2["result"]["edges"])
        assert n2 > n1, f"depth=2 ({n2} edges) must expand beyond " \
                        f"depth=1 ({n1}) for top->mid->leaf"
    C("A4-1 graph depth parameter is honored", a4_depth)

    def a4_path():
        e = s.call("search", {"pattern": {"symbol": {"kind": "function"}},
                              "where": {"path": "pkg/beta.py"}})
        s.is_ok(e)
        syms = e["result"]["symbols"]
        assert syms, "expected functions in pkg/beta.py"
        bad = [x["qualname"] for x in syms
               if x["def"]["path"] != "pkg/beta.py"]
        assert not bad, f"path filter dropped silently: {bad}"
    C("A4-2 search path filter is honored, never dropped", a4_path)

    def a4_check_scope():
        e = s.call("check", {"scope": {"paths": ["data.yml"]}})
        if e["status"] == "error":
            s.is_error(e, "UNSUPPORTED")
        else:
            assert e["status"] == "degraded" and \
                e["result"].get("skipped"), \
                "non-checkable scope must be UNSUPPORTED or degraded+skipped"
    C("A4-3 non-checkable scope never yields a vacuous clean", a4_check_scope)

    # ---------------------------------------------------------- A5 patches
    C("A5-1 hunk-less diff is PATCH_FAILED, never a no-op world",
      lambda: s.is_error(s.call("overlay.create", {"patch": {
          "format": "unified-diff",
          "content": "--- a/pkg/alpha.py\n+++ b/pkg/alpha.py\nGARBAGE\n"}}),
          "PATCH_FAILED"))
    C("A5-2 empty patch is PATCH_FAILED",
      lambda: s.is_error(s.call("overlay.create", {"patch": {
          "format": "unified-diff", "content": ""}}), "PATCH_FAILED"))

    # ---------------------------------------------------------- A6 overlays
    def a6_drift():
        e = s.call("overlay.create", {"patch": {"format": "files", "files": {
            "pkg/alpha.py": s.read("pkg/alpha.py") + "\n# probe\n"}}})
        s.is_ok(e)
        ovl = e["result"]["overlay_id"]
        base = s.read("notes.md")
        s.write("notes.md", base + "drift\n")
        try:
            e2 = s.call("refs", {"name": "leaf", "overlay": ovl})
            s.is_error(e2, "SNAPSHOT_GONE")
        finally:
            s.write("notes.md", base)
    C("A6-1 overlay base drift is SNAPSHOT_GONE", a6_drift)

    def a6_apply():
        e = s.call("overlay.create", {"patch": {"format": "files", "files": {
            "pkg/alpha.py": s.read("pkg/alpha.py")}}})
        s.is_ok(e)
        e2 = s.call("edit.rename", {"name": "leaf", "to": "leaf9",
                                    "mode": "apply",
                                    "overlay": e["result"]["overlay_id"]})
        s.is_error(e2)
    C("A6-2 applying a transform from an overlay world is refused", a6_apply)

    C("A6-3 overlay path escape is rejected",
      lambda: s.is_error(s.call("overlay.create", {"patch": {
          "format": "files", "files": {"../escape.py": "x = 1\n"}}}),
          "BAD_REQUEST"))

    # ---------------------------------------------------------- A7 transforms
    C("A7-1 rename to a keyword is BAD_REQUEST",
      lambda: s.is_error(s.call("edit.rename",
                                {"name": "leaf", "to": "class"}),
                         "BAD_REQUEST"))
    C("A7-2 rename to an invalid identifier is BAD_REQUEST",
      lambda: s.is_error(s.call("edit.rename",
                                {"name": "leaf", "to": "not valid"}),
                         "BAD_REQUEST"))

    def a7_truthful():
        e = s.call("edit.rename", {"name": "leaf", "to": "leaf_renamed"})
        s.is_ok(e)
        r = e["result"]
        assert r["applied"] is False, "diff mode must not claim applied"
        assert r["occurrences_rewritten"] >= 2 and r["diff"], \
            "def + call sites must be rewritten in the diff"
        before = s.read("pkg/alpha.py")
        e2 = s.call("edit.rename", {"name": "leaf", "to": "leaf_renamed",
                                    "mode": "apply"})
        s.is_ok(e2)
        after = s.read("pkg/alpha.py")
        assert e2["result"]["applied"] is True and before != after and \
            "leaf_renamed" in after, "applied:true must mean the disk changed"
        assert "post_check" in e2["result"]
        e3 = s.call("edit.rename", {"name": "leaf_renamed", "to": "leaf",
                                    "mode": "apply"})
        s.is_ok(e3)
        assert s.read("pkg/alpha.py") == before, "rename round-trip failed"
    C("A7-3 applied/files_changed/diff are truthful", a7_truthful)

    # ---------------------------------------------------------- A8 collateral
    def a8_same_named():
        e = s.call("edit.rename", {"sym": "py:pkg.alpha.Alpha.go",
                                   "to": "go2"})
        s.is_ok(e)
        conf = json.dumps(e["result"]["conflicts"])
        assert "Beta.go" in conf or "beta" in conf, \
            "same-named Beta.go must appear as a conflict entry"
    C("A8-1 same-named symbols are enumerated as conflicts", a8_same_named)

    # ---------------------------------------------------------- A9 errors
    def a9_coverage():
        e = s.call("describe", {"name": "no_such_symbol_zz"})
        s.is_error(e, "NOT_FOUND")
        assert "coverage" in e, "error envelopes must keep coverage"
    C("A9-1 error envelopes carry coverage", a9_coverage)

    def a9_broken_file():
        s.write("pkg/broken.py", "def broken(:\n")
        try:
            e = s.call("refs", {"name": "leaf"})
            assert e["status"] == "partial", \
                f"a syntax-broken file must force partial, got {e['status']}"
            assert "pkg/broken.py" in e["coverage"]["files_errored"]
            assert e["result"]["total"] >= 1, \
                "other files must still be answerable"
        finally:
            s.remove("pkg/broken.py")
    C("A9-2 un-indexable files force partial + are enumerated",
      a9_broken_file)

    C("A9-3 nonexistent workspace is WORKSPACE_NOT_FOUND",
      lambda: s.is_error(
          s.call("resolve", {"name": "x"},
                 root=os.path.join(s.ws, "no-such-dir-zz")),
          "WORKSPACE_NOT_FOUND"))

    C("A9-4 malformed request shape is BAD_REQUEST, not INTERNAL",
      lambda: s.is_error(s.call("describe", {"at": {"path": "pkg/alpha.py"}}),
                         "BAD_REQUEST"))

    # ---------------------------------------------------------- A10 budgets
    def a10_zero():
        e = s.call("refs", {"name": "rate", "budget": {"items": 0}})
        s.is_ok(e)
        kept = e["result"].get("occurrences", [])
        assert kept == [] and e["elided"]["count"] == e["result"]["total"], \
            "items:0 must keep nothing and elide exactly total"
    C("A10-1 items:0 is a valid count-only budget", a10_zero)

    C("A10-2 negative budget is BAD_REQUEST",
      lambda: s.is_error(s.call("refs", {"name": "rate",
                                         "budget": {"items": -1}}),
                         "BAD_REQUEST"))
    C("A10-3 unrecognized cursor is BAD_REQUEST, never a silent restart",
      lambda: s.is_error(s.call("refs", {"name": "rate", "cursor": "zzz"}),
                         "BAD_REQUEST"))

    def a10_walk():
        total, seen, cursor, rounds = None, [], None, 0
        while rounds < 50:
            req = {"pattern": {"regex": r"\brate\b"},
                   "budget": {"items": 1}}
            if cursor:
                req["cursor"] = cursor
            e = s.call("search", req)
            s.is_ok(e)
            if total is None:
                total = e["result"]["total"]
            assert e["result"]["total"] == total, "total drifted mid-walk"
            seen += [(m["path"], m["line"]) for m in e["result"]["matches"]]
            if "elided" not in e:
                break
            cursor = e["elided"]["cursor"]
            rounds += 1
        assert len(seen) == total, f"walk yielded {len(seen)} of {total}"
        assert len(set(seen)) == len(seen), "cursor walk produced duplicates"
    C("A10-4 cursor walk is complete, deterministic, duplicate-free",
      a10_walk)

    C("A10-5 an unfillable token budget is BUDGET_TOO_SMALL",
      lambda: s.is_error(s.call("refs", {"name": "rate",
                                         "budget": {"tokens": 1}}),
                         "BUDGET_TOO_SMALL"))

    def a10_batch():
        e = s.call("batch", {"queries": [
            {"id": "good", "verb": "describe", "name": "Alpha"},
            {"id": "bad", "verb": "nonsense"}]})
        assert "confidence" in e and "index" in e, \
            "batch wrapper must carry the full envelope"
        assert e["status"] in ("partial", "degraded"), \
            "one failed sub-query must not report complete"
        by_id = {r["id"]: r for r in e["results"]}
        assert by_id["good"]["status"] == "complete"
        assert by_id["bad"]["status"] == "error"
    C("A10-6 batch wrapper is enveloped; sub-errors are isolated",
      a10_batch)

    # ---------------------------------------------------------- A11 stdin
    def a11_stdin():
        e = s.call("describe", {"name": "Alpha"}, stdin_open=True,
                   timeout=30)
        s.is_ok(e)
    C("A11-1 an open, silent stdin pipe never hangs the CLI", a11_stdin)

    # ---------------------------------------------------------- determinism
    def det():
        e1 = s.strip_volatile(s.call("refs", {"name": "rate",
                                              "group_by": "file"}))
        e2 = s.strip_volatile(s.call("refs", {"name": "rate",
                                              "group_by": "file"}))
        assert e1 == e2, "identical request+snapshot must give identical " \
                         "responses (after stripping volatile timing fields)"
    C("DET-1 responses are deterministic per snapshot", det)

    def snap_tracks():
        e1 = s.call("refs", {"name": "leaf"})
        sid1, n1 = e1["snapshot"]["id"], e1["result"]["total"]
        base = s.read("pkg/beta.py")
        s.write("pkg/beta.py", base +
                "\n\ndef extra() -> int:\n    from .alpha import leaf\n"
                "    return leaf()\n")
        try:
            e2 = s.call("refs", {"name": "leaf"})
            assert e2["snapshot"]["id"] != sid1, \
                "snapshot id must change when content changes"
            assert e2["result"]["total"] > n1, \
                "new reference must be visible immediately"
        finally:
            s.write("pkg/beta.py", base)
    C("SNAP-1 snapshot ids and answers track the disk", snap_tracks)

    # ---------------------------------------------------------- L3 tier
    caps = s.call("capabilities")
    backend = caps["result"]["langs"].get("python", {}).get(
        "check_backend", "")
    has_l3 = "mypy" in backend

    def l3_attr():
        if not has_l3:
            raise SkipCheck("no mypy backend")
        bad = s.read("pkg/alpha.py").replace("self.rate", "self.rte", 1)
        e = s.call("overlay.create",
                   {"patch": {"format": "files",
                              "files": {"pkg/alpha.py": bad}}})
        s.is_ok(e)
        e2 = s.call("check", {"overlay": e["result"]["overlay_id"],
                              "scope": {"paths": ["pkg/alpha.py"]}})
        s.is_ok(e2)
        assert e2["result"]["clean"] is False and any(
            "rte" in d["message"] for d in e2["result"]["diagnostics"]), \
            "L3 check must catch the attribute typo"
    C("L3-1 check catches attribute errors (beyond name errors)", l3_attr)

    def l3_typeof():
        if not has_l3:
            raise SkipCheck("no mypy backend")
        e = s.call("type-of", {
            "at": {"path": "pkg/beta.py", "line": 10, "anchor": "b = Beta()"},
            "expr": "b"})
        s.is_ok(e)
        assert "Beta" in e["result"]["type"], \
            f"expected Beta, got {e['result']['type']!r}"
    C("L3-2 type-of returns a real inferred type", l3_typeof)

    def l3_typeof_honest():
        if not has_l3:
            raise SkipCheck("no mypy backend")
        e = s.call("type-of", {
            "at": {"path": "pkg/beta.py", "line": 10, "anchor": "b = Beta()"},
            "expr": "undefined_zz"})
        s.is_error(e)
        assert e["error"].get("mypy_errors"), \
            "an untypeable expression must error with the reason"
    C("L3-3 type-of on an undefined name errors, never reveals Any",
      l3_typeof_honest)

    # ---------------------------------------------------------- capabilities
    def cap_shape():
        s.is_ok(caps)
        langs = caps["result"]["langs"]
        assert langs, "capabilities must enumerate languages"
        for entry in langs.values():
            assert isinstance(entry.get("level"), int), \
                "each language needs an integer conformance level"
    C("CAP-1 capabilities declares integer levels per language", cap_shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="python3 quarry.py",
                    help="command that invokes the implementation")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="keep the fixture workspace")
    args = ap.parse_args()

    ws = tempfile.mkdtemp(prefix="quarry-conformance-")
    for rel, content in FIXTURE.items():
        full = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(full) or ws, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    s = Suite(args.cmd, ws, verbose=args.verbose)
    try:
        build_checks(s)
    finally:
        if args.keep:
            print(f"fixture kept at {ws}")
        else:
            shutil.rmtree(ws, ignore_errors=True)

    passed = sum(1 for _, st, _ in s.results if st == "pass")
    failed = [(n, d) for n, st, d in s.results if st == "fail"]
    skipped = sum(1 for _, st, _ in s.results if st == "skip")
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped "
          f"of {len(s.results)} checks")
    if failed:
        print("\nfailures:")
        for n, d in failed:
            print(f"  {n}: {d}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
