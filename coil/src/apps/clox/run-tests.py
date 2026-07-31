#!/usr/bin/env python3
"""Crafting Interpreters test harness for the Coil clox port.

Faithfully replicates the expectation-parsing and validation logic of
craftinginterpreters/tool/bin/test.dart for the "clox" suite (which runs
every test except test/scanning and test/expressions).

Usage:
  ./run-tests.py <clox-binary> [test-dir] [--filter SUBSTR] [-v]

The clox binary is invoked as: <clox-binary> <path-to.lox>
"""
import os, re, subprocess, sys

EXPECT_OUTPUT   = re.compile(r"// expect: ?(.*)")
EXPECT_ERROR    = re.compile(r"// (Error.*)")
ERROR_LINE      = re.compile(r"// \[((java|c) )?line (\d+)\] (Error.*)")
EXPECT_RT_ERROR = re.compile(r"// expect runtime error: (.+)")
SYNTAX_ERROR    = re.compile(r"\[.*line (\d+)\] (Error.+)")
STACK_TRACE     = re.compile(r"\[line (\d+)\]")
NONTEST         = re.compile(r"// nontest")


class Test:
    def __init__(self, path):
        self.path = path
        self.expected_output = []        # list of (line_num, text)
        self.expected_errors = set()     # {"[N] Error..."}
        self.expected_rt_error = None
        self.rt_error_line = None
        self.expected_exit = 0
        self.is_test = True

    def parse(self):
        with open(self.path) as f:
            lines = f.read().split("\n")
        for i, line in enumerate(lines, 1):
            if NONTEST.search(line):
                self.is_test = False
                return
            m = EXPECT_OUTPUT.search(line)
            if m:
                self.expected_output.append((i, m.group(1)))
                continue
            m = EXPECT_ERROR.search(line)
            if m:
                self.expected_errors.add(f"[{i}] {m.group(1)}")
                self.expected_exit = 65
                continue
            m = ERROR_LINE.search(line)
            if m:
                # group(2) is the language tag (java|c) or None
                lang = m.group(2)
                if lang is None or lang == "c":
                    self.expected_errors.add(f"[{m.group(3)}] {m.group(4)}")
                    self.expected_exit = 65
                continue
            m = EXPECT_RT_ERROR.search(line)
            if m:
                self.rt_error_line = i
                self.expected_rt_error = m.group(1)
                self.expected_exit = 70

    def run(self, binary):
        try:
            p = subprocess.run([binary, self.path], capture_output=True,
                               text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return ["Timed out."]
        out = p.stdout.split("\n")
        err = p.stderr.split("\n")
        failures = []
        if self.expected_rt_error is not None:
            failures += self._validate_rt_error(err)
        else:
            failures += self._validate_compile_errors(err)
        failures += self._validate_exit(p.returncode, err)
        failures += self._validate_output(out)
        return failures

    def _validate_rt_error(self, err):
        f = []
        if len(err) < 2:
            return [f"Expected runtime error '{self.expected_rt_error}' and got none."]
        if err[0] != self.expected_rt_error:
            f.append(f"Expected runtime error '{self.expected_rt_error}' and got:")
            f.append(err[0])
        match = None
        for line in err[1:]:
            match = STACK_TRACE.search(line)
            if match:
                break
        if match is None:
            f.append("Expected stack trace and got: " + repr(err[1:]))
        else:
            if int(match.group(1)) != self.rt_error_line:
                f.append(f"Expected runtime error on line {self.rt_error_line} "
                         f"but was on line {int(match.group(1))}.")
        return f

    def _validate_compile_errors(self, err):
        f = []
        found = set()
        for line in err:
            m = SYNTAX_ERROR.search(line)
            if m:
                e = f"[{m.group(1)}] {m.group(2)}"
                if e in self.expected_errors:
                    found.add(e)
                else:
                    f.append("Unexpected error: " + line)
            elif line != "":
                f.append("Unexpected output on stderr: " + line)
        for e in self.expected_errors - found:
            f.append("Missing expected error: " + e)
        return f

    def _validate_exit(self, code, err):
        if code == self.expected_exit:
            return []
        return [f"Expected return code {self.expected_exit} and got {code}. "
                f"Stderr: {err[:10]}"]

    def _validate_output(self, out):
        if out and out[-1] == "":
            out = out[:-1]
        f = []
        for i, line in enumerate(out):
            if i >= len(self.expected_output):
                f.append(f"Got output '{line}' when none was expected.")
                continue
            ln, txt = self.expected_output[i]
            if txt != line:
                f.append(f"Expected output '{txt}' on line {ln} and got '{line}'.")
        for i in range(len(out), len(self.expected_output)):
            ln, txt = self.expected_output[i]
            f.append(f"Missing expected output '{txt}' on line {ln}.")
        return f


def main():
    args = [a for a in sys.argv[1:]]
    verbose = False
    filt = None
    if "-v" in args:
        verbose = True; args.remove("-v")
    if "--filter" in args:
        i = args.index("--filter"); filt = args[i+1]; del args[i:i+2]
    if not args:
        print("usage: run-tests.py <clox-binary> [test-dir] [--filter S] [-v]")
        sys.exit(2)
    binary = os.path.abspath(args[0])
    test_dir = args[1] if len(args) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tests", "lox")

    files = []
    for root, _, names in os.walk(test_dir):
        for n in sorted(names):
            if n.endswith(".lox"):
                files.append(os.path.join(root, n))
    files.sort()

    passed = failed = skipped = 0
    fail_names = []
    for path in files:
        rel = os.path.relpath(path, test_dir)
        if filt and filt not in rel:
            continue
        t = Test(path)
        t.parse()
        if not t.is_test:
            skipped += 1
            continue
        failures = t.run(binary)
        if failures:
            failed += 1
            fail_names.append(rel)
            if verbose:
                print(f"FAIL {rel}")
                for m in failures:
                    print("     " + m)
        else:
            passed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped "
          f"({passed+failed} tests)")
    if fail_names and not verbose:
        print("Failed:")
        for n in fail_names:
            print("  " + n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
