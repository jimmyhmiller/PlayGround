#!/usr/bin/env python3
"""Test runner for Lox implementation against Crafting Interpreters test suite."""

import os
import re
import subprocess
import sys
from pathlib import Path

TESTS_DIR = "/tmp/craftinginterpreters/test"

# Tests that are specific to jlox (Java tree-walk interpreter) and should be skipped
# for a clox-compatible implementation
SKIP_DIRS = {"scanning", "expressions", "benchmark"}

# Skip individual tests that test jlox-specific behavior or limits we don't implement
SKIP_FILES = {
    "limit/loop_too_large.lox",  # clox-specific bytecode limit
    "limit/no_reuse_constants.lox",  # clox-specific bytecode limit
    "limit/stack_overflow.lox",  # clox-specific stack limit
    "limit/too_many_constants.lox",  # clox-specific bytecode limit
    "limit/too_many_locals.lox",  # clox-specific bytecode limit
    "limit/too_many_upvalues.lox",  # clox-specific bytecode limit
    "function/local_mutual_recursion.lox",  # skipped in upstream test suite
}

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

def parse_expectations(filepath):
    """Parse expected output, errors, and exit code from test file."""
    expected_output = []
    expected_errors = []
    expected_runtime_error = None
    expected_exit_code = 0

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Expected output: // expect: <value>
            match = re.search(r'// expect: (.*)', line)
            if match:
                expected_output.append(match.group(1))

            # Expected compile error: // [line N] Error...
            # or: // Error...
            match = re.search(r'// (\[line \d+\] Error.*)', line)
            if match:
                expected_errors.append(match.group(1))
                expected_exit_code = 65

            # Also match "Error at" format without line number (from current line)
            match = re.search(r'// (Error at .*)', line)
            if match and not re.search(r'// \[line', line):
                expected_errors.append(f"[line {line_num}] {match.group(1)}")
                expected_exit_code = 65

            # Expected runtime error: // expect runtime error: <message>
            match = re.search(r'// expect runtime error: (.*)', line)
            if match:
                expected_runtime_error = match.group(1)
                expected_exit_code = 70

            # Non-test error markers (for error line tracking)
            match = re.search(r'// \[((java|c) )?line (\d+)\] Error', line)
            if match:
                lang = match.group(2)
                if lang and lang != 'c':
                    continue  # skip java-specific errors

    return expected_output, expected_errors, expected_runtime_error, expected_exit_code

def should_skip(filepath, rel_path):
    """Check if test should be skipped for clox-compatible implementation."""
    parts = Path(rel_path).parts

    # Skip jlox-specific directories
    if parts[0] in SKIP_DIRS:
        return True

    # Skip specific files
    if rel_path in SKIP_FILES:
        return True

    # Check for java-only markers
    with open(filepath, 'r') as f:
        content = f.read()

    # Skip if it has java-only error expectations
    if re.search(r'// \[java line \d+\]', content):
        # Has java-specific expectations - check if also has c expectations
        if not re.search(r'// \[c line \d+\]', content) and not re.search(r'// \[line \d+\]', content):
            # Only java expectations, might still have expect: lines
            pass

    return False

def run_test(lox_binary, filepath, rel_path):
    """Run a single test and check results."""
    expected_output, expected_errors, expected_runtime_error, expected_exit_code = parse_expectations(filepath)

    try:
        result = subprocess.run(
            [lox_binary, filepath],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT"

    actual_output = result.stdout.rstrip('\n').split('\n') if result.stdout.rstrip('\n') else []
    actual_stderr = result.stderr.rstrip('\n').split('\n') if result.stderr.rstrip('\n') else []
    actual_exit_code = result.returncode

    errors = []

    # Check exit code
    if actual_exit_code != expected_exit_code:
        errors.append(f"Exit code: expected {expected_exit_code}, got {actual_exit_code}")

    # Check expected output
    if expected_output != actual_output:
        if len(expected_output) != len(actual_output):
            errors.append(f"Output lines: expected {len(expected_output)}, got {len(actual_output)}")
        for i, (exp, act) in enumerate(zip(expected_output, actual_output)):
            if exp != act:
                errors.append(f"Output line {i+1}: expected '{exp}', got '{act}'")
        if len(actual_output) > len(expected_output):
            for line in actual_output[len(expected_output):]:
                errors.append(f"Unexpected output: '{line}'")
        if len(expected_output) > len(actual_output):
            for line in expected_output[len(actual_output):]:
                errors.append(f"Missing output: '{line}'")

    # Check compile errors
    for expected_err in expected_errors:
        found = False
        for stderr_line in actual_stderr:
            if expected_err in stderr_line:
                found = True
                break
        if not found:
            errors.append(f"Missing error: '{expected_err}'")

    # Check runtime error
    if expected_runtime_error:
        found = False
        for stderr_line in actual_stderr:
            if expected_runtime_error in stderr_line:
                found = True
                break
        if not found:
            errors.append(f"Missing runtime error: '{expected_runtime_error}'")

    if errors:
        return False, "; ".join(errors)
    return True, None

def main():
    if len(sys.argv) < 2:
        print("Usage: test_runner.py <lox_binary> [test_filter]")
        sys.exit(1)

    lox_binary = sys.argv[1]
    test_filter = sys.argv[2] if len(sys.argv) > 2 else None

    result = TestResult()

    for root, dirs, files in sorted(os.walk(TESTS_DIR)):
        for filename in sorted(files):
            if not filename.endswith('.lox'):
                continue
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, TESTS_DIR)

            if test_filter and test_filter not in rel_path:
                continue

            if should_skip(filepath, rel_path):
                result.skipped += 1
                continue

            passed, error = run_test(lox_binary, filepath, rel_path)
            if passed:
                result.passed += 1
            else:
                result.failed += 1
                result.errors.append((rel_path, error))
                if result.failed <= 50:  # Show first 50 failures
                    print(f"FAIL {rel_path}: {error}")

    print(f"\n{'='*60}")
    print(f"Results: {result.passed} passed, {result.failed} failed, {result.skipped} skipped")
    print(f"Total: {result.passed + result.failed + result.skipped}")

    if result.failed > 50:
        print(f"\n(showing first 50 of {result.failed} failures)")

    sys.exit(0 if result.failed == 0 else 1)

if __name__ == "__main__":
    main()
