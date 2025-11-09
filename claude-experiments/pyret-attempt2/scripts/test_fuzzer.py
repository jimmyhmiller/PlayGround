#!/usr/bin/env python3
"""
Property-based / fuzzing test generator for Pyret parser.

Generates random valid Pyret expressions and tests them against
the official Pyret parser to ensure our implementation matches.

Usage:
    python3 test_fuzzer.py [--count N] [--max-depth D] [--seed S]
"""

import random
import subprocess
import sys
import argparse
from typing import List, Tuple

# Configuration
OPERATORS = ["+", "-", "*", "/", "<", ">", "<=", ">=", "==", "<>", "and", "or", "^", "is"]
IDENTIFIERS = ["x", "y", "z", "a", "b", "c", "foo", "bar", "obj", "result", "value"]
NUMBERS = ["0", "1", "2", "3", "5", "10", "42", "100"]
STRINGS = ['""', '"hello"', '"test"', '"abc"']
BOOLEANS = ["true", "false"]


class ExprGenerator:
    """Generates random valid Pyret expressions"""

    def __init__(self, max_depth: int = 5, seed: int = None):
        self.max_depth = max_depth
        if seed is not None:
            random.seed(seed)

    def generate(self) -> str:
        """Generate a random expression"""
        return self._expr(depth=0)

    def _expr(self, depth: int) -> str:
        """Generate an expression at given depth"""
        if depth >= self.max_depth:
            # At max depth, only generate primitives
            return self._primitive()

        # Choose what kind of expression to generate
        choice = random.randint(1, 100)

        if choice <= 20:
            return self._primitive()
        elif choice <= 35:
            return self._binop(depth)
        elif choice <= 45:
            return self._paren(depth)
        elif choice <= 55:
            return self._call(depth)
        elif choice <= 65:
            return self._dot(depth)
        elif choice <= 75:
            return self._array(depth)
        elif choice <= 85:
            return self._chained_call(depth)
        elif choice <= 92:
            return self._chained_dot(depth)
        else:
            return self._mixed_postfix(depth)

    def _primitive(self) -> str:
        """Generate a primitive expression"""
        choice = random.randint(1, 4)
        if choice == 1:
            return random.choice(NUMBERS)
        elif choice == 2:
            return random.choice(IDENTIFIERS)
        elif choice == 3:
            return random.choice(STRINGS)
        else:
            return random.choice(BOOLEANS)

    def _binop(self, depth: int) -> str:
        """Generate binary operation"""
        left = self._expr(depth + 1)
        op = random.choice(OPERATORS)
        right = self._expr(depth + 1)
        return f"{left} {op} {right}"

    def _paren(self, depth: int) -> str:
        """Generate parenthesized expression"""
        inner = self._expr(depth + 1)
        # Occasionally nest parens
        if random.random() < 0.2 and depth < self.max_depth - 1:
            return f"(({inner}))"
        return f"({inner})"

    def _call(self, depth: int) -> str:
        """Generate function call"""
        func = random.choice(IDENTIFIERS)
        num_args = random.choices([0, 1, 2, 3], weights=[10, 40, 30, 20])[0]

        args = [self._expr(depth + 1) for _ in range(num_args)]
        args_str = ", ".join(args)

        # Whitespace sensitivity: sometimes add space before paren
        if random.random() < 0.1 and num_args > 0:
            return f"{func} ({args_str})"
        return f"{func}({args_str})"

    def _dot(self, depth: int) -> str:
        """Generate dot access"""
        obj = self._expr(depth + 1) if random.random() < 0.5 else random.choice(IDENTIFIERS)
        field = random.choice(IDENTIFIERS)
        return f"{obj}.{field}"

    def _array(self, depth: int) -> str:
        """Generate array expression"""
        if random.random() < 0.1:
            return "[]"

        num_elements = random.choices([1, 2, 3, 4, 5], weights=[20, 30, 25, 15, 10])[0]
        elements = [self._expr(depth + 1) for _ in range(num_elements)]
        return f"[{', '.join(elements)}]"

    def _chained_call(self, depth: int) -> str:
        """Generate chained function calls"""
        num_calls = random.randint(2, 4)
        expr = random.choice(IDENTIFIERS)

        for _ in range(num_calls):
            num_args = random.choices([0, 1, 2], weights=[40, 40, 20])[0]
            args = [self._expr(depth + 2) for _ in range(num_args)]
            expr += f"({', '.join(args)})"

        return expr

    def _chained_dot(self, depth: int) -> str:
        """Generate chained dot access"""
        num_dots = random.randint(2, 5)
        expr = random.choice(IDENTIFIERS)

        for _ in range(num_dots):
            expr += f".{random.choice(IDENTIFIERS)}"

        return expr

    def _mixed_postfix(self, depth: int) -> str:
        """Generate mixed dot and call operations"""
        expr = random.choice(IDENTIFIERS)
        num_ops = random.randint(2, 5)

        for _ in range(num_ops):
            if random.random() < 0.5:
                # Add dot
                expr += f".{random.choice(IDENTIFIERS)}"
            else:
                # Add call
                num_args = random.choices([0, 1, 2], weights=[50, 30, 20])[0]
                args = [self._primitive() for _ in range(num_args)]
                expr += f"({', '.join(args)})"

        return expr


def test_expression(expr: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Test an expression against the comparison script.
    Returns (success, error_message)
    """
    try:
        result = subprocess.run(
            ["bash", "compare_parsers.sh", expr],
            capture_output=True,
            timeout=5,
            cwd="/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2"
        )

        success = result.returncode == 0

        if verbose and not success:
            error = result.stdout.decode() + result.stderr.decode()
            return (False, error)

        return (success, "")

    except subprocess.TimeoutExpired:
        return (False, "Timeout")
    except Exception as e:
        return (False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Fuzz test the Pyret parser")
    parser.add_argument("--count", type=int, default=100, help="Number of expressions to generate")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum expression depth")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--save-failures", type=str, help="Save failures to file")

    args = parser.parse_args()

    print("=" * 60)
    print("Pyret Parser Fuzzer")
    print("=" * 60)
    print(f"Generating {args.count} random expressions (max depth: {args.max_depth})")
    if args.seed is not None:
        print(f"Using seed: {args.seed}")
    print()

    generator = ExprGenerator(max_depth=args.max_depth, seed=args.seed)

    passed = 0
    failed = 0
    failures = []

    for i in range(args.count):
        expr = generator.generate()

        if args.verbose:
            print(f"\nTest {i + 1}/{args.count}")
            print(f"Expression: {expr}")

        success, error = test_expression(expr, verbose=args.verbose)

        if success:
            passed += 1
            if args.verbose:
                print("✅ PASS")
            else:
                print(".", end="", flush=True)
        else:
            failed += 1
            failures.append((expr, error))
            if args.verbose:
                print("❌ FAIL")
                if error:
                    print(f"Error: {error[:200]}")
            else:
                print("F", end="", flush=True)

    print("\n")
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total: {args.count}")
    print(f"Passed: {passed} ({100 * passed // args.count}%)")
    print(f"Failed: {failed} ({100 * failed // args.count}%)")
    print()

    if failures:
        print("Failed expressions:")
        for expr, error in failures[:20]:  # Show first 20
            print(f"  ❌ {expr}")
            if args.verbose and error:
                print(f"     {error[:100]}")

        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

        if args.save_failures:
            with open(args.save_failures, 'w') as f:
                for expr, error in failures:
                    f.write(f"{expr}\n")
            print(f"\nFailures saved to: {args.save_failures}")

    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
