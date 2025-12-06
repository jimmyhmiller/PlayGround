#!/usr/bin/env python3
"""
Analyze and compare cross-language JavaScript parser benchmarks
Creates a comprehensive comparison table from Java, JavaScript, and Rust results
"""

import json
import re
import sys
from typing import Dict, List

def parse_java_results(json_file: str) -> Dict[str, Dict[str, float]]:
    """Parse JMH JSON results"""
    results = {}

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        for benchmark in data:
            name = benchmark['benchmark']
            # Extract parser name and test case
            # e.g., "ComparativeParserBenchmark.ourParser_SmallFunction"
            match = re.search(r'\.(\w+)_(.+)$', name)
            if match:
                parser = match.group(1)
                test_case = match.group(2)

                # Convert camelCase to Title Case with spaces
                test_case = re.sub(r'([A-Z])', r' \1', test_case).strip()

                score = benchmark['primaryMetric']['score']

                if test_case not in results:
                    results[test_case] = {}

                # Map parser names to display names
                parser_map = {
                    'ourParser': 'Our Parser (Java)',
                    'rhinoParser': 'Rhino (Java)',
                    'nashornParser': 'Nashorn (Java)',
                    'graalJS': 'GraalJS (Java)'
                }

                display_name = parser_map.get(parser, parser)
                results[test_case][display_name] = score

    except FileNotFoundError:
        print(f"Warning: {json_file} not found", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {json_file}", file=sys.stderr)

    return results

def parse_javascript_results(txt_file: str) -> Dict[str, Dict[str, float]]:
    """Parse JavaScript benchmark text results"""
    results = {}

    try:
        with open(txt_file, 'r') as f:
            content = f.read()

        # Find JSON results section
        json_match = re.search(r'JSON Results.*?\n(\{.*?\})\s*$', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))

            for test_case, parsers in data.items():
                results[test_case] = {}
                for parser_data in parsers:
                    parser_name = f"{parser_data['parser']} (JS)"
                    results[test_case][parser_name] = parser_data['avgMicros']

    except FileNotFoundError:
        print(f"Warning: {txt_file} not found", file=sys.stderr)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not parse {txt_file}: {e}", file=sys.stderr)

    return results

def parse_rust_results(txt_file: str) -> Dict[str, Dict[str, float]]:
    """Parse Rust benchmark text results"""
    results = {}
    current_test = None

    try:
        with open(txt_file, 'r') as f:
            for line in f:
                # Find test case name
                test_match = re.match(r'Benchmark: (.+)', line)
                if test_match:
                    current_test = test_match.group(1)
                    results[current_test] = {}
                    continue

                # Find parser results
                if current_test:
                    result_match = re.search(r'([🥇🥈🥉])\s+(.+?)\s+\|\s+([\d.]+)', line)
                    if result_match:
                        parser_name = result_match.group(2).strip()
                        avg_micros = float(result_match.group(3))
                        results[current_test][parser_name] = avg_micros

    except FileNotFoundError:
        print(f"Warning: {txt_file} not found", file=sys.stderr)

    return results

def merge_results(*result_dicts) -> Dict[str, Dict[str, float]]:
    """Merge multiple result dictionaries"""
    merged = {}

    for result_dict in result_dicts:
        for test_case, parsers in result_dict.items():
            if test_case not in merged:
                merged[test_case] = {}
            merged[test_case].update(parsers)

    return merged

def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print a comprehensive comparison table"""

    print("\n" + "═" * 100)
    print("CROSS-LANGUAGE JAVASCRIPT PARSER BENCHMARK COMPARISON")
    print("═" * 100)
    print("\nAll times in microseconds (µs) - Lower is better")
    print("─" * 100)

    for test_case in sorted(results.keys()):
        parsers = results[test_case]

        if not parsers:
            continue

        print(f"\n{test_case}:")
        print("─" * 100)

        # Sort by performance
        sorted_parsers = sorted(parsers.items(), key=lambda x: x[1])

        fastest = sorted_parsers[0][1]

        print(f"{'Rank':<6} {'Parser':<30} {'Time (µs)':>15} {'vs Fastest':>15} {'Relative Speed'}")
        print("─" * 100)

        medals = ['🥇', '🥈', '🥉']

        for i, (parser, time) in enumerate(sorted_parsers):
            rank = medals[i] if i < 3 else f"{i+1}."
            ratio = time / fastest

            # Color code by language
            if 'Rust' in parser:
                color_code = '\033[0;32m'  # Green
            elif 'JS)' in parser or 'JavaScript' in parser:
                color_code = '\033[0;33m'  # Yellow
            else:
                color_code = '\033[0;34m'  # Blue (Java/JVM)

            reset_code = '\033[0m'

            print(f"{rank:<6} {color_code}{parser:<30}{reset_code} {time:>15.3f} {ratio:>15.2f}x {'█' * min(int(ratio * 5), 50)}")

    print("\n" + "═" * 100)
    print("Legend: 🟢 Rust  🟡 JavaScript  🔵 Java/JVM")
    print("═" * 100)

def print_summary(results: Dict[str, Dict[str, float]]):
    """Print a summary of winners by category"""

    print("\n" + "═" * 100)
    print("SUMMARY: FASTEST PARSER BY TEST CASE")
    print("═" * 100)

    for test_case in sorted(results.keys()):
        parsers = results[test_case]
        if parsers:
            fastest = min(parsers.items(), key=lambda x: x[1])
            print(f"\n{test_case}:")
            print(f"  🥇 {fastest[0]}: {fastest[1]:.3f} µs")

    print("\n" + "═" * 100)
    print("OVERALL ANALYSIS")
    print("═" * 100)

    # Calculate average rankings
    rankings = {}
    for test_case, parsers in results.items():
        sorted_parsers = sorted(parsers.items(), key=lambda x: x[1])
        for i, (parser, _) in enumerate(sorted_parsers):
            if parser not in rankings:
                rankings[parser] = []
            rankings[parser].append(i + 1)

    avg_rankings = {p: sum(ranks) / len(ranks) for p, ranks in rankings.items()}

    print("\nAverage Ranking (lower is better):")
    print("─" * 100)

    for parser, avg_rank in sorted(avg_rankings.items(), key=lambda x: x[1]):
        print(f"  {avg_rank:>5.2f}  {parser}")

    print("═" * 100)

def main():
    """Main analysis function"""

    # Parse results from all languages
    java_results = parse_java_results('benchmark-results/java-results.json')
    js_results = parse_javascript_results('benchmark-results/javascript-results.txt')
    rust_results = parse_rust_results('benchmark-results/rust-results.txt')

    # Merge all results
    all_results = merge_results(java_results, js_results, rust_results)

    if not all_results:
        print("Error: No benchmark results found. Run ./run-all-benchmarks.sh first.", file=sys.stderr)
        sys.exit(1)

    # Print comparison table
    print_comparison_table(all_results)

    # Print summary
    print_summary(all_results)

if __name__ == '__main__':
    main()
