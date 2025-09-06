#!/usr/bin/env python3
"""
Snapshot Comparison Tool for IonGraph Rust Port

This tool compares the Rust port snapshot tests with the original TypeScript
implementation to ensure we've ported them correctly.
"""

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Paths
RUST_DIR = Path("/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/iongraph-rust")
TYPESCRIPT_DIR = Path("/Users/jimmyhmiller/Documents/Code/open-source/iongraph")

RUST_SNAPSHOTS_DIR = RUST_DIR / "src" / "snapshots"
TS_SNAPSHOTS_DIR = TYPESCRIPT_DIR / "src" / "test" / "__snapshots__"

@dataclass
class SnapshotTest:
    name: str
    category: str
    test_case: str
    content: Any
    file_path: Path

@dataclass
class ComparisonResult:
    rust_test: Optional[SnapshotTest]
    typescript_test: Optional[SnapshotTest]
    status: str  # "MATCHED", "MISSING_IN_RUST", "MISSING_IN_TS", "DIFFERENT"
    details: str

def parse_typescript_snapshots() -> List[SnapshotTest]:
    """Parse TypeScript snapshot files to extract test data."""
    snapshots = []
    
    ts_snap_file = TS_SNAPSHOTS_DIR / "Graph.snapshots.test.ts.snap"
    if not ts_snap_file.exists():
        print(f"❌ TypeScript snapshots not found at: {ts_snap_file}")
        return snapshots
    
    with open(ts_snap_file, 'r') as f:
        content = f.read()
    
    # Parse exports pattern: exports[`Graph Snapshots > Category > Test Name`] = `content`;
    pattern = r"exports\[`Graph Snapshots > (.*?) > (.*?) 1`\] = `(.*?)`;(?=\n\nexports|\n\n$|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    for category, test_name, test_content in matches:
        # Try to parse as JSON first
        try:
            parsed_content = json.loads(test_content.strip())
        except json.JSONDecodeError:
            # If not JSON, keep as string
            parsed_content = test_content.strip()
        
        snapshots.append(SnapshotTest(
            name=f"{category.lower().replace(' ', '_')}_{test_name.lower().replace(' ', '_').replace('-', '_')}",
            category=category,
            test_case=test_name,
            content=parsed_content,
            file_path=ts_snap_file
        ))
    
    return snapshots

def parse_rust_snapshots() -> List[SnapshotTest]:
    """Parse Rust snapshot files to extract test data."""
    snapshots = []
    
    if not RUST_SNAPSHOTS_DIR.exists():
        print(f"❌ Rust snapshots directory not found at: {RUST_SNAPSHOTS_DIR}")
        return snapshots
    
    for snap_file in RUST_SNAPSHOTS_DIR.glob("*.snap"):
        if snap_file.name.endswith(".snap.new"):
            continue
            
        with open(snap_file, 'r') as f:
            content = f.read()
        
        # Parse snapshot content (simplified YAML parsing)
        try:
            # Remove the header (---\nsource: ...\nexpression: ...\n---)
            yaml_content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
            # For now, keep as string - full YAML parsing would need external lib
            parsed_content = yaml_content.strip()
        except Exception:
            parsed_content = content
        
        # Extract test info from filename
        # e.g., iongraph_rust__graph__tests__block_ir_simple.snap
        parts = snap_file.stem.split('__')
        if len(parts) >= 4:
            test_name = parts[-1]  # e.g., "block_ir_simple"
            category = "block_ir" if "block_ir" in test_name else "layout"
        else:
            test_name = snap_file.stem
            category = "unknown"
        
        snapshots.append(SnapshotTest(
            name=test_name,
            category=category,
            test_case=test_name,
            content=parsed_content,
            file_path=snap_file
        ))
    
    return snapshots

def categorize_typescript_tests(ts_tests: List[SnapshotTest]) -> Dict[str, List[SnapshotTest]]:
    """Categorize TypeScript tests by type."""
    categories = {
        "Block Intermediate Representation Snapshots": [],
        "Layout Node Structure Snapshots": [],
        "Instruction Highlighting Snapshots": [],
        "Navigation State Snapshots": [],
        "Pan and Zoom State Snapshots": []
    }
    
    for test in ts_tests:
        if test.category in categories:
            categories[test.category].append(test)
    
    return categories

def find_matching_rust_test(rust_tests: List[SnapshotTest], ts_test: SnapshotTest) -> Optional[SnapshotTest]:
    """Find a matching Rust test for a TypeScript test."""
    # Mapping of TypeScript test patterns to Rust test names
    mappings = {
        # Block IR tests - NEW MAPPINGS
        "should create expected block IR for simple pass": ["block_ir_simple", "test_block_ir_snapshot_simple"],
        "should create expected block IR for loop pass": ["block_ir_loop", "test_block_ir_snapshot_loop"], 
        "should create expected block IR for complex control flow": ["block_ir_complex", "test_block_ir_snapshot_complex"],
        
        # Layout tests
        "should create expected layout structure for simple pass": ["layout_simple_pass"],
        "should create expected layout structure for loop pass": ["layout_loop_pass"],
        "should create expected layout structure for complex control flow": ["layout_complex_pass"],
        
        # Metrics tests (our custom addition)
        "layout_metrics_snapshot_simple": ["layout_metrics_snapshot_simple"],
        "layout_metrics_snapshot_loop": ["layout_metrics_snapshot_loop"],
        "layout_metrics_snapshot_complex": ["layout_metrics_snapshot_complex"],
    }
    
    for ts_pattern, rust_names in mappings.items():
        if ts_pattern in ts_test.test_case:
            for rust_test in rust_tests:
                for rust_name in rust_names:
                    if rust_name in rust_test.name:
                        return rust_test
    
    return None

def compare_content(rust_content: Any, ts_content: Any) -> tuple[bool, str]:
    """Compare the content of two tests."""
    if type(rust_content) != type(ts_content):
        return False, f"Type mismatch: Rust={type(rust_content)}, TS={type(ts_content)}"
    
    if isinstance(ts_content, list) and isinstance(rust_content, list):
        if len(ts_content) != len(rust_content):
            return False, f"Array length mismatch: Rust={len(rust_content)}, TS={len(ts_content)}"
        
        # Compare each element
        for i, (rust_item, ts_item) in enumerate(zip(rust_content, ts_content)):
            if isinstance(ts_item, dict) and isinstance(rust_item, dict):
                # Compare dict keys
                ts_keys = set(ts_item.keys())
                rust_keys = set(rust_item.keys())
                
                if ts_keys != rust_keys:
                    missing_in_rust = ts_keys - rust_keys
                    extra_in_rust = rust_keys - ts_keys
                    details = []
                    if missing_in_rust:
                        details.append(f"Missing in Rust: {missing_in_rust}")
                    if extra_in_rust:
                        details.append(f"Extra in Rust: {extra_in_rust}")
                    return False, f"Item {i} key mismatch: {', '.join(details)}"
    
    return True, "Content structure matches"

def generate_report(comparisons: List[ComparisonResult]) -> str:
    """Generate a detailed comparison report."""
    report = []
    report.append("# 📊 IonGraph Snapshot Comparison Report")
    report.append("=" * 60)
    report.append("")
    
    # Summary stats
    matched = sum(1 for c in comparisons if c.status == "MATCHED")
    missing_rust = sum(1 for c in comparisons if c.status == "MISSING_IN_RUST")
    missing_ts = sum(1 for c in comparisons if c.status == "MISSING_IN_TS")
    different = sum(1 for c in comparisons if c.status == "DIFFERENT")
    
    report.append(f"## 📈 Summary Statistics")
    report.append(f"- **Total Tests**: {len(comparisons)}")
    report.append(f"- **✅ Matched**: {matched}")
    report.append(f"- **❌ Missing in Rust**: {missing_rust}")
    report.append(f"- **❓ Missing in TypeScript**: {missing_ts}")
    report.append(f"- **🔄 Different**: {different}")
    report.append("")
    
    # Coverage percentage
    coverage = (matched / len(comparisons) * 100) if comparisons else 0
    report.append(f"**🎯 Port Coverage: {coverage:.1f}%**")
    report.append("")
    
    # Detailed results by status
    for status in ["MATCHED", "MISSING_IN_RUST", "DIFFERENT", "MISSING_IN_TS"]:
        status_tests = [c for c in comparisons if c.status == status]
        if not status_tests:
            continue
            
        status_emoji = {"MATCHED": "✅", "MISSING_IN_RUST": "❌", "DIFFERENT": "🔄", "MISSING_IN_TS": "❓"}
        report.append(f"## {status_emoji[status]} {status.replace('_', ' ').title()}")
        report.append("")
        
        for comp in status_tests:
            if comp.typescript_test:
                report.append(f"### {comp.typescript_test.category} > {comp.typescript_test.test_case}")
                report.append(f"**TypeScript**: `{comp.typescript_test.file_path.name}`")
                if comp.rust_test:
                    report.append(f"**Rust**: `{comp.rust_test.file_path.name}`")
                else:
                    report.append(f"**Rust**: *Not implemented*")
                report.append(f"**Details**: {comp.details}")
            elif comp.rust_test:
                report.append(f"### {comp.rust_test.name}")
                report.append(f"**Rust**: `{comp.rust_test.file_path.name}`")
                report.append(f"**TypeScript**: *Not found*")
                report.append(f"**Details**: {comp.details}")
            report.append("")
    
    return "\n".join(report)

def main():
    print("🔍 IonGraph Snapshot Comparison Tool")
    print("=" * 50)
    print()
    
    # Parse snapshots
    print("📖 Parsing TypeScript snapshots...")
    ts_tests = parse_typescript_snapshots()
    print(f"   Found {len(ts_tests)} TypeScript tests")
    
    print("📖 Parsing Rust snapshots...")
    rust_tests = parse_rust_snapshots()
    print(f"   Found {len(rust_tests)} Rust tests")
    print()
    
    # Compare tests
    print("🔍 Comparing snapshots...")
    comparisons = []
    
    # Check each TypeScript test
    for ts_test in ts_tests:
        rust_test = find_matching_rust_test(rust_tests, ts_test)
        
        if rust_test:
            # Compare content
            matches, details = compare_content(rust_test.content, ts_test.content)
            status = "MATCHED" if matches else "DIFFERENT"
        else:
            status = "MISSING_IN_RUST"
            details = "No corresponding Rust test found"
        
        comparisons.append(ComparisonResult(
            rust_test=rust_test,
            typescript_test=ts_test,
            status=status,
            details=details
        ))
    
    # Check for Rust tests not in TypeScript
    for rust_test in rust_tests:
        # Skip if already covered
        if any(c.rust_test and c.rust_test.name == rust_test.name for c in comparisons):
            continue
            
        comparisons.append(ComparisonResult(
            rust_test=rust_test,
            typescript_test=None,
            status="MISSING_IN_TS",
            details="Rust-specific test (not in original TypeScript)"
        ))
    
    # Generate and save report
    report = generate_report(comparisons)
    
    report_path = RUST_DIR / "SNAPSHOT_COMPARISON_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📊 Report saved to: {report_path}")
    print()
    print("📋 Summary:")
    
    matched = sum(1 for c in comparisons if c.status == "MATCHED")
    missing_rust = sum(1 for c in comparisons if c.status == "MISSING_IN_RUST")
    coverage = (matched / len([c for c in comparisons if c.typescript_test]) * 100) if comparisons else 0
    
    print(f"   🎯 Port Coverage: {coverage:.1f}%")
    print(f"   ✅ Matched: {matched}")
    print(f"   ❌ Missing in Rust: {missing_rust}")
    
    if missing_rust > 0:
        print()
        print("🚨 Missing Tests in Rust:")
        for comp in comparisons:
            if comp.status == "MISSING_IN_RUST" and comp.typescript_test:
                print(f"   - {comp.typescript_test.category} > {comp.typescript_test.test_case}")

if __name__ == "__main__":
    main()