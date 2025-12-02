#!/bin/bash
# Master script to regenerate all TypeScript fixtures and Rust tests

set -e

echo "🔄 IonGraph Test Suite Regeneration"
echo "===================================="
echo ""

echo "📁 Step 1: Generating TypeScript fixtures from ion-examples..."
node generate-all-fixtures.mjs

echo ""
echo "📝 Step 2: Generating Rust test suite from manifest..."
node generate-test-suite.mjs

echo ""
echo "✅ Done!"
echo ""
echo "Summary:"
echo "  - TypeScript fixtures: tests/fixtures/ion-examples/*.svg"
echo "  - Manifest: tests/fixtures/ion-examples/manifest.json"
echo "  - Rust tests: tests/ion_examples_comprehensive.rs"
echo ""
echo "To run the tests:"
echo "  cargo test --test ion_examples_comprehensive -- --nocapture"
echo ""
echo "To run specific tests:"
echo "  cargo test --test ion_examples_comprehensive test_mega_complex"
echo "  cargo test --test ion_examples_comprehensive test_simple_add"
