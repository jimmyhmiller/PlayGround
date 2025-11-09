#!/bin/bash
# Quick test script to check specific tests

echo "Testing object trailing comma..."
cargo test test_parse_object_trailing_comma 2>&1 | tail -20

echo ""
echo "Testing pyret match object trailing comma..."
cargo test test_pyret_match_object_trailing_comma 2>&1 | tail -20

echo ""
echo "Testing error trailing comma in args..."
cargo test test_error_trailing_comma_in_args 2>&1 | tail -20

echo ""
echo "Testing error array trailing comma..."
cargo test test_error_array_trailing_comma 2>&1 | tail -20
