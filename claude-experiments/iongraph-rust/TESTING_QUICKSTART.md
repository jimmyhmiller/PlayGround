# Testing Quick Start Guide

## 🎯 What Was Created

A comprehensive testing infrastructure with **2742 tests** comparing Rust vs TypeScript output for all ion-examples.

## 📊 Current Status

```
✅ 1726 tests PASSING (63%)
❌ 1016 tests FAILING (37% - layout differences)
⏱️  Test time: ~28 seconds
```

## 🚀 Quick Commands

### Run All Tests
```bash
cargo test --test ion_examples_comprehensive
```

### Run Tests for Specific File
```bash
cargo test --test ion_examples_comprehensive test_simple_add
cargo test --test ion_examples_comprehensive test_mega_complex
cargo test --test ion_examples_comprehensive test_branching
```

### Run Single Test
```bash
cargo test --test ion_examples_comprehensive test_simple_add_func0_pass0 -- --nocapture
```

### Regenerate All Fixtures and Tests
```bash
./regenerate-all-tests.sh
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `generate-all-fixtures.mjs` | Generates TS fixtures from ion-examples |
| `generate-test-suite.mjs` | Generates Rust tests from manifest |
| `regenerate-all-tests.sh` | Master script to regenerate everything |
| `tests/ion_examples_comprehensive.rs` | 2742 comprehensive tests |
| `tests/fixtures/ion-examples/` | 2742 TypeScript SVG fixtures |
| `tests/fixtures/ion-examples/manifest.json` | Fixture metadata |

## 📖 Documentation

- **ION_EXAMPLES_TESTING.md** - Complete testing infrastructure guide
- **TEST_RESULTS_SUMMARY.md** - Detailed test results and analysis
- **PROGRESS.md** - Overall project progress

## 🎯 What's Next

The **layout algorithm** needs fixing to match TypeScript exactly. The tests will guide this work:

1. Run tests to see failures
2. Fix layout calculations
3. Re-run tests to verify
4. Repeat until 100% pass rate

## 💡 Examples

### See What's Failing
```bash
cargo test --test ion_examples_comprehensive test_fibonacci -- --nocapture
```

### Debug Specific Failure
```bash
# Run test
cargo test --test ion_examples_comprehensive test_simple_add_func0_pass17 -- --nocapture

# Generate Rust output
./target/release/iongraph-rust ion-examples/simple-add.json 0 17 /tmp/rust.svg

# Compare
diff tests/fixtures/ion-examples/ts-simple-add-func0-pass17.svg /tmp/rust.svg

# Visual check
open tests/fixtures/ion-examples/ts-simple-add-func0-pass17.svg
open /tmp/rust.svg
```

## 🎉 Success Stories

These files have **100% tests passing**:
- ✅ branching.json (11/11 tests)
- ✅ closure.json (11/11 tests)
- ✅ destructuring.json (7/7 tests)

## 📈 Progress Tracking

As you fix the layout algorithm, track progress:

```bash
# Before fixes
cargo test --test ion_examples_comprehensive | grep "test result:"
# test result: FAILED. 1726 passed; 1016 failed

# After fixes
cargo test --test ion_examples_comprehensive | grep "test result:"
# test result: ok. 2742 passed; 0 failed ← GOAL!
```

## 🛠️ Maintenance

### Add New Examples
1. Add JSON to `ion-examples/`
2. Run `./regenerate-all-tests.sh`
3. Tests automatically created

### Update TypeScript Implementation
```bash
cd ~/Documents/Code/open-source/iongraph2
npm run build
cd /path/to/iongraph-rust
./regenerate-all-tests.sh
```

## 📞 Need Help?

See detailed documentation:
- `ION_EXAMPLES_TESTING.md` - How the infrastructure works
- `TEST_RESULTS_SUMMARY.md` - Current test results and analysis
