# Pyret Parser Development Automation

This directory contains an automated workflow script that uses the **Claude Agent SDK** to iteratively develop the Pyret parser until all tests pass.

## 🎯 What It Does

The automation script implements your development loop:

1. **Run Tests** - Execute `cargo test --test comparison_tests`
2. **Analyze Results** - Count passing, ignored, and failed tests
3. **Ask Claude** - Send appropriate prompt based on test state
   - If ignored tests exist: "Please work on the parser"
   - If all tests pass: "Act as testing expert and create new comparison tests"
4. **Verify Changes** - Ensure tests aren't deleted and no regressions occur
5. **Update Docs** - Ask Claude to update documentation
6. **Commit** - Git commit with message "Changes"
7. **Repeat** - Loop until all tests pass or max iterations reached

## 🚀 Quick Start

### Installation

```bash
cd automation
npm install
```

### Running

```bash
npm start
```

Or with tsx directly:
```bash
npx tsx auto-parser-dev.ts
```

## ⚙️ Configuration Options

Set environment variables to customize:

```bash
# Optional: Maximum iterations before stopping (default: 30)
MAX_ITERATIONS=30 npm start
```

## 📊 How It Works

### Main Loop

Each iteration:

1. **Test Analysis**
   - Runs `cargo test --test comparison_tests`
   - Parses output: `X passed; Y failed; Z ignored`
   - Tracks test count to detect deletions

2. **Decision Logic**
   - **Case A: Ignored tests exist**
     - Prompt: "Please work on the parser to make more ignored tests pass"
     - Claude implements features needed for ignored tests

   - **Case B: All tests passing, no ignored**
     - Prompt: "You are a testing expert, we have a working parser, but we need to find areas that aren't complete with some comparison tests to actual pyret. Be sure to use real code. Once these tests are in place and failing, please update them to be ignored and update the documentation to tell people what work to do next"
     - Claude generates new tests for uncovered features

3. **Ask Claude to Work**
   - Sends prompt via Agent SDK
   - Claude reads/writes files and makes changes

4. **Compilation Check** ✨ NEW
   - Runs `cargo check` to verify code compiles
   - If compilation fails:
     - Extracts error messages
     - Sends errors to Claude: "CRITICAL: The code does not compile..."
     - Retries compilation after fix
     - Aborts if still broken after retry
   - This prevents "tests deleted" false positives from non-compiling code

5. **Safety Checks**
   - **Test Deletion Check**: If total tests decreased, abort (with one retry to restore)
   - **Regression Check**: If passing tests decreased, attempt to fix (with one retry)

6. **Documentation Update**
   - Asks Claude: "Please update the documentation for what to work on next"

7. **Git Commit**
   - Commits all changes with message: "Changes"

### Termination Conditions

The script stops when:

- ✅ **Success**: All tests passing (including newly generated ones)
- ⏸️ **Max Iterations**: Reached configured limit (default: 30)
- ❌ **Compilation Error**: Code doesn't compile and couldn't be fixed after retry
- ❌ **Test Deletion**: Tests deleted and not restored after retry
- ❌ **Regression**: Passing tests decreased and not fixed after retry

## 📈 Progress Tracking

The script tracks progress by:

1. **Test Count Trends**
   - Initial: `X passing / Y total (Z ignored)`
   - After iteration: `X' passing / Y' total (Z' ignored)`
   - Progress = more passing OR new tests added

2. **Iteration Results**
   - Each iteration records before/after state
   - Summary report shows overall progress

3. **Colored Output**
   - 🟢 Green: Progress made
   - 🟡 Yellow: Warnings
   - 🔴 Red: Errors/failures
   - 🔵 Blue: Information

## 🛡️ Safety Features

### Test Deletion Prevention

If the total number of tests decreases:

1. **Detect**: Compare total before/after
2. **Alert**: Log warning in red
3. **Retry**: Ask Claude to restore deleted tests
4. **Verify**: Re-run tests to confirm restoration
5. **Abort**: If not restored, stop execution

### Regression Handling

If passing tests decrease:

1. **Detect**: Compare passing count before/after
2. **Alert**: Log regression warning
3. **Retry**: Ask Claude to fix without deleting tests
4. **Verify**: Re-run tests to confirm fix
5. **Abort**: If not fixed, stop execution

### Iteration Limit

Default maximum of 30 iterations prevents:
- Infinite loops
- Excessive API costs
- Runaway execution

## 📝 Example Output

```
╔════════════════════════════════════════════╗
║   Pyret Parser Automation Script          ║
║   Max Iterations: 30                      ║
╚════════════════════════════════════════════╝

==================================================
Iteration 1/30
==================================================

📊 Running tests...
   Total: 81 | Passing: 72 | Ignored: 9 | Failed: 0

🔧 Working on parser (9 ignored tests remaining)...

🤖 Asking Claude...

📝 Claude's response (first 200 chars):
   I'll work on implementing method fields in objects, which is the highest priority feature. Let me start by examining the current object parsing code and the test that's waiting...

📊 Running tests...
   Total: 81 | Passing: 73 | Ignored: 8 | Failed: 0

📚 Updating documentation...

📝 Committing changes...
   ✓ Committed successfully

==================================================
Iteration 2/30
==================================================

...
```

## 🐛 Troubleshooting

### Dry Run Mode

Test the script without making actual API calls:
```bash
npm start -- --dry-run
```

### "Tests were deleted"

The script detected test removal. This triggers:
1. Automatic retry to restore tests
2. If retry fails, script aborts
3. Manual intervention needed

### "Regression not fixed"

Tests that were passing started failing:
1. Script attempts one auto-fix
2. If fix fails, script aborts
3. Check git log: `git log --oneline`
4. Revert if needed: `git reset --hard HEAD~1`

### Limiting Iterations

To reduce the number of iterations:
```bash
MAX_ITERATIONS=10 npm start
```

## 📚 Architecture

```
auto-parser-dev.ts
├── Configuration Loading
├── Helper Functions
│   ├── runCommand() - Execute shell commands
│   ├── parseTestResults() - Parse cargo test output (handles multiple test files)
│   ├── runTests() - Run comparison tests
│   ├── verifyNoTestDeletion() - Check test count
│   ├── checkProgress() - Detect improvements
│   ├── gitCommit() - Commit changes
│   └── askClaude() - Call Claude via Agent SDK query()
├── Error Handlers
│   ├── handleTestDeletion() - Restore deleted tests
│   └── handleRegression() - Fix failing tests
├── Main Loop
│   └── mainLoop() - Orchestrate iterations
└── Summary
    └── printSummary() - Display final report
```

## 🔧 Technical Details

### Claude Agent SDK Integration

The script uses the `@anthropic-ai/claude-agent-sdk` package with the following configuration:

```typescript
query({
  prompt: "Your task",
  options: {
    cwd: PROJECT_ROOT,              // Work in project directory
    permissionMode: 'acceptEdits',  // Auto-approve file edits
    maxTurns: 1                     // Single turn per request
  }
})
```

This gives Claude full access to:
- Read/write files in the project
- Run commands (via Bash tool)
- Access project documentation (CLAUDE.md)
- Make edits without requiring manual approval

## 🔄 Integration with Existing Workflow

This automation replaces your manual workflow:

**Before (Manual):**
```bash
# Step 1: Ask Claude to work on parser
Please work on the parser
> some tests are no longer ignored and pass

# Step 2: Ask Claude to update docs
Please update the documentation for what to work on next

# Step 3: Commit
git add . && git commit -m "Changes"

# Step 4: Repeat
```

**After (Automated):**
```bash
cd automation
npm start
# Script does all steps automatically
```

## 📖 Related Files

- **`../tests/comparison_tests.rs`** - Comparison tests that verify against official Pyret parser
- **`../CLAUDE.md`** - Project documentation that Claude reads
- **`../NEXT_STEPS.md`** - Next priority tasks (updated by automation)
- **`../compare_parsers.sh`** - Script used by tests to compare parsers

## 🤝 Contributing

To modify the automation:

1. Edit `auto-parser-dev.ts`
2. Test changes: `npm start`
3. Update this README if behavior changes

## 📄 License

MIT (same as parent project)

---

**Last Updated**: 2025-11-02
**Status**: Ready to use
**Current Tests**: 72/81 passing (88.9%), 9 ignored
