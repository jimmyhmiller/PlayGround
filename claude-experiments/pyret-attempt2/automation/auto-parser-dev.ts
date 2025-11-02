#!/usr/bin/env tsx

import Anthropic from '@anthropic-ai/sdk';
import { execSync } from 'child_process';
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Configuration
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const MAX_ITERATIONS = parseInt(process.env.MAX_ITERATIONS || '30');
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || 'claude-sonnet-4-5-20250929';
const PROJECT_ROOT = join(process.cwd(), '..');
const DRY_RUN = process.argv.includes('--dry-run') || process.argv.includes('-d');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

// Types
interface TestResults {
  total: number;
  passing: number;
  ignored: number;
  failed: number;
  deleted: boolean;
}

interface IterationResult {
  iteration: number;
  before: TestResults;
  after: TestResults;
  success: boolean;
  message: string;
}

// Validate setup (skip in dry-run mode)
if (!DRY_RUN && !ANTHROPIC_API_KEY) {
  console.error(`${colors.red}Error: ANTHROPIC_API_KEY not found in environment${colors.reset}`);
  console.error('Please copy .env.example to .env and add your API key');
  process.exit(1);
}

const client = DRY_RUN ? null : new Anthropic({ apiKey: ANTHROPIC_API_KEY });

// Helper functions
function log(message: string, color: string = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function runCommand(command: string, cwd: string = PROJECT_ROOT): string {
  try {
    return execSync(command, { cwd, encoding: 'utf-8', stdio: 'pipe' });
  } catch (error: any) {
    return error.stdout || error.stderr || '';
  }
}

function parseTestResults(output: string): TestResults {
  // Parse cargo test output - need to get the LAST "test result" line
  // since there are multiple test files (unit tests, parser_tests, comparison_tests)
  // We only care about comparison_tests which is the last one
  // Example: "test result: ok. 73 passed; 0 failed; 8 ignored; 0 measured; 0 filtered out"

  // Get all test result lines
  const lines = output.split('\n');
  const testResultLines = lines.filter(line => line.includes('test result:'));

  if (testResultLines.length === 0) {
    log(`Warning: Could not find any test result lines`, colors.yellow);
    return { total: 0, passing: 0, ignored: 0, failed: 0, deleted: false };
  }

  // The last test result line is from comparison_tests (they run last and take longest)
  const lastResultLine = testResultLines[testResultLines.length - 1];

  const resultMatch = lastResultLine.match(/test result:.*?(\d+) passed.*?(\d+) failed.*?(\d+) ignored/);

  if (!resultMatch) {
    log(`Warning: Could not parse test results from: ${lastResultLine}`, colors.yellow);
    return { total: 0, passing: 0, ignored: 0, failed: 0, deleted: false };
  }

  const passing = parseInt(resultMatch[1]);
  const failed = parseInt(resultMatch[2]);
  const ignored = parseInt(resultMatch[3]);
  const total = passing + failed + ignored;

  return { total, passing, ignored, failed, deleted: false };
}

function runTests(): TestResults {
  log('\n📊 Running tests...', colors.cyan);
  const output = runCommand('cargo test --test comparison_tests 2>&1');

  if (DRY_RUN) {
    // In dry-run mode, show the test result lines for debugging
    const lines = output.split('\n').filter(line => line.includes('test result:'));
    log(`   [DRY-RUN] Found ${lines.length} test result lines:`, colors.yellow);
    lines.forEach((line, idx) => {
      log(`   [${idx + 1}] ${line}`, colors.yellow);
    });
  }

  const results = parseTestResults(output);

  log(`   Total: ${results.total} | Passing: ${colors.green}${results.passing}${colors.reset} | Ignored: ${colors.yellow}${results.ignored}${colors.reset} | Failed: ${colors.red}${results.failed}${colors.reset}`);

  return results;
}

function verifyNoTestDeletion(before: TestResults, after: TestResults): boolean {
  // Check if total number of tests decreased
  if (after.total < before.total) {
    const deleted = before.total - after.total;
    log(`\n❌ ERROR: ${deleted} test(s) were deleted!`, colors.red);
    log(`   Before: ${before.total} tests | After: ${after.total} tests`, colors.red);
    return false;
  }
  return true;
}

function checkProgress(before: TestResults, after: TestResults): boolean {
  // Progress means: more tests passing OR new ignored tests added (test generation)
  const passingIncreased = after.passing > before.passing;
  const newTestsAdded = after.total > before.total;

  return passingIncreased || newTestsAdded;
}

function gitCommit(message: string = 'Changes') {
  log('\n📝 Committing changes...', colors.cyan);

  if (DRY_RUN) {
    log(`   [DRY-RUN] Would commit with message: "${message}"`, colors.yellow);
    return;
  }

  try {
    runCommand('git add .');
    runCommand(`git commit -m "${message}"`);
    log('   ✓ Committed successfully', colors.green);
  } catch (error) {
    log('   ⚠ No changes to commit', colors.yellow);
  }
}

async function askClaude(prompt: string, conversationHistory: any[] = []): Promise<string> {
  log('\n🤖 Asking Claude...', colors.magenta);

  if (DRY_RUN) {
    log(`   [DRY-RUN] Would send prompt (first 200 chars):`, colors.yellow);
    log(`   ${prompt.substring(0, 200)}...`, colors.yellow);
    return '[DRY-RUN] Simulated Claude response';
  }

  try {
    const response = await client!.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: 8000,
      messages: [
        ...conversationHistory,
        { role: 'user', content: prompt }
      ],
    });

    const textContent = response.content.find(block => block.type === 'text');
    return textContent ? (textContent as any).text : '';
  } catch (error: any) {
    log(`   ❌ Error calling Claude API: ${error.message}`, colors.red);
    throw error;
  }
}

async function handleTestDeletion(iteration: number): Promise<boolean> {
  log('\n🔄 Attempting to restore deleted tests...', colors.yellow);

  const prompt = `CRITICAL: Tests have been deleted. You must restore them immediately.
Never delete tests - if a test is failing, mark it as #[ignore] instead of removing it.
Please restore all deleted tests and ensure they are either passing or marked as ignored.`;

  try {
    await askClaude(prompt);

    // Re-run tests to check if they're restored
    const results = runTests();

    if (results.total >= iteration) { // Rough check
      log('   ✓ Tests restored successfully', colors.green);
      return true;
    } else {
      log('   ❌ Tests were not restored', colors.red);
      return false;
    }
  } catch (error) {
    log('   ❌ Failed to restore tests', colors.red);
    return false;
  }
}

async function handleRegression(before: TestResults): Promise<boolean> {
  log('\n🔄 Attempting to fix regression...', colors.yellow);

  const prompt = `CRITICAL: Tests that were passing are now failing.
Before: ${before.passing} passing
After: Check current test results

Please fix the failing tests without removing any tests. If you cannot fix them, mark them as #[ignore] instead.`;

  try {
    await askClaude(prompt);

    // Re-run tests to check if regression is fixed
    const results = runTests();

    if (results.passing >= before.passing) {
      log('   ✓ Regression fixed', colors.green);
      return true;
    } else {
      log('   ❌ Regression not fixed', colors.red);
      return false;
    }
  } catch (error) {
    log('   ❌ Failed to fix regression', colors.red);
    return false;
  }
}

async function mainLoop() {
  log(`${colors.bright}╔════════════════════════════════════════════╗${colors.reset}`);
  log(`${colors.bright}║   Pyret Parser Automation Script          ║${colors.reset}`);
  log(`${colors.bright}║   Max Iterations: ${MAX_ITERATIONS.toString().padEnd(28)}║${colors.reset}`);
  if (DRY_RUN) {
    log(`${colors.bright}║   ${colors.yellow}DRY-RUN MODE${colors.reset}${colors.bright} (no API calls/commits)     ║${colors.reset}`);
  }
  log(`${colors.bright}╚════════════════════════════════════════════╝${colors.reset}`);

  const iterations: IterationResult[] = [];
  let conversationHistory: any[] = [];

  // In dry-run mode, only do 1-2 iterations to test the flow
  const iterationLimit = DRY_RUN ? 2 : MAX_ITERATIONS;

  for (let i = 1; i <= iterationLimit; i++) {
    log(`\n${colors.bright}${'='.repeat(50)}${colors.reset}`);
    log(`${colors.bright}Iteration ${i}/${iterationLimit}${colors.reset}`);
    log(`${colors.bright}${'='.repeat(50)}${colors.reset}`);

    // Step 1: Run tests and get baseline
    const beforeResults = runTests();

    // Step 2: Determine what to ask Claude
    let prompt: string;

    if (beforeResults.passing === beforeResults.total && beforeResults.ignored === 0) {
      // All tests passing, no ignored tests - time to generate new tests
      log('\n🎉 All tests passing! Asking Claude to generate new tests...', colors.green);
      prompt = `You are a testing expert, we have a working parser, but we need to find areas that aren't complete with some comparison tests to actual pyret. Be sure to use real code. Once these tests are in place and failing, please update them to be ignored and update the documentation to tell people what work to do next`;
    } else {
      // There are ignored tests - work on implementing features
      log(`\n🔧 Working on parser (${beforeResults.ignored} ignored tests remaining)...`, colors.blue);
      prompt = `Please work on the parser to make more ignored tests pass. Current status:
- Total tests: ${beforeResults.total}
- Passing: ${beforeResults.passing}
- Ignored: ${beforeResults.ignored}
- Failed: ${beforeResults.failed}

Focus on implementing the features needed for the ignored tests.`;
    }

    // Step 3: Ask Claude to work on the parser
    try {
      const claudeResponse = await askClaude(prompt, conversationHistory);
      conversationHistory.push(
        { role: 'user', content: prompt },
        { role: 'assistant', content: claudeResponse }
      );

      log(`\n📝 Claude's response (first 200 chars):`, colors.cyan);
      log(`   ${claudeResponse.substring(0, 200)}...`);
    } catch (error) {
      log(`\n❌ Failed to get response from Claude`, colors.red);
      break;
    }

    // Step 4: Run tests again and verify
    const afterResults = runTests();

    // Step 5: Check for test deletion
    if (!verifyNoTestDeletion(beforeResults, afterResults)) {
      log('\n⚠ Attempting to restore deleted tests...', colors.yellow);
      const restored = await handleTestDeletion(i);

      if (!restored) {
        log('\n❌ ABORTED: Tests were deleted and could not be restored', colors.red);
        iterations.push({
          iteration: i,
          before: beforeResults,
          after: afterResults,
          success: false,
          message: 'Tests deleted and not restored'
        });
        break;
      }
    }

    // Step 6: Check for regression
    if (afterResults.passing < beforeResults.passing) {
      log('\n⚠ Regression detected - fewer tests passing', colors.yellow);
      const fixed = await handleRegression(beforeResults);

      if (!fixed) {
        log('\n❌ ABORTED: Regression could not be fixed', colors.red);
        iterations.push({
          iteration: i,
          before: beforeResults,
          after: afterResults,
          success: false,
          message: 'Regression not fixed'
        });
        break;
      }
    }

    // Step 7: Update documentation
    log('\n📚 Updating documentation...', colors.cyan);
    try {
      await askClaude('Please update the documentation for what to work on next', conversationHistory);
    } catch (error) {
      log('   ⚠ Failed to update documentation', colors.yellow);
    }

    // Step 8: Commit changes
    gitCommit('Changes');

    // Record iteration results
    const madeProgress = checkProgress(beforeResults, afterResults);
    iterations.push({
      iteration: i,
      before: beforeResults,
      after: afterResults,
      success: madeProgress,
      message: madeProgress ? 'Progress made' : 'No progress'
    });

    // Step 9: Check if we're done
    if (afterResults.passing === afterResults.total && afterResults.ignored === 0 && afterResults.total > beforeResults.total) {
      log('\n🎊 All tests passing (including newly generated tests)!', colors.green);
      break;
    }

    // Small delay between iterations
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  // Print summary
  printSummary(iterations);
}

function printSummary(iterations: IterationResult[]) {
  log(`\n${colors.bright}╔════════════════════════════════════════════╗${colors.reset}`);
  log(`${colors.bright}║           SUMMARY REPORT                   ║${colors.reset}`);
  log(`${colors.bright}╚════════════════════════════════════════════╝${colors.reset}`);

  log(`\nTotal iterations: ${iterations.length}`);

  if (iterations.length > 0) {
    const first = iterations[0].before;
    const last = iterations[iterations.length - 1].after;

    log(`\nInitial state:`);
    log(`  Passing: ${first.passing}/${first.total}`);
    log(`  Ignored: ${first.ignored}`);

    log(`\nFinal state:`);
    log(`  Passing: ${last.passing}/${last.total}`);
    log(`  Ignored: ${last.ignored}`);

    log(`\nProgress:`);
    log(`  New passing tests: ${colors.green}+${last.passing - first.passing}${colors.reset}`);
    log(`  Change in ignored: ${last.ignored - first.ignored}`);

    if (last.passing === last.total && last.ignored === 0) {
      log(`\n${colors.green}${colors.bright}✓ ALL TESTS PASSING!${colors.reset}`);
    } else {
      log(`\n${colors.yellow}Still work to do: ${last.ignored} ignored tests remaining${colors.reset}`);
    }
  }

  log(`\n${colors.bright}${'='.repeat(50)}${colors.reset}\n`);
}

// Run the main loop
mainLoop().catch(error => {
  log(`\n❌ Fatal error: ${error.message}`, colors.red);
  console.error(error);
  process.exit(1);
});
