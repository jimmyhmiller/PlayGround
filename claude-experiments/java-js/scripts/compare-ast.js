#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Get file path from command line
const cacheFile = process.argv[2];
if (!cacheFile) {
  console.error('Usage: node compare-ast.js <cache-file-path>');
  process.exit(1);
}

// Read expected AST from cache
const expectedAst = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
const metadata = expectedAst._metadata;

if (!metadata || !metadata.sourceFile) {
  console.error('No metadata found in cache file');
  process.exit(1);
}

console.log('Source file:', metadata.sourceFile);
console.log('Source type:', metadata.sourceType);
console.log();

// Parse with our Java parser
const isModule = metadata.sourceType === 'module';
const javaCmd = `mvn exec:java -Dexec.mainClass="com.jsparser.ParserCLI" -Dexec.args="'${metadata.sourceFile}' ${isModule}" -q 2>&1 | grep -v "WARNING:"`;

let actualAst;
try {
  const javaOutput = execSync(javaCmd, { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
  actualAst = JSON.parse(javaOutput);
} catch (err) {
  console.error('Failed to parse with Java parser:');
  console.error(err.message);
  if (err.stdout) {
    console.error(err.stdout.substring(0, 500));
  }
  process.exit(1);
}

// Remove metadata from expected
delete expectedAst._metadata;

// Deep compare and find differences
function findDifferences(expected, actual, path = '') {
  const diffs = [];

  if (typeof expected !== typeof actual) {
    diffs.push({
      path,
      issue: 'Type mismatch',
      expected: typeof expected,
      actual: typeof actual
    });
    return diffs;
  }

  if (Array.isArray(expected)) {
    if (!Array.isArray(actual)) {
      diffs.push({ path, issue: 'Expected array, got object' });
      return diffs;
    }
    if (expected.length !== actual.length) {
      diffs.push({
        path,
        issue: 'Array length mismatch',
        expected: expected.length,
        actual: actual.length
      });
    }
    for (let i = 0; i < Math.min(expected.length, actual.length); i++) {
      diffs.push(...findDifferences(expected[i], actual[i], `${path}[${i}]`));
    }
    return diffs;
  }

  if (typeof expected === 'object' && expected !== null) {
    const expKeys = Object.keys(expected).sort();
    const actKeys = Object.keys(actual || {}).sort();

    const missingKeys = expKeys.filter(k => !actKeys.includes(k));
    const extraKeys = actKeys.filter(k => !expKeys.includes(k));

    if (missingKeys.length > 0) {
      diffs.push({
        path,
        issue: 'Missing keys',
        keys: missingKeys
      });
    }

    if (extraKeys.length > 0) {
      diffs.push({
        path,
        issue: 'Extra keys',
        keys: extraKeys
      });
    }

    for (const key of expKeys) {
      if (actKeys.includes(key)) {
        diffs.push(...findDifferences(
          expected[key],
          actual[key],
          path ? `${path}.${key}` : key
        ));
      }
    }
    return diffs;
  }

  if (expected !== actual) {
    diffs.push({
      path,
      issue: 'Value mismatch',
      expected: JSON.stringify(expected).substring(0, 100),
      actual: JSON.stringify(actual).substring(0, 100)
    });
  }

  return diffs;
}

const differences = findDifferences(expectedAst, actualAst);

if (differences.length === 0) {
  console.log('✓ ASTs match!');
} else {
  console.log(`✗ Found ${differences.length} differences:\n`);

  // Group by issue type
  const grouped = {};
  differences.forEach(diff => {
    const key = diff.issue;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(diff);
  });

  // Show first few of each type
  Object.entries(grouped).forEach(([issue, diffs]) => {
    console.log(`${issue} (${diffs.length} occurrences):`);
    diffs.slice(0, 5).forEach(d => {
      console.log(`  Path: ${d.path || '(root)'}`);
      if (d.keys) console.log(`  Keys: ${d.keys.join(', ')}`);
      if (d.expected !== undefined) console.log(`  Expected: ${d.expected}`);
      if (d.actual !== undefined) console.log(`  Actual: ${d.actual}`);
      console.log();
    });
    if (diffs.length > 5) {
      console.log(`  ... and ${diffs.length - 5} more\n`);
    }
  });
}
