#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const ORACLE_DIR = "/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/oracle";
const RUST_DIR = "/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/rust";

// Recursively walk the object and collect all numeric values with their paths
function collectNumbers(obj, path = []) {
  const results = [];

  if (Array.isArray(obj)) {
    obj.forEach((item, index) => {
      if (typeof item === 'number') {
        results.push({ path: [...path, index], value: item });
      } else if (typeof item === 'object' && item !== null) {
        results.push(...collectNumbers(item, [...path, index]));
      }
    });
  } else if (typeof obj === 'object' && obj !== null) {
    Object.entries(obj).forEach(([key, value]) => {
      if (typeof value === 'number') {
        results.push({ path: [...path, key], value: value });
      } else if (typeof value === 'object' && value !== null) {
        results.push(...collectNumbers(value, [...path, key]));
      }
    });
  }

  return results;
}

function pathToString(pathArray) {
  return pathArray.map(p => typeof p === 'number' ? `[${p}]` : `.${p}`).join('').replace(/^\./, '');
}

function analyzeDifferences(filename) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Analyzing: ${filename}`);
  console.log('='.repeat(60));

  const oracleData = JSON.parse(fs.readFileSync(path.join(ORACLE_DIR, filename), 'utf8'));
  const rustData = JSON.parse(fs.readFileSync(path.join(RUST_DIR, filename), 'utf8'));

  const oracleNumbers = collectNumbers(oracleData);
  const rustNumbers = collectNumbers(rustData);

  console.log(`Oracle numbers found: ${oracleNumbers.length}`);
  console.log(`Rust numbers found: ${rustNumbers.length}`);

  // Create maps for comparison
  const oracleMap = new Map(oracleNumbers.map(n => [pathToString(n.path), n.value]));
  const rustMap = new Map(rustNumbers.map(n => [pathToString(n.path), n.value]));

  const differences = {
    offsetBy5: [],
    notOffsetBy5: [],
    onlyInOracle: [],
    onlyInRust: []
  };

  // Compare values at same paths
  oracleMap.forEach((oracleValue, path) => {
    if (rustMap.has(path)) {
      const rustValue = rustMap.get(path);
      const diff = rustValue - oracleValue;

      if (diff === 5) {
        differences.offsetBy5.push({ path, oracle: oracleValue, rust: rustValue, diff });
      } else if (diff !== 0) {
        differences.notOffsetBy5.push({ path, oracle: oracleValue, rust: rustValue, diff });
      }
    } else {
      differences.onlyInOracle.push({ path, value: oracleValue });
    }
  });

  rustMap.forEach((rustValue, path) => {
    if (!oracleMap.has(path)) {
      differences.onlyInRust.push({ path, value: rustValue });
    }
  });

  console.log(`\nDifferences offset by exactly 5: ${differences.offsetBy5.length}`);
  console.log(`Differences NOT offset by 5: ${differences.notOffsetBy5.length}`);
  console.log(`Only in Oracle: ${differences.onlyInOracle.length}`);
  console.log(`Only in Rust: ${differences.onlyInRust.length}`);

  if (differences.notOffsetBy5.length > 0) {
    console.log(`\n${'─'.repeat(60)}`);
    console.log('Sample differences NOT offset by 5:');
    console.log('─'.repeat(60));

    const samples = differences.notOffsetBy5.slice(0, 20);
    samples.forEach(({ path, oracle, rust, diff }) => {
      console.log(`Path: ${path}`);
      console.log(`  Oracle: ${oracle}`);
      console.log(`  Rust:   ${rust}`);
      console.log(`  Diff:   ${diff} (${diff > 0 ? '+' : ''}${diff})`);
      console.log();
    });

    if (differences.notOffsetBy5.length > 20) {
      console.log(`... and ${differences.notOffsetBy5.length - 20} more\n`);
    }
  }

  // Show distribution of differences
  const diffCounts = {};
  differences.notOffsetBy5.forEach(({ diff }) => {
    diffCounts[diff] = (diffCounts[diff] || 0) + 1;
  });

  if (Object.keys(diffCounts).length > 0) {
    console.log('─'.repeat(60));
    console.log('Distribution of non-5 differences:');
    console.log('─'.repeat(60));
    Object.entries(diffCounts)
      .sort(([a], [b]) => parseInt(a) - parseInt(b))
      .forEach(([diff, count]) => {
        console.log(`  Offset ${diff}: ${count} occurrences`);
      });
  }

  return differences;
}

// Analyze all JSON files
const files = ['authors.json', 'cohorts.json', 'dirs.json', 'domains.json', 'exts.json', 'survival.json'];

console.log('Finding differences that are NOT offset by exactly 5...\n');

const allDifferences = {};

files.forEach(file => {
  try {
    allDifferences[file] = analyzeDifferences(file);
  } catch (error) {
    console.error(`Error processing ${file}:`, error.message);
  }
});

// Summary
console.log('\n' + '='.repeat(60));
console.log('SUMMARY');
console.log('='.repeat(60));

files.forEach(file => {
  const diffs = allDifferences[file];
  if (diffs) {
    console.log(`${file}:`);
    console.log(`  Offset by 5: ${diffs.offsetBy5.length}`);
    console.log(`  NOT offset by 5: ${diffs.notOffsetBy5.length}`);
    console.log(`  Only in Oracle: ${diffs.onlyInOracle.length}`);
    console.log(`  Only in Rust: ${diffs.onlyInRust.length}`);
  }
});
