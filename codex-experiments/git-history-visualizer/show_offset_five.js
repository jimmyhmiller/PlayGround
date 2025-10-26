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

function showExamplesOffByFive(filename, numExamples = 10) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`File: ${filename}`);
  console.log('='.repeat(70));

  const oracleData = JSON.parse(fs.readFileSync(path.join(ORACLE_DIR, filename), 'utf8'));
  const rustData = JSON.parse(fs.readFileSync(path.join(RUST_DIR, filename), 'utf8'));

  const oracleNumbers = collectNumbers(oracleData);
  const rustNumbers = collectNumbers(rustData);

  // Create maps for comparison
  const oracleMap = new Map(oracleNumbers.map(n => [pathToString(n.path), n.value]));
  const rustMap = new Map(rustNumbers.map(n => [pathToString(n.path), n.value]));

  const offsetByFive = [];

  // Compare values at same paths
  oracleMap.forEach((oracleValue, path) => {
    if (rustMap.has(path)) {
      const rustValue = rustMap.get(path);
      const diff = rustValue - oracleValue;

      if (diff === 5) {
        offsetByFive.push({ path, oracle: oracleValue, rust: rustValue });
      }
    }
  });

  console.log(`\nTotal values offset by exactly +5: ${offsetByFive.length}`);
  console.log(`\nShowing first ${Math.min(numExamples, offsetByFive.length)} examples:\n`);

  offsetByFive.slice(0, numExamples).forEach(({ path, oracle, rust }, index) => {
    console.log(`${index + 1}. Path: ${path}`);
    console.log(`   Oracle: ${oracle}`);
    console.log(`   Rust:   ${rust}`);
    console.log(`   Diff:   +5\n`);
  });

  return offsetByFive.length;
}

// Show examples from all files
const files = ['authors.json', 'cohorts.json', 'dirs.json', 'domains.json', 'exts.json', 'survival.json'];

console.log('\nExamples of values offset by exactly +5 (Rust - Oracle):\n');

files.forEach(file => {
  try {
    showExamplesOffByFive(file, 5);
  } catch (error) {
    console.error(`Error processing ${file}:`, error.message);
  }
});
