#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const ORACLE_DIR = "/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/oracle";
const RUST_DIR = "/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/rust";

function showRecords(filename) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`File: ${filename}`);
  console.log('='.repeat(70));

  const oracleData = JSON.parse(fs.readFileSync(path.join(ORACLE_DIR, filename), 'utf8'));
  const rustData = JSON.parse(fs.readFileSync(path.join(RUST_DIR, filename), 'utf8'));

  console.log('\nORACLE VERSION:');
  console.log('-'.repeat(70));
  console.log(JSON.stringify(oracleData, null, 2).substring(0, 1500));
  console.log('\n... (truncated)\n');

  console.log('\nRUST VERSION:');
  console.log('-'.repeat(70));
  console.log(JSON.stringify(rustData, null, 2).substring(0, 1500));
  console.log('\n... (truncated)\n');
}

// Show a few different files
const files = ['authors.json', 'survival.json', 'exts.json'];

files.forEach(file => {
  try {
    showRecords(file);
  } catch (error) {
    console.error(`Error processing ${file}:`, error.message);
  }
});
