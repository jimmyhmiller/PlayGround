#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const ORACLE_DIR = "/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/oracle";
const RUST_DIR = "/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/rust";

const oracleData = JSON.parse(fs.readFileSync(path.join(ORACLE_DIR, 'authors.json'), 'utf8'));
const rustData = JSON.parse(fs.readFileSync(path.join(RUST_DIR, 'authors.json'), 'utf8'));

console.log('STRUCTURE OF authors.json:');
console.log('==========================\n');

console.log('Labels (author names):');
console.log(oracleData.labels);

console.log('\n\nFirst 10 timestamps:');
console.log(oracleData.ts.slice(0, 10));

console.log('\n\nY array structure:');
console.log(`Number of series: ${oracleData.y.length}`);
console.log(`First series length: ${oracleData.y[0].length}`);

console.log('\n\nComparing Y[0] values (first 70 elements):');
console.log('Index | Oracle  | Rust    | Diff');
console.log('------|---------|---------|------');

for (let i = 60; i < 70; i++) {
  const oracleVal = oracleData.y[0][i];
  const rustVal = rustData.y[0][i];
  const diff = rustVal - oracleVal;
  console.log(`${i.toString().padStart(5)} | ${oracleVal.toString().padStart(7)} | ${rustVal.toString().padStart(7)} | ${diff > 0 ? '+' : ''}${diff}`);
}

console.log('\n\nFull Y array record at index 65:');
console.log('Oracle y[0][65]:', oracleData.y[0][65]);
console.log('Rust   y[0][65]:', rustData.y[0][65]);
console.log('Difference:', rustData.y[0][65] - oracleData.y[0][65]);

console.log('\n\nWhat does this number represent?');
console.log('These are likely indices into the commits array or timestamp positions.');
console.log('Oracle value 512652 vs Rust value 512657 = +5 offset');
