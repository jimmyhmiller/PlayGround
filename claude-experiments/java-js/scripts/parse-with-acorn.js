#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const acorn = require('acorn');

const filePath = process.argv[2];
if (!filePath) {
  console.error('Usage: node parse-with-acorn.js <file-path>');
  process.exit(1);
}

try {
  const source = fs.readFileSync(filePath, 'utf-8');

  const ast = acorn.parse(source, {
    ecmaVersion: 'latest',
    sourceType: 'module',
    locations: true
  });

  console.log(JSON.stringify(ast, null, 2));
} catch (e) {
  console.error('Parse error:', e.message);
  process.exit(1);
}
