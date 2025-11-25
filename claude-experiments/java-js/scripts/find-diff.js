#!/usr/bin/env node

const fs = require('fs');
const acorn = require('acorn');
const { execSync } = require('child_process');

const filePath = process.argv[2];
if (!filePath) {
  console.error('Usage: node find-diff.js <file-path>');
  process.exit(1);
}

try {
  const source = fs.readFileSync(filePath, 'utf-8');

  // Parse with Acorn
  const acornAst = acorn.parse(source, {
    ecmaVersion: 'latest',
    sourceType: 'module',
    locations: true
  });

  // Parse with Java (via DirectoryTester approach)
  const javaOutput = execSync(
    `cd .. && mvn exec:java -Dexec.mainClass="com.jsparser.ASTComparator" -Dexec.args="${filePath}" -q 2>&1`,
    { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
  );

  // Extract Java AST from output
  const javaMatch = javaOutput.match(/=== Java AST ===\n([\s\S]+?)(\n=== |$)/);
  if (!javaMatch) {
    console.error('Could not extract Java AST from output');
    console.log(javaOutput);
    process.exit(1);
  }

  const javaAst = JSON.parse(javaMatch[1]);

  // Compare
  const diffs = findDifferences(acornAst, javaAst, '');

  if (diffs.length === 0) {
    console.log('✓ ASTs match!');
  } else {
    console.log(`✗ Found ${diffs.length} differences:\n`);
    diffs.slice(0, 20).forEach(diff => console.log(diff));
    if (diffs.length > 20) {
      console.log(`\n... and ${diffs.length - 20} more`);
    }
  }
} catch (e) {
  console.error('Error:', e.message);
  process.exit(1);
}

function findDifferences(obj1, obj2, path) {
  const diffs = [];

  if (typeof obj1 !== typeof obj2) {
    diffs.push(`${path}: type mismatch (${typeof obj1} vs ${typeof obj2})`);
    return diffs;
  }

  if (obj1 === null || obj2 === null) {
    if (obj1 !== obj2) {
      diffs.push(`${path}: ${obj1} !== ${obj2}`);
    }
    return diffs;
  }

  if (typeof obj1 !== 'object') {
    if (obj1 !== obj2) {
      diffs.push(`${path}: ${JSON.stringify(obj1)} !== ${JSON.stringify(obj2)}`);
    }
    return diffs;
  }

  if (Array.isArray(obj1) !== Array.isArray(obj2)) {
    diffs.push(`${path}: array mismatch`);
    return diffs;
  }

  if (Array.isArray(obj1)) {
    if (obj1.length !== obj2.length) {
      diffs.push(`${path}: length mismatch (${obj1.length} vs ${obj2.length})`);
    }
    const len = Math.min(obj1.length, obj2.length);
    for (let i = 0; i < len; i++) {
      diffs.push(...findDifferences(obj1[i], obj2[i], `${path}[${i}]`));
    }
    return diffs;
  }

  const keys1 = Object.keys(obj1);
  const keys2 = Object.keys(obj2);
  const allKeys = [...new Set([...keys1, ...keys2])];

  for (const key of allKeys) {
    if (!(key in obj1)) {
      diffs.push(`${path}.${key}: missing in acorn`);
    } else if (!(key in obj2)) {
      diffs.push(`${path}.${key}: missing in java`);
    } else {
      diffs.push(...findDifferences(obj1[key], obj2[key], `${path}.${key}`));
    }
  }

  return diffs;
}
