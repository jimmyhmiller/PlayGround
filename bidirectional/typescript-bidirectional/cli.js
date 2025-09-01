#!/usr/bin/env node

import { readFileSync } from 'fs';
import { processProgram, formatType } from './index.js';

// ANSI color codes
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m', 
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

function colorize(text, color) {
  return `${colors[color]}${text}${colors.reset}`;
}

function printUsage() {
  console.log(`${colors.bold}TypeScript Bidirectional Type Checker${colors.reset}`);
  console.log('');
  console.log('Usage: node cli.js <filename>');
  console.log('');
  console.log('Examples:');
  console.log('  node cli.js good-example.ts    # Should type check successfully');
  console.log('  node cli.js bad-example.ts     # Should show type errors');
}

function printResult(filename, result, success) {
  const status = success ? colorize('✓ SUCCESS', 'green') : colorize('✗ FAILED', 'red');
  console.log(`${colors.bold}${colors.cyan}Type checking: ${filename}${colors.reset}`);
  console.log(`Status: ${status}`);
  console.log('');
  
  if (success) {
    console.log(colorize('Functions found:', 'blue'));
    const functions = Object.entries(result.context).filter(([name, type]) => type.kind === 'function');
    if (functions.length === 0) {
      console.log('  (no functions defined)');
    } else {
      functions.forEach(([name, type]) => {
        console.log(`  ${colorize(name, 'cyan')}: ${colorize(formatType(type), 'yellow')}`);
      });
    }
    console.log('');
    
    console.log(colorize('Variables found:', 'blue'));
    const variables = Object.entries(result.context).filter(([name, type]) => type.kind !== 'function');
    if (variables.length === 0) {
      console.log('  (no variables defined)');
    } else {
      variables.forEach(([name, type]) => {
        console.log(`  ${colorize(name, 'cyan')}: ${colorize(formatType(type), 'yellow')}`);
      });
    }
    console.log('');
    
    if (result.results && result.results.length > 0) {
      console.log(colorize('Expression results:', 'blue'));
      result.results.forEach((res, i) => {
        if (res.kind === 'expression') {
          console.log(`  Expression ${i + 1}: ${colorize(formatType(res.type), 'yellow')}`);
        }
      });
    }
  }
}

function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    printUsage();
    process.exit(1);
  }
  
  if (args[0] === '--help' || args[0] === '-h') {
    printUsage();
    process.exit(0);
  }
  
  const filename = args[0];
  
  try {
    const code = readFileSync(filename, 'utf8');
    console.log('');
    
    try {
      const result = processProgram(code);
      printResult(filename, result, true);
      console.log(colorize('🎉 All types check out!', 'green'));
      process.exit(0);
    } catch (typeError) {
      console.log(`${colors.bold}${colors.cyan}Type checking: ${filename}${colors.reset}`);
      console.log(`Status: ${colorize('✗ FAILED', 'red')}`);
      console.log('');
      console.log(colorize('Type Error:', 'red'));
      console.log(`  ${typeError.message}`);
      console.log('');
      console.log(colorize('💡 Fix the type error and try again!', 'yellow'));
      process.exit(1);
    }
  } catch (fileError) {
    console.error(colorize(`Error reading file: ${fileError.message}`, 'red'));
    process.exit(1);
  }
}

main();