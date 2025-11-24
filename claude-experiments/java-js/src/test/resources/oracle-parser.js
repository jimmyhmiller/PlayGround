#!/usr/bin/env node

/**
 * Oracle parser using acorn CLI to generate ESTree-compatible AST
 */

const { execSync } = require('child_process');
const fs = require('fs');

if (process.argv.length < 3) {
    console.error('Usage: node oracle-parser.js <source-file>');
    process.exit(1);
}

const sourceFile = process.argv[2];

// Check if source has module syntax
function shouldUseModuleMode(filePath) {
    const source = fs.readFileSync(filePath, 'utf-8');
    // Simple heuristic: check for import/export keywords
    return /\b(import|export)\b/.test(source);
}

try {
    const moduleFlag = shouldUseModuleMode(sourceFile) ? '--module' : '';
    // Use npx to run the local acorn installation
    // Use ecma2022 to support private class fields (#) and other ES2022 features
    const command = `npx acorn --ecma2022 --locations ${moduleFlag} "${sourceFile}"`;

    const output = execSync(command, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    console.log(output);
} catch (error) {
    console.error('Parse error:', error.message);
    process.exit(1);
}
