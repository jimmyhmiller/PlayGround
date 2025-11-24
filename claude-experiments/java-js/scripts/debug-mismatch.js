#!/usr/bin/env node

const fs = require('fs');
const acorn = require('acorn');

if (process.argv.length < 3) {
    console.error('Usage: node debug-mismatch.js <test-file-path>');
    process.exit(1);
}

const file = process.argv[2];
const source = fs.readFileSync(file, 'utf-8');

// Parse with Acorn (fresh)
const acornAst = acorn.parse(source, {
    ecmaVersion: 2025,
    locations: true,
    sourceType: 'script'
});

// Read from cache
const cacheFile = file.replace('test262/test/', 'test262-cache/') + '.json';
const cachedAst = JSON.parse(fs.readFileSync(cacheFile, 'utf-8'));

// Remove metadata
delete cachedAst._metadata;

// Serialize both
const acornJson = JSON.stringify(acornAst, (k, v) => typeof v === 'bigint' ? null : v, 2);
const cacheJson = JSON.stringify(cachedAst, (k, v) => typeof v === 'bigint' ? null : v, 2);

console.log('File:', file.split('/').slice(-1)[0]);
console.log('Match:', acornJson === cacheJson);
console.log('Acorn JSON length:', acornJson.length);
console.log('Cache JSON length:', cacheJson.length);

if (acornJson !== cacheJson) {
    const acornLines = acornJson.split('\n');
    const cacheLines = cacheJson.split('\n');

    console.log('\nFirst 10 differences:');
    let diffCount = 0;
    for (let i = 0; i < Math.min(acornLines.length, cacheLines.length) && diffCount < 10; i++) {
        if (acornLines[i] !== cacheLines[i]) {
            console.log(`\nLine ${i + 1}:`);
            console.log('  Acorn:', acornLines[i].substring(0, 100));
            console.log('  Cache:', cacheLines[i].substring(0, 100));
            diffCount++;
        }
    }
}
