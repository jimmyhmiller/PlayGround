#!/usr/bin/env node

const fs = require('fs');

// Read JSON failures
const jsonData = JSON.parse(fs.readFileSync('/tmp/all_test262_failures.json', 'utf8'));

// Normalize errors by errorType + tokenType
const normalized = {};

for (const error of jsonData) {
    let key;

    if (error.errorType === 'UnexpectedToken' ||
        error.errorType === 'ExpectedToken' ||
        error.errorType === 'MissingToken') {
        // For ParseExceptions, use errorType + tokenType + expected
        const parts = [error.errorType];

        if (error.tokenType) {
            parts.push(error.tokenType);
        }

        if (error.expected) {
            parts.push(error.expected);
        }

        key = parts.join('|');
    } else {
        // For other exceptions, just use errorType
        key = error.errorType || 'Unknown';
    }

    if (!normalized[key]) {
        normalized[key] = {
            count: 0,
            errorType: error.errorType,
            tokenType: error.tokenType,
            expected: error.expected,
            examples: []
        };
    }

    normalized[key].count++;

    if (normalized[key].examples.length < 3) {
        normalized[key].examples.push({
            file: error.file,
            message: error.message
        });
    }
}

// Sort by count descending
const sorted = Object.entries(normalized)
    .sort((a, b) => b[1].count - a[1].count);

// Print results
console.log('=== NORMALIZED ERROR CATEGORIES ===\n');
console.log(`Total unique error categories: ${sorted.length}`);
console.log(`Total failures: ${jsonData.length}\n`);

console.log('Top 20 error categories:\n');

for (let i = 0; i < Math.min(20, sorted.length); i++) {
    const [key, data] = sorted[i];
    const pct = ((data.count / jsonData.length) * 100).toFixed(2);

    console.log(`${i + 1}. [${data.count} failures, ${pct}%]`);
    console.log(`   ErrorType: ${data.errorType}`);

    if (data.tokenType) {
        console.log(`   TokenType: ${data.tokenType}`);
    }

    if (data.expected) {
        console.log(`   Expected: ${data.expected}`);
    }

    console.log(`   Examples:`);
    for (const ex of data.examples) {
        const fileName = ex.file.split('/').pop();
        console.log(`     - ${fileName}`);
        console.log(`       ${ex.message}`);
    }
    console.log();
}

// Save full categorization to file
const output = {
    summary: {
        totalCategories: sorted.length,
        totalFailures: jsonData.length
    },
    categories: sorted.map(([key, data]) => ({
        key,
        count: data.count,
        percentage: ((data.count / jsonData.length) * 100).toFixed(2),
        errorType: data.errorType,
        tokenType: data.tokenType,
        expected: data.expected,
        examples: data.examples
    }))
};

fs.writeFileSync('/tmp/normalized_errors.json', JSON.stringify(output, null, 2));
console.log('✓ Wrote full categorization to: /tmp/normalized_errors.json');
