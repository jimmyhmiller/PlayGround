#!/usr/bin/env node

const fs = require('fs');

// Read the normalized JSON data
const data = JSON.parse(fs.readFileSync('/tmp/normalized_errors.json', 'utf8'));

// Read the full JSON failures to get all examples per category
const allFailures = JSON.parse(fs.readFileSync('/tmp/all_test262_failures.json', 'utf8'));

// Group all failures by category key
const categorized = {};

for (const error of allFailures) {
    let key;

    if (error.errorType === 'UnexpectedToken' ||
        error.errorType === 'ExpectedToken' ||
        error.errorType === 'MissingToken') {
        const parts = [error.errorType];

        if (error.tokenType) {
            parts.push(error.tokenType);
        }

        if (error.expected) {
            parts.push(error.expected);
        }

        key = parts.join('|');
    } else {
        key = error.errorType || 'Unknown';
    }

    if (!categorized[key]) {
        categorized[key] = [];
    }

    categorized[key].push(error);
}

// Generate markdown
let output = '# Parse Error Categories - Complete\n\n';
output += `**Total Failures:** ${data.summary.totalFailures}\n`;
output += `**Unique Categories:** ${data.summary.totalCategories}\n\n`;
output += '---\n\n';

// Sort by count
const sorted = data.categories;

for (let i = 0; i < sorted.length; i++) {
    const cat = sorted[i];
    const errors = categorized[cat.key] || [];

    output += `## ${i + 1}. ${cat.key}\n\n`;
    output += `**Count:** ${cat.count} failures (${cat.percentage}%)\n\n`;

    if (cat.tokenType) {
        output += `**Token Type:** ${cat.tokenType}\n`;
    }
    if (cat.expected) {
        output += `**Expected:** ${cat.expected}\n`;
    }

    output += `\n**All ${cat.count} error messages:**\n\n`;

    for (const error of errors) {
        const fileName = error.file.split('/').pop();
        output += `- \`${fileName}\`: ${error.message}\n`;
    }

    output += '\n---\n\n';
}

fs.writeFileSync('ERROR_CATEGORIES_COMPLETE.md', output);
console.log('✓ Wrote complete error categories to: ERROR_CATEGORIES_COMPLETE.md');
console.log(`  Total: ${data.summary.totalFailures} failures across ${data.summary.totalCategories} categories`);
