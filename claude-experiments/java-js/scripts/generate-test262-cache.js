#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { Worker } = require('worker_threads');
const os = require('os');

const TEST262_DIR = path.join(__dirname, '..', 'test-oracles', 'test262', 'test');
const CACHE_DIR = path.join(__dirname, '..', 'test-oracles', 'test262-cache');
const NUM_WORKERS = os.cpus().length;

// Ensure cache directory exists
if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
}

function shouldSkip(filePath) {
    // Skip staging directory - experimental/non-standard tests
    if (filePath.includes('/staging/')) {
        return true;
    }

    // Skip annexB directory - legacy/deprecated/web-compat edge cases that Acorn rejects
    if (filePath.includes('/annexB/')) {
        return true;
    }

    // Skip decorator tests - Acorn doesn't support decorators
    if (filePath.includes('/decorator/')) {
        return true;
    }

    // Skip files with 'accessor' in the name - Acorn doesn't support auto-accessors
    if (filePath.includes('accessor')) {
        return true;
    }

    // Skip negative parse tests - they are expected to fail parsing
    return shouldSkipNegativeTest(filePath);
}

// Collect all files first
function collectFiles(dir, files = []) {
    const entries = fs.readdirSync(dir);

    for (const entry of entries) {
        const fullPath = path.join(dir, entry);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
            collectFiles(fullPath, files);
        } else if (entry.endsWith('.js') && !shouldSkip(fullPath)) {
            files.push(fullPath);
        }
    }

    return files;
}

// Parse Test262 frontmatter
function parseFrontmatter(source) {
    const match = source.match(/\/\*---\n([\s\S]*?)\n---\*\//);
    if (!match) return null;

    const yaml = match[1];
    const result = { isModule: false, shouldSkip: false };

    // Check for module flag - inline format: flags: [module, async]
    const flagsInline = yaml.match(/^flags:\s*\[(.*?)\]/m);
    if (flagsInline) {
        const flags = flagsInline[1].split(',').map(f => f.trim());
        result.isModule = flags.includes('module');
    }

    // Check for module flag - multiline format
    const flagsMultiline = yaml.match(/^flags:\s*\n((?:\s+-\s+.*\n?)+)/m);
    if (flagsMultiline) {
        result.isModule = /^\s+-\s+module\s*$/m.test(flagsMultiline[1]);
    }

    // Check for negative parse tests - should skip these
    // negative:
    //   phase: parse
    if (/^negative:/m.test(yaml)) {
        const negativeSection = yaml.match(/^negative:\s*\n((?:\s+.*\n?)+)/m);
        if (negativeSection && /^\s+phase:\s*parse\s*$/m.test(negativeSection[1])) {
            result.shouldSkip = true;
        }
    }

    return result;
}

// Check if source has module flag in Test262 frontmatter
function shouldUseModuleMode(filePath) {
    try {
        // Files ending in _FIXTURE.js are module fixtures - always parse as modules
        if (filePath.endsWith('_FIXTURE.js')) {
            return true;
        }

        const source = fs.readFileSync(filePath, 'utf-8');
        const frontmatter = parseFrontmatter(source);
        return frontmatter ? frontmatter.isModule : false;
    } catch (e) {
        return false;
    }
}

// Check if this is a negative parse test (should skip)
function shouldSkipNegativeTest(filePath) {
    try {
        const source = fs.readFileSync(filePath, 'utf-8');
        const frontmatter = parseFrontmatter(source);
        return frontmatter ? frontmatter.shouldSkip : false;
    } catch (e) {
        return false;
    }
}

// Process a single file
function processFile(filePath) {
    try {
        const moduleFlag = shouldUseModuleMode(filePath) ? '--module' : '';
        // Use ecma2025 to support all modern JavaScript features
        const command = `acorn --ecma2025 --locations ${moduleFlag} "${filePath}"`;

        const output = execSync(command, {
            encoding: 'utf-8',
            maxBuffer: 10 * 1024 * 1024,
            stdio: ['pipe', 'pipe', 'pipe']
        });

        const relativePath = path.relative(TEST262_DIR, filePath);
        const cacheFilePath = path.join(CACHE_DIR, relativePath + '.json');
        const cacheFileDir = path.dirname(cacheFilePath);

        if (!fs.existsSync(cacheFileDir)) {
            fs.mkdirSync(cacheFileDir, { recursive: true });
        }

        fs.writeFileSync(cacheFilePath, output.trim(), 'utf8');
        return { success: true };
    } catch (e) {
        const errorMsg = e.stderr ? e.stderr.toString().split('\n')[0] : e.message;
        return { success: false, error: errorMsg };
    }
}

console.log('Generating test262 acorn cache...');
console.log('Scanning:', TEST262_DIR);
console.log('Cache directory:', CACHE_DIR);
console.log(`Using ${NUM_WORKERS} workers`);

// Collect all files
const allFiles = collectFiles(TEST262_DIR);
console.log(`Found ${allFiles.length} files to process`);

// Process files in batches
const BATCH_SIZE = 100;
let processed = 0;
let successful = 0;
let failed = 0;
const failedFiles = [];

for (let i = 0; i < allFiles.length; i += BATCH_SIZE) {
    const batch = allFiles.slice(i, i + BATCH_SIZE);

    for (const file of batch) {
        const result = processFile(file);
        processed++;

        if (result.success) {
            successful++;
        } else {
            failed++;
            if (failedFiles.length < 20) {
                failedFiles.push(`${file}: ${result.error}`);
            }
        }
    }

    console.log(`Progress: ${processed}/${allFiles.length} files processed, ${successful} cached, ${failed} failed`);
}

console.log('\n=== Cache Generation Results ===');
console.log(`Total files: ${allFiles.length}`);
console.log(`Successfully cached: ${successful} (${(successful * 100 / allFiles.length).toFixed(2)}%)`);
console.log(`Failed to parse: ${failed} (${(failed * 100 / allFiles.length).toFixed(2)}%)`);

if (failedFiles.length > 0) {
    console.log('\nFirst 20 failed files:');
    failedFiles.forEach(f => console.log('  ' + f));
}

console.log('\nCache generation complete!');
