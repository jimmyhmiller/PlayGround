#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const acorn = require('acorn');

const CURATED_DIR = path.join(__dirname, '..', 'test-oracles', 'curated');
const CACHE_DIR = path.join(__dirname, '..', 'test-oracles', 'curated-cache');
const FORCE_REGENERATE = process.argv.includes('--force');

// Ensure cache directory exists
if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
}

// Calculate SHA-256 hash of file content
function getFileHash(filePath) {
    const content = fs.readFileSync(filePath);
    return crypto.createHash('sha256').update(content).digest('hex');
}

// Check if cache is valid for a file
function isCacheValid(sourceFile, cacheFile) {
    if (!fs.existsSync(cacheFile)) {
        return false;
    }

    try {
        const cacheContent = fs.readFileSync(cacheFile, 'utf-8');
        const cache = JSON.parse(cacheContent);

        if (!cache._metadata || !cache._metadata.sourceHash) {
            return false;
        }

        const currentHash = getFileHash(sourceFile);
        return cache._metadata.sourceHash === currentHash;
    } catch (e) {
        return false;
    }
}

// Collect all JS files recursively
function collectFiles(dir, files = []) {
    const entries = fs.readdirSync(dir);

    for (const entry of entries) {
        const fullPath = path.join(dir, entry);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
            collectFiles(fullPath, files);
        } else if (entry.endsWith('.js') || entry.endsWith('.mjs')) {
            files.push(fullPath);
        }
    }

    return files;
}

// Serialize AST to JSON (same as test262 cache)
function serializeAST(ast) {
    return JSON.stringify(ast, function(_, value) {
        return typeof value === "bigint" ? null : value;
    }, 2);
}

// Process a single file
function processFile(filePath) {
    const relativePath = path.relative(CURATED_DIR, filePath);
    const cacheFilePath = path.join(CACHE_DIR, relativePath + '.json');

    // Check if cache is valid
    if (!FORCE_REGENERATE && isCacheValid(filePath, cacheFilePath)) {
        return { success: true, cached: true };
    }

    try {
        const source = fs.readFileSync(filePath, 'utf-8');

        // Determine sourceType - .mjs files are always modules
        const isModule = filePath.endsWith('.mjs');

        const options = {
            ecmaVersion: 2025,
            locations: true,
            sourceType: isModule ? 'module' : 'script'
        };

        const ast = acorn.parse(source, options);

        const cacheFileDir = path.dirname(cacheFilePath);
        if (!fs.existsSync(cacheFileDir)) {
            fs.mkdirSync(cacheFileDir, { recursive: true });
        }

        // Add metadata
        ast._metadata = {
            sourceHash: getFileHash(filePath),
            generatedAt: new Date().toISOString(),
            acornVersion: acorn.version,
            sourceFile: relativePath
        };

        const jsonOutput = serializeAST(ast);
        fs.writeFileSync(cacheFilePath, jsonOutput, 'utf8');
        return { success: true, cached: false };
    } catch (e) {
        return { success: false, error: e.message };
    }
}

console.log('Generating curated files acorn cache...');
console.log('Source directory:', CURATED_DIR);
console.log('Cache directory:', CACHE_DIR);
if (FORCE_REGENERATE) {
    console.log('Mode: FORCE REGENERATE');
} else {
    console.log('Mode: Incremental');
}

// Collect all files
const allFiles = collectFiles(CURATED_DIR);
console.log(`Found ${allFiles.length} files to process`);

let processed = 0;
let successful = 0;
let cached = 0;
let regenerated = 0;
let failed = 0;
const failedFiles = [];

for (const file of allFiles) {
    const result = processFile(file);
    processed++;

    if (result.success) {
        successful++;
        if (result.cached) {
            cached++;
        } else {
            regenerated++;
        }
    } else {
        failed++;
        failedFiles.push(`${path.relative(CURATED_DIR, file)}: ${result.error}`);
    }
}

console.log('\n=== Cache Generation Results ===');
console.log(`Total files: ${allFiles.length}`);
console.log(`Successfully processed: ${successful}`);
console.log(`  - Used existing cache: ${cached}`);
console.log(`  - Generated: ${regenerated}`);
console.log(`Failed to parse: ${failed}`);

if (failedFiles.length > 0) {
    console.log('\nFailed files:');
    failedFiles.forEach(f => console.log('  ' + f));
}

console.log('\nCache generation complete!');
