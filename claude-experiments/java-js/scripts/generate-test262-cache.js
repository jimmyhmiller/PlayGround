#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const acorn = require('acorn');
const os = require('os');

const TEST262_DIR = path.join(__dirname, '..', 'test-oracles', 'test262', 'test');
const CACHE_DIR = path.join(__dirname, '..', 'test-oracles', 'test262-cache');
const NUM_WORKERS = os.cpus().length;
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

        // Check if cache has hash metadata
        if (!cache._metadata || !cache._metadata.sourceHash) {
            return false;
        }

        // Compare hash with current file
        const currentHash = getFileHash(sourceFile);
        return cache._metadata.sourceHash === currentHash;
    } catch (e) {
        return false;
    }
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

// Serialize AST to JSON exactly like Acorn CLI does
function serializeAST(ast) {
    // Match Acorn CLI's JSON.stringify behavior:
    // BigInt values in the AST are converted to null
    // (The bigint field already contains the string representation)
    return JSON.stringify(ast, function(_, value) {
        return typeof value === "bigint" ? null : value;
    }, 2);
}

// Process a single file
function processFile(filePath) {
    const relativePath = path.relative(TEST262_DIR, filePath);
    const cacheFilePath = path.join(CACHE_DIR, relativePath + '.json');

    // Check if cache is valid and not forcing regeneration
    if (!FORCE_REGENERATE && isCacheValid(filePath, cacheFilePath)) {
        return { success: true, cached: true };
    }

    try {
        const source = fs.readFileSync(filePath, 'utf-8');

        // Parse with Acorn using same options as CLI
        const options = {
            ecmaVersion: 2025,
            locations: true,
            sourceType: shouldUseModuleMode(filePath) ? 'module' : 'script'
        };

        const ast = acorn.parse(source, options);

        const cacheFileDir = path.dirname(cacheFilePath);

        if (!fs.existsSync(cacheFileDir)) {
            fs.mkdirSync(cacheFileDir, { recursive: true });
        }

        // Add metadata before serialization
        ast._metadata = {
            sourceHash: getFileHash(filePath),
            generatedAt: new Date().toISOString(),
            acornVersion: acorn.version,
            sourceFile: relativePath
        };

        // Serialize using exact same method as Acorn CLI
        const jsonOutput = serializeAST(ast);

        fs.writeFileSync(cacheFilePath, jsonOutput, 'utf8');
        return { success: true, cached: false };
    } catch (e) {
        return { success: false, error: e.message };
    }
}

console.log('Generating test262 acorn cache...');
console.log('Scanning:', TEST262_DIR);
console.log('Cache directory:', CACHE_DIR);
if (FORCE_REGENERATE) {
    console.log('Mode: FORCE REGENERATE (ignoring existing cache)');
} else {
    console.log('Mode: Incremental (using cached files with valid hashes)');
}

// Collect all files
const allFiles = collectFiles(TEST262_DIR);
console.log(`Found ${allFiles.length} files to process`);

// Process files in batches
const BATCH_SIZE = 100;
let processed = 0;
let successful = 0;
let cached = 0;
let regenerated = 0;
let failed = 0;
const failedFiles = [];

for (let i = 0; i < allFiles.length; i += BATCH_SIZE) {
    const batch = allFiles.slice(i, i + BATCH_SIZE);

    for (const file of batch) {
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
            if (failedFiles.length < 20) {
                failedFiles.push(`${file}: ${result.error}`);
            }
        }
    }

    console.log(`Progress: ${processed}/${allFiles.length} files, ${cached} cached, ${regenerated} regenerated, ${failed} failed`);
}

console.log('\n=== Cache Generation Results ===');
console.log(`Total files: ${allFiles.length}`);
console.log(`Successfully processed: ${successful} (${(successful * 100 / allFiles.length).toFixed(2)}%)`);
console.log(`  - Used existing cache: ${cached}`);
console.log(`  - Regenerated: ${regenerated}`);
console.log(`Failed to parse: ${failed} (${(failed * 100 / allFiles.length).toFixed(2)}%)`);

if (failedFiles.length > 0) {
    console.log('\nFirst 20 failed files:');
    failedFiles.forEach(f => console.log('  ' + f));
}

console.log('\nCache generation complete!');
if (!FORCE_REGENERATE && regenerated > 0) {
    console.log(`\nNote: ${regenerated} files were regenerated due to hash mismatches.`);
    console.log('To force regeneration of all files, run with --force flag.');
}
