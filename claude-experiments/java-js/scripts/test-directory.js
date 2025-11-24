#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const acorn = require('acorn');

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length === 0) {
    console.error('Usage: node scripts/test-directory.js <directory-path>');
    console.error('');
    console.error('Examples:');
    console.error('  node scripts/test-directory.js ../my-project');
    console.error('  node scripts/test-directory.js ~/code/app');
    process.exit(1);
}

const targetDir = args[0];

if (!fs.existsSync(targetDir)) {
    console.error(`Directory not found: ${targetDir}`);
    process.exit(1);
}

function collectJsFiles(dir, files = [], maxDepth = 20, currentDepth = 0) {
    if (currentDepth > maxDepth) {
        return files;
    }

    try {
        const entries = fs.readdirSync(dir);

        for (const entry of entries) {
            // Skip only .git directory
            if (entry === '.git') {
                continue;
            }

            const fullPath = path.join(dir, entry);

            try {
                const stat = fs.statSync(fullPath);

                if (stat.isDirectory()) {
                    collectJsFiles(fullPath, files, maxDepth, currentDepth + 1);
                } else if (entry.endsWith('.js') || entry.endsWith('.mjs')) {
                    files.push(fullPath);
                }
            } catch (e) {
                // Skip files we can't stat (permission issues, etc.)
                continue;
            }
        }
    } catch (e) {
        // Skip directories we can't read
        return files;
    }

    return files;
}

function shouldUseModuleMode(filePath, source) {
    // .mjs files are always modules
    if (filePath.endsWith('.mjs')) {
        return true;
    }

    // Check for ES module syntax
    if (source.includes('import ') || source.includes('export ')) {
        return true;
    }

    return false;
}

function serializeAST(ast) {
    return JSON.stringify(ast, function(_, value) {
        return typeof value === "bigint" ? null : value;
    }, 2);
}

function parseWithAcorn(source, filePath) {
    try {
        const sourceType = shouldUseModuleMode(filePath, source) ? 'module' : 'script';
        const ast = acorn.parse(source, {
            ecmaVersion: 2025,
            locations: true,
            sourceType: sourceType
        });
        return { success: true, ast, sourceType };
    } catch (e) {
        return { success: false, error: e.message };
    }
}

function parseWithJava(source, isModule, filePath) {
    try {
        // Create temp file
        const tempFile = `/tmp/adhoc_test_${Date.now()}.js`;
        fs.writeFileSync(tempFile, source);

        // Call Java parser via Maven
        const command = `mvn exec:java -Dexec.mainClass="com.jsparser.Parser" -Dexec.args="${tempFile} ${isModule ? 'module' : 'script'}" -q 2>&1`;
        const output = execSync(command, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });

        // Clean up
        fs.unlinkSync(tempFile);

        return { success: true, ast: JSON.parse(output) };
    } catch (e) {
        return { success: false, error: e.message };
    }
}

function compareASTs(expected, actual) {
    // Simple deep equality check
    return JSON.stringify(expected) === JSON.stringify(actual);
}

console.log('Ad-hoc directory testing (real-time comparison)');
console.log('Directory:', path.resolve(targetDir));
console.log('');

// Collect all files
console.log('Scanning for JavaScript files...');
const jsFiles = collectJsFiles(targetDir);
console.log(`Found ${jsFiles.length} JavaScript files\n`);

if (jsFiles.length === 0) {
    console.log('No JavaScript files found');
    process.exit(0);
}

// Process files
let processed = 0;
let acornSuccess = 0;
let acornFailed = 0;
let javaSuccess = 0;
let javaFailed = 0;
let matched = 0;
let mismatched = 0;
let skipped = 0;

const failedFiles = [];
const mismatchedFiles = [];

console.log('Testing files (this may take a while for large projects)...\n');

for (const file of jsFiles) {
    processed++;

    if (processed % 100 === 0) {
        console.log(`Progress: ${processed}/${jsFiles.length} (${matched} matched, ${mismatched} mismatched, ${javaFailed} failed)`);
    }

    const relativePath = path.relative(targetDir, file);

    try {
        const source = fs.readFileSync(file, 'utf-8');

        // Skip very large files (> 500KB)
        if (source.length > 500000) {
            skipped++;
            continue;
        }

        // Parse with Acorn
        const acornResult = parseWithAcorn(source, file);

        if (!acornResult.success) {
            acornFailed++;
            skipped++; // Skip if Acorn can't parse it
            continue;
        }

        acornSuccess++;

        // Parse with Java
        const javaResult = parseWithJava(source, acornResult.sourceType === 'module', file);

        if (!javaResult.success) {
            javaFailed++;
            if (failedFiles.length < 50) {
                failedFiles.push({
                    file: relativePath,
                    error: javaResult.error
                });
            }
            continue;
        }

        javaSuccess++;

        // Compare ASTs
        if (compareASTs(acornResult.ast, javaResult.ast)) {
            matched++;
        } else {
            mismatched++;
            if (mismatchedFiles.length < 20) {
                mismatchedFiles.push(relativePath);
            }
        }
    } catch (e) {
        javaFailed++;
        if (failedFiles.length < 50) {
            failedFiles.push({
                file: relativePath,
                error: e.message
            });
        }
    }
}

console.log('\n=== Results ===');
console.log(`Total files: ${jsFiles.length}`);
console.log(`Skipped (too large or Acorn failed): ${skipped}`);
console.log(`\nAcorn results:`);
console.log(`  ✓ Successfully parsed: ${acornSuccess} (${(acornSuccess * 100 / jsFiles.length).toFixed(2)}%)`);
console.log(`  ✗ Failed to parse: ${acornFailed} (${(acornFailed * 100 / jsFiles.length).toFixed(2)}%)`);
console.log(`\nJava parser results (on files Acorn succeeded):`);
console.log(`  ✓ Successfully parsed: ${javaSuccess} (${(javaSuccess * 100 / acornSuccess).toFixed(2)}%)`);
console.log(`  ✗ Failed to parse: ${javaFailed} (${(javaFailed * 100 / acornSuccess).toFixed(2)}%)`);
console.log(`\nComparison (on files both parsers succeeded):`);
const totalSuccess = javaSuccess;
console.log(`  ✓ Exact matches: ${matched} (${totalSuccess > 0 ? (matched * 100 / totalSuccess).toFixed(2) : 0}%)`);
console.log(`  ✗ AST mismatches: ${mismatched} (${totalSuccess > 0 ? (mismatched * 100 / totalSuccess).toFixed(2) : 0}%)`);

if (mismatchedFiles.length > 0) {
    console.log('\nFirst 20 mismatched files:');
    mismatchedFiles.forEach(f => console.log(`  ${f}`));
}

if (failedFiles.length > 0) {
    console.log('\nFirst 50 failed files:');
    failedFiles.forEach(f => console.log(`  ${f.file}: ${f.error}`));
}
