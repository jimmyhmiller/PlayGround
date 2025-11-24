#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const acorn = require('acorn');
const crypto = require('crypto');

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length === 0) {
    console.error('Usage: node scripts/test-package.js <package-name|path-to-package.json>');
    console.error('');
    console.error('Examples:');
    console.error('  node scripts/test-package.js express');
    console.error('  node scripts/test-package.js /path/to/project/package.json');
    console.error('  node scripts/test-package.js /path/to/project');
    process.exit(1);
}

const input = args[0];
const CACHE_DIR = path.join(__dirname, '..', 'test-oracles', 'adhoc-cache');
const TEMP_DIR = path.join(__dirname, '..', 'temp-packages');

// Ensure directories exist
if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
}
if (!fs.existsSync(TEMP_DIR)) {
    fs.mkdirSync(TEMP_DIR, { recursive: true });
}

function getFileHash(filePath) {
    const content = fs.readFileSync(filePath);
    return crypto.createHash('sha256').update(content).digest('hex');
}

function isPackageInstalled(packageDir) {
    return fs.existsSync(path.join(packageDir, 'node_modules'));
}

function installPackage(packageName) {
    console.log(`Installing package: ${packageName}...`);
    const packageDir = path.join(TEMP_DIR, packageName.replace(/[^a-zA-Z0-9-]/g, '_'));

    if (!fs.existsSync(packageDir)) {
        fs.mkdirSync(packageDir, { recursive: true });
    }

    // Create package.json if it doesn't exist
    const pkgJsonPath = path.join(packageDir, 'package.json');
    if (!fs.existsSync(pkgJsonPath)) {
        fs.writeFileSync(pkgJsonPath, JSON.stringify({
            name: 'test-package',
            version: '1.0.0',
            dependencies: {}
        }, null, 2));
    }

    // Install the package
    try {
        execSync(`npm install ${packageName}`, {
            cwd: packageDir,
            stdio: 'inherit'
        });
        console.log(`✓ Package installed to: ${packageDir}`);
        return packageDir;
    } catch (e) {
        console.error(`✗ Failed to install package: ${e.message}`);
        process.exit(1);
    }
}

function findSourceRoot(packageDir, packageName) {
    // Common patterns for source directories
    const nodeModulesPath = path.join(packageDir, 'node_modules', packageName);

    if (!fs.existsSync(nodeModulesPath)) {
        console.error(`Package not found at: ${nodeModulesPath}`);
        return null;
    }

    return nodeModulesPath;
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

function processFile(filePath, cacheSubDir) {
    const relativePath = path.relative(path.dirname(cacheSubDir), filePath);
    const cacheFileName = relativePath.replace(/[\/\\]/g, '_') + '.json';
    const cacheFilePath = path.join(cacheSubDir, cacheFileName);

    try {
        const source = fs.readFileSync(filePath, 'utf-8');

        // Skip very large files (> 500KB)
        if (source.length > 500000) {
            return { success: false, skipped: true, reason: 'too large' };
        }

        // Determine source type
        const sourceType = shouldUseModuleMode(filePath, source) ? 'module' : 'script';

        // Parse with Acorn
        const ast = acorn.parse(source, {
            ecmaVersion: 2025,
            locations: true,
            sourceType: sourceType
        });

        // Add metadata
        ast._metadata = {
            sourceHash: getFileHash(filePath),
            generatedAt: new Date().toISOString(),
            acornVersion: acorn.version,
            sourceFile: relativePath,
            sourceType: sourceType
        };

        // Serialize and save
        const jsonOutput = serializeAST(ast);
        fs.writeFileSync(cacheFilePath, jsonOutput, 'utf8');

        return {
            success: true,
            cached: true,
            sourceType: sourceType,
            cacheFile: cacheFilePath
        };
    } catch (e) {
        return {
            success: false,
            error: e.message,
            position: e.pos
        };
    }
}

function main() {
    let sourceDir;
    let packageName;

    // Determine if input is a package name or path
    if (fs.existsSync(input)) {
        // It's a local path
        if (fs.statSync(input).isDirectory()) {
            sourceDir = input;
            packageName = path.basename(input);
        } else if (input.endsWith('package.json')) {
            sourceDir = path.dirname(input);
            const pkgJson = JSON.parse(fs.readFileSync(input, 'utf-8'));
            packageName = pkgJson.name || path.basename(sourceDir);
        } else {
            console.error('Input must be a directory, package.json file, or package name');
            process.exit(1);
        }
        console.log(`Testing local package: ${packageName}`);
        console.log(`Source directory: ${sourceDir}`);
    } else {
        // It's a package name - install it
        packageName = input;
        const packageDir = installPackage(packageName);
        sourceDir = findSourceRoot(packageDir, packageName);

        if (!sourceDir) {
            console.error('Could not find package source directory');
            process.exit(1);
        }
        console.log(`Package source: ${sourceDir}`);
    }

    // Create cache subdirectory for this package
    const cacheSubDir = path.join(CACHE_DIR, packageName.replace(/[^a-zA-Z0-9-]/g, '_'));
    if (!fs.existsSync(cacheSubDir)) {
        fs.mkdirSync(cacheSubDir, { recursive: true });
    }

    // Collect all JS files
    console.log('\nScanning for JavaScript files...');
    const jsFiles = collectJsFiles(sourceDir);
    console.log(`Found ${jsFiles.length} JavaScript files\n`);

    if (jsFiles.length === 0) {
        console.log('No JavaScript files found to test');
        process.exit(0);
    }

    // Process each file
    let processed = 0;
    let successful = 0;
    let failed = 0;
    let skipped = 0;
    const failedFiles = [];
    const successfulFiles = [];

    console.log('Processing files...');
    for (const file of jsFiles) {
        const result = processFile(file, cacheSubDir);
        processed++;

        if (result.success) {
            successful++;
            successfulFiles.push({
                file: path.relative(sourceDir, file),
                cacheFile: result.cacheFile,
                sourceType: result.sourceType
            });
        } else if (result.skipped) {
            skipped++;
        } else {
            failed++;
            if (failedFiles.length < 50) {
                failedFiles.push({
                    file: path.relative(sourceDir, file),
                    error: result.error
                });
            }
        }

        if (processed % 10 === 0) {
            console.log(`Progress: ${processed}/${jsFiles.length} (${successful} success, ${failed} failed, ${skipped} skipped)`);
        }
    }

    // Write results summary
    const summaryPath = path.join(cacheSubDir, '_summary.json');
    const summary = {
        packageName: packageName,
        sourceDir: sourceDir,
        timestamp: new Date().toISOString(),
        total: jsFiles.length,
        successful: successful,
        failed: failed,
        skipped: skipped,
        successfulFiles: successfulFiles,
        failedFiles: failedFiles
    };
    fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

    console.log('\n=== Results ===');
    console.log(`Total files: ${jsFiles.length}`);
    console.log(`✓ Successfully parsed: ${successful} (${(successful * 100 / jsFiles.length).toFixed(2)}%)`);
    console.log(`✗ Failed to parse: ${failed} (${(failed * 100 / jsFiles.length).toFixed(2)}%)`);
    console.log(`⊘ Skipped: ${skipped} (${(skipped * 100 / jsFiles.length).toFixed(2)}%)`);
    console.log(`\nCache directory: ${cacheSubDir}`);
    console.log(`Summary file: ${summaryPath}`);

    if (failedFiles.length > 0) {
        console.log('\nFirst 50 failed files:');
        failedFiles.forEach(f => console.log(`  ${f.file}: ${f.error}`));
    }

    console.log('\nNext step: Run Java tests against this cache');
    console.log(`  mvn test -Dtest=AdhocPackageTest -DpackageName=${packageName}`);
}

main();
