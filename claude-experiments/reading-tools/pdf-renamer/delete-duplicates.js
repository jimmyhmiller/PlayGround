#!/usr/bin/env node

const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
const crypto = require('crypto');

/**
 * Calculate SHA256 hash of a file
 */
async function calculateHash(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fsSync.createReadStream(filePath);

    stream.on('data', (data) => hash.update(data));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', (error) => reject(error));
  });
}

/**
 * Recursively find all PDF files in a directory
 */
async function findPDFs(directory) {
  const pdfs = [];

  async function traverse(dir) {
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          await traverse(fullPath);
        } else if (entry.isFile() && path.extname(entry.name).toLowerCase() === '.pdf') {
          pdfs.push(fullPath);
        }
      }
    } catch (error) {
      console.error(`Error reading directory ${dir}:`, error.message);
    }
  }

  await traverse(directory);
  return pdfs;
}

/**
 * Find and optionally delete duplicate PDF files based on hash
 */
async function deleteDuplicates(directory, options = {}) {
  const { dryRun = false } = options;

  console.log(`Scanning directory: ${directory}`);
  const pdfFiles = await findPDFs(directory);
  console.log(`Found ${pdfFiles.length} PDF files`);
  console.log('');

  // Map of hash -> array of file paths
  const hashMap = new Map();

  console.log('Calculating hashes...');
  for (const pdfPath of pdfFiles) {
    try {
      const hash = await calculateHash(pdfPath);

      if (!hashMap.has(hash)) {
        hashMap.set(hash, []);
      }
      hashMap.get(hash).push(pdfPath);
    } catch (error) {
      console.error(`❌ ERROR hashing ${path.basename(pdfPath)}: ${error.message}`);
    }
  }

  console.log('');
  console.log('=== Duplicate Analysis ===');
  console.log('');

  const stats = {
    totalFiles: pdfFiles.length,
    uniqueFiles: 0,
    duplicateSets: 0,
    duplicateFiles: 0,
    deleted: 0,
    errors: 0,
  };

  /**
   * Get the depth of a path (number of directory separators)
   */
  function getDepth(filePath) {
    return filePath.split(path.sep).length;
  }

  for (const [hash, files] of hashMap.entries()) {
    if (files.length > 1) {
      stats.duplicateSets++;
      stats.duplicateFiles += files.length - 1; // Don't count the one we keep

      console.log(`📦 Duplicate set (${files.length} copies):`);

      // Sort to determine which file to keep
      const sorted = files.sort((a, b) => {
        const aDepth = getDepth(a);
        const bDepth = getDepth(b);

        // Prefer more nested (deeper) paths
        if (aDepth !== bDepth) {
          return bDepth - aDepth; // Higher depth wins (b > a means b first)
        }

        // If same depth, prefer non-hash filenames
        const aName = path.basename(a);
        const bName = path.basename(b);
        const aIsHash = /^[0-9a-f]{32}_/.test(aName);
        const bIsHash = /^[0-9a-f]{32}_/.test(bName);

        if (aIsHash && !bIsHash) return 1;  // b wins
        if (!aIsHash && bIsHash) return -1; // a wins

        // Otherwise prefer shorter names
        return aName.length - bName.length;
      });

      const keep = sorted[0];
      const toDelete = sorted.slice(1);

      console.log(`  ✓ KEEP: ${keep}`);

      for (const duplicate of toDelete) {
        if (dryRun) {
          console.log(`  🗑️  DRY-RUN DELETE: ${duplicate}`);
          stats.deleted++;
        } else {
          try {
            await fs.unlink(duplicate);
            console.log(`  ✓ DELETED: ${duplicate}`);
            stats.deleted++;
          } catch (error) {
            console.error(`  ❌ ERROR deleting ${duplicate}: ${error.message}`);
            stats.errors++;
          }
        }
      }
      console.log('');
    } else {
      stats.uniqueFiles++;
    }
  }

  console.log('=== Summary ===');
  console.log(`Total files scanned: ${stats.totalFiles}`);
  console.log(`Unique files (no duplicates): ${stats.uniqueFiles}`);
  console.log(`Duplicate sets found: ${stats.duplicateSets}`);
  console.log(`Duplicate files: ${stats.duplicateFiles}`);
  console.log(`Files deleted: ${stats.deleted}`);
  console.log(`Errors: ${stats.errors}`);
  console.log('');

  if (stats.deleted > 0) {
    const savedSpace = stats.deleted;
    console.log(`Space savings: ~${savedSpace} files removed`);
  }

  if (dryRun) {
    console.log('This was a dry run. No files were actually deleted.');
    console.log('Run without --dry-run to perform the actual deletions.');
  }
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
PDF Duplicate Remover - Delete hash-identical duplicate PDF files

Usage:
  node delete-duplicates.js [options]

Options:
  --dir <path>        Directory containing PDFs (required)
  --dry-run           Preview what would be deleted without actually deleting
  -h, --help          Show this help message

Strategy:
  For each set of duplicates (files with identical hashes):
  - Keeps the file with the cleanest/shortest filename
  - Prefers non-hash filenames over hash-based filenames
  - Deletes all other copies

Examples:
  # Dry run to preview what would be deleted
  node delete-duplicates.js --dir /path/to/pdfs --dry-run

  # Actually delete duplicates
  node delete-duplicates.js --dir /path/to/pdfs
`);
    process.exit(0);
  }

  const directory = args.includes('--dir')
    ? args[args.indexOf('--dir') + 1]
    : null;

  if (!directory) {
    console.error('Error: --dir parameter is required');
    console.error('Run with --help for usage information');
    process.exit(1);
  }

  const dryRun = args.includes('--dry-run');

  deleteDuplicates(directory, { dryRun })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

module.exports = { deleteDuplicates };
