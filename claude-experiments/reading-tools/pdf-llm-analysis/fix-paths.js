#!/usr/bin/env node

/**
 * Fix stale paths in PDF index by matching SHA256 hashes.
 *
 * SAFE: Writes to a NEW file, does NOT modify the original.
 */

import { readFile, writeFile } from 'fs/promises';
import { createHash } from 'crypto';
import { readFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';

/**
 * Recursively find all PDF files in a directory
 */
function findPDFs(dir, files = []) {
  const entries = readdirSync(dir);

  for (const entry of entries) {
    const fullPath = join(dir, entry);
    try {
      const stat = statSync(fullPath);
      if (stat.isDirectory()) {
        findPDFs(fullPath, files);
      } else if (entry.toLowerCase().endsWith('.pdf')) {
        files.push(fullPath);
      }
    } catch (err) {
      // Skip files we can't access
      console.error(`  Warning: Could not access ${fullPath}`);
    }
  }

  return files;
}

/**
 * Calculate SHA256 hash of a file
 */
function hashFile(filePath) {
  const content = readFileSync(filePath);
  return createHash('sha256').update(content).digest('hex');
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 2 || args.includes('--help') || args.includes('-h')) {
    console.log(`
Fix PDF Index Paths

Usage:
  node fix-paths.js <index-file> <pdf-directory> [output-file]

Arguments:
  index-file     Path to the JSON index file with stale paths
  pdf-directory  Directory to scan for actual PDF files
  output-file    Output file (default: <index-file>-fixed.json)

Example:
  node fix-paths.js ../pdf-indexer/pdf-index.json /Users/jimmyhmiller/Documents/Code/readings

The script will:
  1. Read the existing index
  2. Scan the directory for all PDFs
  3. Compute SHA256 hash for each PDF
  4. Update paths in the index based on hash matching
  5. Write to a NEW file (original is NOT modified)
    `);
    process.exit(0);
  }

  const indexPath = args[0];
  const pdfDir = args[1];
  const outputPath = args[2] || indexPath.replace('.json', '-fixed.json');

  console.log('='.repeat(60));
  console.log('Fix PDF Index Paths');
  console.log('='.repeat(60));
  console.log(`Input index:  ${indexPath}`);
  console.log(`PDF directory: ${pdfDir}`);
  console.log(`Output file:  ${outputPath}`);
  console.log('');

  // Read the index
  console.log('Reading index file...');
  const indexData = JSON.parse(await readFile(indexPath, 'utf-8'));

  if (!Array.isArray(indexData)) {
    throw new Error('Index file is not an array');
  }

  console.log(`Found ${indexData.length} entries in index`);

  // Build hash -> entry map
  const hashToEntries = new Map();
  for (const entry of indexData) {
    if (entry.hash) {
      if (!hashToEntries.has(entry.hash)) {
        hashToEntries.set(entry.hash, []);
      }
      hashToEntries.get(entry.hash).push(entry);
    }
  }
  console.log(`${hashToEntries.size} unique hashes in index`);
  console.log('');

  // Find all PDFs in directory
  console.log(`Scanning ${pdfDir} for PDFs...`);
  const pdfFiles = findPDFs(pdfDir);
  console.log(`Found ${pdfFiles.length} PDF files`);
  console.log('');

  // Hash each PDF and update paths
  console.log('Matching files by hash...');
  let matched = 0;
  let notInIndex = 0;
  let alreadyCorrect = 0;

  for (let i = 0; i < pdfFiles.length; i++) {
    const pdfPath = pdfFiles[i];

    if ((i + 1) % 50 === 0 || i === pdfFiles.length - 1) {
      process.stdout.write(`\r  Progress: ${i + 1}/${pdfFiles.length}`);
    }

    try {
      const hash = hashFile(pdfPath);
      const entries = hashToEntries.get(hash);

      if (entries) {
        for (const entry of entries) {
          if (entry.path === pdfPath) {
            alreadyCorrect++;
          } else {
            entry.path = pdfPath;
            entry.fileName = pdfPath.split('/').pop();
            matched++;
          }
        }
      } else {
        notInIndex++;
      }
    } catch (err) {
      console.error(`\n  Error hashing ${pdfPath}: ${err.message}`);
    }
  }

  console.log('\n');

  // Count entries with paths that still don't exist
  let stillMissing = 0;
  for (const entry of indexData) {
    try {
      statSync(entry.path);
    } catch {
      stillMissing++;
    }
  }

  // Write output
  console.log(`Writing to ${outputPath}...`);
  await writeFile(outputPath, JSON.stringify(indexData, null, 2));

  // Summary
  console.log('');
  console.log('='.repeat(60));
  console.log('Summary:');
  console.log(`  Paths updated:      ${matched}`);
  console.log(`  Already correct:    ${alreadyCorrect}`);
  console.log(`  PDFs not in index:  ${notInIndex}`);
  console.log(`  Still missing:      ${stillMissing} (files no longer exist)`);
  console.log('='.repeat(60));
  console.log('');
  console.log(`Original file unchanged: ${indexPath}`);
  console.log(`Fixed index written to:  ${outputPath}`);
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
