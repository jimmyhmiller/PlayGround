#!/usr/bin/env node

import { readFile, writeFile } from 'fs/promises';
import { extractMetadataWithLLM } from './extract-metadata.js';

/**
 * Calculate Levenshtein distance between two strings
 * @param {string} a - First string
 * @param {string} b - Second string
 * @returns {number} Edit distance
 */
function levenshteinDistance(a, b) {
  if (!a || !b) return Math.max(a?.length || 0, b?.length || 0);

  const matrix = [];

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[b.length][a.length];
}

/**
 * Calculate similarity ratio between two strings (0-1, higher is more similar)
 * @param {string} a - First string
 * @param {string} b - Second string
 * @returns {number} Similarity ratio
 */
function similarityRatio(a, b) {
  if (!a || !b) return 0;

  const longer = a.length > b.length ? a : b;
  const distance = levenshteinDistance(a.toLowerCase(), b.toLowerCase());

  return (longer.length - distance) / longer.length;
}

/**
 * Check if two titles match (fuzzy matching)
 * Handles cases where one title is a prefix/substring of another (e.g., with/without subtitle)
 * @param {string} title1 - First title
 * @param {string} title2 - Second title
 * @param {number} threshold - Similarity threshold (default: 0.8)
 * @returns {boolean} True if titles match
 */
function titlesMatch(title1, title2, threshold = 0.8) {
  if (!title1 || !title2) return false;

  const t1Lower = title1.toLowerCase().trim();
  const t2Lower = title2.toLowerCase().trim();

  // Check if one is a substring of the other (handles title with/without subtitle)
  const shorter = t1Lower.length < t2Lower.length ? t1Lower : t2Lower;
  const longer = t1Lower.length < t2Lower.length ? t2Lower : t1Lower;

  if (longer.startsWith(shorter)) {
    console.log(`  Match: One title is a prefix of the other`);
    return true;
  }

  // Otherwise use fuzzy matching
  const similarity = similarityRatio(title1, title2);
  console.log(`  Similarity: ${(similarity * 100).toFixed(1)}% (threshold: ${(threshold * 100).toFixed(0)}%)`);

  return similarity >= threshold;
}

/**
 * Update PDF index with LLM-extracted metadata
 * @param {string} indexPath - Path to index JSON file
 * @param {boolean} verify - Whether to verify existing metadata
 * @param {boolean} forceAll - Whether to process all PDFs without ocr_title (regardless of metadataFound)
 * @param {number} numPages - Number of pages to analyze per PDF
 * @param {number} threshold - Similarity threshold for matching
 * @param {number} dpi - Image resolution in DPI (lower = faster)
 * @param {number} parallel - Number of PDFs to process in parallel
 */
async function updateIndex(indexPath, verify = false, forceAll = false, numPages = 3, threshold = 0.8, dpi = 72, parallel = 1) {
  console.log('='.repeat(60));
  console.log('Configuration:');
  console.log(`  Index file: ${indexPath}`);
  console.log(`  Verify mode: ${verify ? 'enabled' : 'disabled'}`);
  console.log(`  Force all mode: ${forceAll ? 'enabled' : 'disabled'}`);
  console.log(`  Pages per PDF: ${numPages}`);
  console.log(`  Image resolution: ${dpi} DPI`);
  console.log(`  Similarity threshold: ${(threshold * 100).toFixed(0)}%`);
  console.log(`  Parallel workers: ${parallel}`);
  console.log('='.repeat(60));
  console.log('');

  // Read the index file
  const indexData = JSON.parse(await readFile(indexPath, 'utf-8'));

  if (!Array.isArray(indexData)) {
    throw new Error('Index file is not an array');
  }

  console.log(`Found ${indexData.length} PDFs in index`);

  // Count different categories
  const withMetadata = indexData.filter(e => e.metadataFound).length;
  const withOcrTitle = indexData.filter(e => e.ocr_title !== undefined).length;

  console.log(`  - With embedded metadata: ${withMetadata}`);
  console.log(`  - Already processed (ocr_title): ${withOcrTitle}`);
  console.log('');

  // Filter PDFs that need processing
  const toProcess = indexData.filter(entry => {
    // Skip documents classified as "other" (not articles/papers/books)
    if (entry.document_type === 'other') {
      return false;
    }

    if (verify) {
      // With verify flag, process everything (except "other" type)
      return true;
    } else if (forceAll) {
      // With force-all flag, process all PDFs without ocr_title (regardless of metadataFound)
      return entry.ocr_title === undefined || entry.ocr_title === null;
    } else {
      // Without verify flag, only process PDFs without metadata AND without ocr_title
      // Skip PDFs that already have been successfully processed
      return !entry.metadataFound && (entry.ocr_title === undefined || entry.ocr_title === null);
    }
  });

  console.log(`${toProcess.length} PDFs need processing`);
  console.log('');

  let processedCount = 0;
  let matchedCount = 0;
  let mismatchedCount = 0;
  let errorCount = 0;

  // Process a single PDF entry
  async function processEntry(entry, entryNum) {
    const pdfIndex = indexData.indexOf(entry);
    const prefix = `[${entryNum}/${toProcess.length}]`;

    console.log(`${prefix} Processing: ${entry.fileName}`);

    try {
      // Extract metadata using LLM (quiet mode)
      const llmMetadata = await extractMetadataWithLLM(entry.path, numPages, true, dpi);

      console.log(`${prefix} LLM: "${llmMetadata.title}" by "${llmMetadata.author}"`);

      // Check if we need to update based on matching
      let needsUpdate = false;

      if (!entry.metadataFound) {
        needsUpdate = true;
      } else if (verify) {
        const match = titlesMatch(entry.title, llmMetadata.title, threshold);
        if (match) {
          matchedCount++;
        } else {
          needsUpdate = true;
          mismatchedCount++;
        }
      } else if (forceAll) {
        needsUpdate = true;
      }

      if (needsUpdate) {
        indexData[pdfIndex].ocr_title = llmMetadata.title;
        indexData[pdfIndex].ocr_author = llmMetadata.author;
        processedCount++;
        return true; // Signal that we updated
      }

    } catch (error) {
      console.error(`${prefix} Error: ${error.message}`);
      errorCount++;
    }
    return false;
  }

  // Process in batches for parallel execution
  for (let i = 0; i < toProcess.length; i += parallel) {
    const batch = toProcess.slice(i, i + parallel);
    const batchPromises = batch.map((entry, j) => processEntry(entry, i + j + 1));

    const results = await Promise.all(batchPromises);

    // Save after each batch if any updates were made
    if (results.some(r => r)) {
      await writeFile(indexPath, JSON.stringify(indexData, null, 2));
      console.log(`  [Saved progress: ${processedCount} updated so far]`);
    }
  }

  // Final message
  console.log('Index file updated:', indexPath);

  // Print summary
  console.log('='.repeat(60));
  console.log('Summary:');
  console.log(`Total PDFs in index: ${indexData.length}`);
  console.log(`PDFs examined: ${toProcess.length}`);
  console.log(`PDFs updated with OCR fields: ${processedCount}`);

  if (verify) {
    console.log(`Titles matched: ${matchedCount}`);
    console.log(`Titles mismatched: ${mismatchedCount}`);
  }

  if (errorCount > 0) {
    console.log(`Errors: ${errorCount}`);
  }

  console.log('='.repeat(60));
}

/**
 * Main function - CLI interface
 */
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
PDF Index Updater using Vision LLM

Usage:
  node update-index.js <index-file> [options]

Arguments:
  index-file    Path to index JSON file (required)

Options:
  --verify      Verify existing metadata and add ocr_title if mismatched
  --force-all   Process ALL PDFs without ocr_title (including those with bad metadata)
  --pages N     Number of pages to analyze per PDF (default: 3)
  --threshold T Similarity threshold for matching (0-1, default: 0.8)
  --dpi N       Image resolution in DPI (default: 72, lower = faster)
  --parallel N  Process N PDFs in parallel (default: 1)

Examples:
  # Update PDFs without metadata
  node update-index.js ../pdf-indexer/pdf-index.json

  # Process all PDFs that don't have ocr_title yet (including bad metadata)
  node update-index.js ../pdf-indexer/pdf-index.json --force-all

  # Verify all PDFs and add ocr_title for mismatches
  node update-index.js ../pdf-indexer/pdf-index.json --verify

  # Use 5 pages for better accuracy
  node update-index.js ../pdf-indexer/pdf-index.json --pages 5

  # Use stricter matching (90% similarity required)
  node update-index.js ../pdf-indexer/pdf-index.json --verify --threshold 0.9

  # Use lower resolution for faster processing
  node update-index.js ../pdf-indexer/pdf-index.json --dpi 50

  # Use higher resolution for better accuracy
  node update-index.js ../pdf-indexer/pdf-index.json --dpi 150

  # Process 4 PDFs in parallel for faster batch processing
  node update-index.js ../pdf-indexer/pdf-index.json --force-all --parallel 4

The tool will:
  1. Read the existing index file
  2. Process PDFs without metadata (or all PDFs with --verify/--force-all)
  3. Extract title/author using qwen-vl vision model
  4. Add ocr_title and ocr_author fields when needed
  5. Use fuzzy matching to compare titles (with --verify)
  6. Save the updated index back to the same file
    `);
    process.exit(0);
  }

  const indexPath = args[0];
  const verify = args.includes('--verify');
  const forceAll = args.includes('--force-all');

  // Parse --pages option
  const pagesIndex = args.indexOf('--pages');
  const numPages = pagesIndex !== -1 && args[pagesIndex + 1]
    ? parseInt(args[pagesIndex + 1])
    : 3;

  // Parse --threshold option
  const thresholdIndex = args.indexOf('--threshold');
  const threshold = thresholdIndex !== -1 && args[thresholdIndex + 1]
    ? parseFloat(args[thresholdIndex + 1])
    : 0.8;

  // Parse --dpi option
  const dpiIndex = args.indexOf('--dpi');
  const dpi = dpiIndex !== -1 && args[dpiIndex + 1]
    ? parseInt(args[dpiIndex + 1])
    : 72;

  // Parse --parallel option
  const parallelIndex = args.indexOf('--parallel');
  const parallel = parallelIndex !== -1 && args[parallelIndex + 1]
    ? parseInt(args[parallelIndex + 1])
    : 1;

  if (threshold < 0 || threshold > 1) {
    console.error('Error: Threshold must be between 0 and 1');
    process.exit(1);
  }

  if (dpi < 10 || dpi > 600) {
    console.error('Error: DPI must be between 10 and 600');
    process.exit(1);
  }

  try {
    await updateIndex(indexPath, verify, forceAll, numPages, threshold, dpi, parallel);
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { updateIndex, titlesMatch, similarityRatio, levenshteinDistance };
