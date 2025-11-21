#!/usr/bin/env node

import { readFile, writeFile, mkdir, rename } from 'fs/promises';
import { dirname, join, basename } from 'path';
import { classifyDocumentType } from './extract-metadata.js';

/**
 * Classify PDFs and move non-articles to a separate folder
 * @param {string} indexPath - Path to index JSON file
 * @param {string} targetFolder - Folder name for non-articles (default: 'not-article')
 * @param {number} numPages - Number of pages to analyze per PDF
 * @param {number} dpi - Image resolution in DPI (lower = faster)
 */
async function classifyAndMove(indexPath, targetFolder = 'not-article', numPages = 3, dpi = 72) {
  console.log('='.repeat(60));
  console.log('Configuration:');
  console.log(`  Index file: ${indexPath}`);
  console.log(`  Target folder: ${targetFolder}`);
  console.log(`  Pages per PDF: ${numPages}`);
  console.log(`  Image resolution: ${dpi} DPI`);
  console.log('='.repeat(60));
  console.log('');

  // Read the index file
  const indexData = JSON.parse(await readFile(indexPath, 'utf-8'));

  if (!Array.isArray(indexData)) {
    throw new Error('Index file is not an array');
  }

  console.log(`Found ${indexData.length} PDFs in index`);

  // Filter PDFs without metadata and without ocr_title (the ones that need classification)
  const toClassify = indexData.filter(entry => {
    return !entry.metadataFound && (entry.ocr_title === undefined || entry.ocr_title === null);
  });

  console.log(`${toClassify.length} PDFs need classification`);
  console.log('');

  let classifiedCount = 0;
  let movedCount = 0;
  let errorCount = 0;
  const classifications = { paper: 0, article: 0, book: 0, other: 0 };

  // Process each PDF
  for (let i = 0; i < toClassify.length; i++) {
    const entry = toClassify[i];
    const pdfIndex = indexData.indexOf(entry);

    console.log(`[${i + 1}/${toClassify.length}] Classifying: ${entry.fileName}`);
    console.log(`  Path: ${entry.path}`);

    try {
      // Classify the document
      const classification = await classifyDocumentType(entry.path, numPages, true, dpi);

      console.log(`  Type: ${classification.type}`);
      console.log(`  Confidence: ${classification.confidence}`);

      // Store classification in index
      indexData[pdfIndex].document_type = classification.type;
      indexData[pdfIndex].type_confidence = classification.confidence;
      classifiedCount++;
      classifications[classification.type]++;

      // If it's "other", move to target folder
      if (classification.type === 'other') {
        const baseDir = dirname(entry.path);
        const fileName = basename(entry.path);
        const targetDir = join(baseDir, targetFolder);
        const newPath = join(targetDir, fileName);

        console.log(`  Action: Moving to ${targetFolder}/`);

        // Create target directory if it doesn't exist
        await mkdir(targetDir, { recursive: true });

        // Move the file
        await rename(entry.path, newPath);

        // Update the path in the index
        indexData[pdfIndex].path = newPath;
        indexData[pdfIndex].moved_to_not_article = true;
        movedCount++;

        console.log(`  Moved to: ${newPath}`);
      } else {
        console.log(`  Action: Keeping in current location (type: ${classification.type})`);
      }

      // Save progress after each update
      await writeFile(indexPath, JSON.stringify(indexData, null, 2));

    } catch (error) {
      console.error(`  Error: ${error.message}`);
      console.error(`  Skipping this PDF`);
      errorCount++;
    }

    console.log('');
  }

  // Print summary
  console.log('='.repeat(60));
  console.log('Summary:');
  console.log(`Total PDFs in index: ${indexData.length}`);
  console.log(`PDFs classified: ${classifiedCount}`);
  console.log('');
  console.log('Classifications:');
  console.log(`  Papers: ${classifications.paper}`);
  console.log(`  Articles: ${classifications.article}`);
  console.log(`  Books: ${classifications.book}`);
  console.log(`  Other: ${classifications.other}`);
  console.log('');
  console.log(`PDFs moved to ${targetFolder}/: ${movedCount}`);

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
PDF Classifier and Mover

Usage:
  node classify-and-move.js <index-file> [options]

Arguments:
  index-file    Path to index JSON file (required)

Options:
  --folder NAME Folder name for non-articles (default: 'not-article')
  --pages N     Number of pages to analyze per PDF (default: 3)
  --dpi N       Image resolution in DPI (default: 72, lower = faster)

Examples:
  # Classify PDFs without metadata and move "other" types
  node classify-and-move.js ../pdf-indexer/pdf-index.json

  # Use custom folder name
  node classify-and-move.js ../pdf-indexer/pdf-index.json --folder misc

  # Use 5 pages for better accuracy
  node classify-and-move.js ../pdf-indexer/pdf-index.json --pages 5

  # Use lower resolution for faster processing
  node classify-and-move.js ../pdf-indexer/pdf-index.json --dpi 50

The tool will:
  1. Read the existing index file
  2. Find PDFs without metadata and without ocr_title
  3. Classify each PDF as paper/article/book/other
  4. Move PDFs classified as "other" to a separate folder
  5. Update the index with classifications and new paths
    `);
    process.exit(0);
  }

  const indexPath = args[0];

  // Parse --folder option
  const folderIndex = args.indexOf('--folder');
  const targetFolder = folderIndex !== -1 && args[folderIndex + 1]
    ? args[folderIndex + 1]
    : 'not-article';

  // Parse --pages option
  const pagesIndex = args.indexOf('--pages');
  const numPages = pagesIndex !== -1 && args[pagesIndex + 1]
    ? parseInt(args[pagesIndex + 1])
    : 3;

  // Parse --dpi option
  const dpiIndex = args.indexOf('--dpi');
  const dpi = dpiIndex !== -1 && args[dpiIndex + 1]
    ? parseInt(args[dpiIndex + 1])
    : 72;

  if (dpi < 10 || dpi > 600) {
    console.error('Error: DPI must be between 10 and 600');
    process.exit(1);
  }

  try {
    await classifyAndMove(indexPath, targetFolder, numPages, dpi);
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

export { classifyAndMove };
