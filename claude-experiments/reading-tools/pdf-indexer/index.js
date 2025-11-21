#!/usr/bin/env node

const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
const crypto = require('crypto');
const { PDFParse } = require('pdf-parse');

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
 * Extract metadata (title and author) from a PDF
 */
async function extractMetadata(filePath) {
  let parser = null;
  try {
    const dataBuffer = await fs.readFile(filePath);
    parser = new PDFParse({ data: dataBuffer });

    const result = await parser.getInfo({ parsePageInfo: false });

    // Try to extract title and author from PDF metadata
    const metadata = result.info || {};
    const title = metadata.Title || null;
    const author = metadata.Author || null;

    // Check if we found the required metadata
    const metadataFound = Boolean(title && author);

    // Get parsed dates
    const dates = result.getDateNode ? result.getDateNode() : {};

    return {
      title,
      author,
      metadataFound,
      totalPages: result.total || null,
      // Additional metadata that might be useful
      creator: metadata.Creator || null,
      producer: metadata.Producer || null,
      creationDate: dates.CreationDate || metadata.CreationDate || null,
      modificationDate: dates.ModDate || metadata.ModDate || null,
    };
  } catch (error) {
    console.error(`Error extracting metadata from ${filePath}:`, error.message);
    return {
      title: null,
      author: null,
      metadataFound: false,
      error: error.message,
    };
  } finally {
    // Clean up the parser
    if (parser) {
      try {
        await parser.destroy();
      } catch (destroyError) {
        // Ignore destroy errors
      }
    }
  }
}

/**
 * Process a single PDF file
 */
async function processPDF(filePath) {
  console.log(`Processing: ${filePath}`);

  try {
    const [hash, metadata] = await Promise.all([
      calculateHash(filePath),
      extractMetadata(filePath)
    ]);

    return {
      hash,
      path: filePath,
      fileName: path.basename(filePath),
      title: metadata.title,
      author: metadata.author,
      metadataFound: metadata.metadataFound,
      totalPages: metadata.totalPages,
      creator: metadata.creator,
      producer: metadata.producer,
      creationDate: metadata.creationDate,
      modificationDate: metadata.modificationDate,
      error: metadata.error || null,
      processedAt: new Date().toISOString(),
    };
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return {
      hash: null,
      path: filePath,
      fileName: path.basename(filePath),
      title: null,
      author: null,
      metadataFound: false,
      error: error.message,
      processedAt: new Date().toISOString(),
    };
  }
}

/**
 * Build index of all PDFs in a directory
 */
async function buildIndex(directory, outputFile = 'pdf-index.json') {
  console.log(`Scanning directory: ${directory}`);
  console.log('');

  // Find all PDFs
  const pdfFiles = await findPDFs(directory);
  console.log(`Found ${pdfFiles.length} PDF files`);
  console.log('');

  if (pdfFiles.length === 0) {
    console.log('No PDF files found.');
    return;
  }

  // Process all PDFs
  const index = [];
  for (let i = 0; i < pdfFiles.length; i++) {
    console.log(`[${i + 1}/${pdfFiles.length}]`);
    const entry = await processPDF(pdfFiles[i]);
    index.push(entry);
    console.log('');
  }

  // Write index to file
  const outputPath = path.join(process.cwd(), outputFile);
  await fs.writeFile(outputPath, JSON.stringify(index, null, 2));

  // Print summary
  const withMetadata = index.filter(e => e.metadataFound).length;
  const withoutMetadata = index.filter(e => !e.metadataFound).length;

  console.log('='.repeat(50));
  console.log('Summary:');
  console.log(`Total PDFs processed: ${index.length}`);
  console.log(`PDFs with metadata: ${withMetadata}`);
  console.log(`PDFs without metadata: ${withoutMetadata}`);
  console.log('');
  console.log(`Index saved to: ${outputPath}`);
  console.log('='.repeat(50));

  return index;
}

/**
 * Main function - CLI interface
 */
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
PDF Indexer - Recursively scan and index PDF files

Usage:
  node index.js <directory> [output-file]

Arguments:
  directory     Directory to scan for PDFs (required)
  output-file   Output JSON file name (default: pdf-index.json)

Example:
  node index.js /path/to/pdfs
  node index.js /path/to/pdfs my-pdfs.json

The tool will:
  1. Recursively find all PDF files in the directory
  2. Calculate SHA256 hash for each PDF
  3. Extract title and author metadata
  4. Create a JSON index with all information
  5. Flag PDFs where metadata couldn't be extracted
    `);
    process.exit(0);
  }

  const directory = args[0];
  const outputFile = args[1] || 'pdf-index.json';

  // Check if directory exists
  try {
    const stats = await fs.stat(directory);
    if (!stats.isDirectory()) {
      console.error(`Error: ${directory} is not a directory`);
      process.exit(1);
    }
  } catch (error) {
    console.error(`Error: Directory ${directory} does not exist`);
    process.exit(1);
  }

  // Build the index
  try {
    await buildIndex(directory, outputFile);
  } catch (error) {
    console.error('Error building index:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

// Export functions for potential use as a module
module.exports = {
  findPDFs,
  calculateHash,
  extractMetadata,
  processPDF,
  buildIndex,
};
