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
 * Transliterate common non-ASCII characters to ASCII equivalents
 */
function transliterateToAscii(text) {
  const charMap = {
    // Latin characters with diacritics
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae',
    'ç': 'c',
    'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
    'ñ': 'n',
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o', 'œ': 'oe',
    'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
    'ý': 'y', 'ÿ': 'y',
    'ß': 'ss',
    // Add more as needed
  };

  return text.split('').map(char => charMap[char] || char).join('');
}

/**
 * Sanitize a string for use in filename
 * - Convert non-ASCII to ASCII equivalents
 * - Convert to lowercase
 * - Replace spaces with hyphens
 * - Remove or replace special characters
 * - Remove leading articles (the, a, an)
 */
function sanitizeForFilename(text) {
  if (!text) return '';

  // Transliterate non-ASCII characters to ASCII
  let sanitized = transliterateToAscii(text);

  // Convert to lowercase
  sanitized = sanitized.toLowerCase();

  // Replace em dashes, en dashes with regular hyphens
  sanitized = sanitized.replace(/[—–]/g, '-');

  // Replace colons, slashes, etc with hyphens
  sanitized = sanitized.replace(/[:\/\\|]/g, '-');

  // Remove quotes, parentheses, brackets, and other punctuation
  sanitized = sanitized.replace(/["""''(){}\[\]<>.,;!?*&^%$#@]/g, '');

  // Replace multiple spaces/hyphens with single hyphen
  sanitized = sanitized.replace(/[\s_]+/g, '-');
  sanitized = sanitized.replace(/-+/g, '-');

  // Remove leading/trailing hyphens
  sanitized = sanitized.replace(/^-+|-+$/g, '');

  return sanitized;
}

/**
 * Remove leading articles from title for better sorting
 */
function removeLeadingArticles(title) {
  if (!title) return '';

  // Remove "the ", "a ", "an " from the beginning (case insensitive)
  return title.replace(/^(the|a|an)\s+/i, '');
}

/**
 * Format author name as lastname-firstname
 * Handles various formats:
 * - "FirstName LastName" -> "lastname-firstname"
 * - "LastName, FirstName" -> "lastname-firstname"
 * - "F. LastName" -> "lastname-f"
 * - Multiple authors -> "lastname1-firstname1-lastname2-firstname2"
 */
function formatAuthorName(author) {
  if (!author) return 'unknown-author';

  // Handle multiple authors separated by "and"
  const authors = author.split(/\s+and\s+/i);

  // Take up to 2 authors to avoid super long filenames
  const formattedAuthors = authors.slice(0, 2).map(a => {
    a = a.trim();

    // Handle "LastName, FirstName" format
    if (a.includes(',')) {
      const [last, first] = a.split(',').map(s => s.trim());
      return sanitizeForFilename(`${last}-${first}`);
    }

    // Handle "FirstName LastName" or "F. LastName" format
    const parts = a.split(/\s+/);
    if (parts.length >= 2) {
      const last = parts[parts.length - 1];
      const first = parts.slice(0, -1).join('-');
      return sanitizeForFilename(`${last}-${first}`);
    }

    // Single name
    return sanitizeForFilename(a);
  });

  // If more than 2 authors, add "et-al"
  if (authors.length > 2) {
    formattedAuthors.push('et-al');
  }

  return formattedAuthors.join('-');
}

/**
 * Generate new filename from title and author
 * Format: lastname-firstname--title-words.pdf
 */
function generateFilename(title, author, maxLength = 120) {
  const authorPart = formatAuthorName(author);
  const titlePart = sanitizeForFilename(removeLeadingArticles(title || 'untitled'));

  let filename = `${authorPart}--${titlePart}`;

  // Truncate if too long (leave room for .pdf extension)
  if (filename.length > maxLength - 4) {
    filename = filename.substring(0, maxLength - 4);
    // Remove trailing hyphen if we cut mid-word
    filename = filename.replace(/-+$/, '');
  }

  return `${filename}.pdf`;
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
 * Main rename function
 */
async function renamePDFs(indexPath, targetDirectory, options = {}) {
  const { dryRun = false } = options;

  console.log(`Loading index from: ${indexPath}`);
  const indexData = JSON.parse(await fs.readFile(indexPath, 'utf-8'));

  // Create hash map for quick lookup
  const indexByHash = new Map();
  for (const entry of indexData) {
    indexByHash.set(entry.hash, entry);
  }

  console.log(`Index loaded: ${indexByHash.size} entries`);
  console.log(`Scanning directory: ${targetDirectory}`);

  const pdfFiles = await findPDFs(targetDirectory);
  console.log(`Found ${pdfFiles.length} PDF files`);
  console.log('');

  const stats = {
    total: pdfFiles.length,
    renamed: 0,
    skipped: 0,
    errors: 0,
  };

  for (const pdfPath of pdfFiles) {
    try {
      // Calculate hash of the file
      const hash = await calculateHash(pdfPath);

      // Look up in index
      const indexEntry = indexByHash.get(hash);

      if (!indexEntry) {
        console.log(`⚠️  SKIP: ${path.basename(pdfPath)} (not in index)`);
        stats.skipped++;
        continue;
      }

      // Check if we have title/author info
      const title = indexEntry.ocr_title || indexEntry.title;
      const author = indexEntry.ocr_author || indexEntry.author;

      if (!title && !author) {
        console.log(`⚠️  SKIP: ${path.basename(pdfPath)} (no title/author in index)`);
        stats.skipped++;
        continue;
      }

      // Generate new filename
      const newFilename = generateFilename(title, author);
      const newPath = path.join(path.dirname(pdfPath), newFilename);

      // Check if file already has the correct name
      if (path.basename(pdfPath) === newFilename) {
        console.log(`✓ OK: ${newFilename} (already correct)`);
        continue;
      }

      // Check if target filename already exists
      if (fsSync.existsSync(newPath) && newPath !== pdfPath) {
        console.log(`⚠️  SKIP: ${path.basename(pdfPath)} -> ${newFilename} (target exists)`);
        stats.skipped++;
        continue;
      }

      if (dryRun) {
        console.log(`🔍 DRY-RUN: ${path.basename(pdfPath)}`);
        console.log(`          -> ${newFilename}`);
        stats.renamed++;
      } else {
        await fs.rename(pdfPath, newPath);
        console.log(`✓ RENAMED: ${path.basename(pdfPath)}`);
        console.log(`          -> ${newFilename}`);
        stats.renamed++;
      }

    } catch (error) {
      console.error(`❌ ERROR: ${path.basename(pdfPath)}: ${error.message}`);
      stats.errors++;
    }
  }

  console.log('');
  console.log('=== Summary ===');
  console.log(`Total files: ${stats.total}`);
  console.log(`Renamed: ${stats.renamed}`);
  console.log(`Skipped: ${stats.skipped}`);
  console.log(`Errors: ${stats.errors}`);

  if (dryRun) {
    console.log('');
    console.log('This was a dry run. No files were actually renamed.');
    console.log('Run without --dry-run to perform the actual renames.');
  }
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
PDF Renamer - Rename PDFs based on index metadata

Usage:
  node rename-pdfs.js [options]

Options:
  --index <path>      Path to pdf-index.json (default: ./pdf-index.json)
  --dir <path>        Directory containing PDFs to rename (default: from index)
  --dry-run           Preview changes without actually renaming files
  -h, --help          Show this help message

Examples:
  # Dry run to preview changes
  node rename-pdfs.js --dry-run

  # Rename files for real
  node rename-pdfs.js

  # Custom index and directory
  node rename-pdfs.js --index ./my-index.json --dir /path/to/pdfs
`);
    process.exit(0);
  }

  const indexPath = args.includes('--index')
    ? args[args.indexOf('--index') + 1]
    : path.join(__dirname, '..', 'pdf-indexer', 'pdf-index.json');

  const dryRun = args.includes('--dry-run');

  // Default to the directory from the first entry in index
  let targetDirectory = args.includes('--dir')
    ? args[args.indexOf('--dir') + 1]
    : null;

  // If no directory specified, read from index
  if (!targetDirectory) {
    const indexData = JSON.parse(fsSync.readFileSync(indexPath, 'utf-8'));
    if (indexData.length > 0) {
      targetDirectory = path.dirname(indexData[0].path);
    } else {
      console.error('Error: No directory specified and index is empty');
      process.exit(1);
    }
  }

  renamePDFs(indexPath, targetDirectory, { dryRun })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

module.exports = { generateFilename, formatAuthorName, sanitizeForFilename };
