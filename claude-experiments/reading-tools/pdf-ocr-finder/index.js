#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const PDFParser = require('pdf-parse');

let MIN_TEXT_LENGTH = 100; // PDFs with less than this many characters are considered OCR scans

async function findPDFs(directory) {
  const pdfs = [];

  async function traverse(dir) {
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          await traverse(fullPath);
        } else if (path.extname(entry.name).toLowerCase() === '.pdf') {
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

async function hasExtractableText(filePath) {
  try {
    const dataBuffer = await fs.readFile(filePath);
    const data = await PDFParser(dataBuffer);

    // Extract text content
    const text = data.text || '';

    // Remove whitespace and count actual characters
    const cleanText = text.replace(/\s+/g, '');

    return {
      hasText: cleanText.length >= MIN_TEXT_LENGTH,
      textLength: cleanText.length,
      totalPages: data.numpages || 0
    };
  } catch (error) {
    console.error(`Error parsing PDF ${filePath}:`, error.message);
    return { hasText: null, textLength: 0, totalPages: 0, error: error.message };
  }
}

async function copyPDF(sourcePath, targetDir) {
  try {
    // Ensure target directory exists
    await fs.mkdir(targetDir, { recursive: true });

    const fileName = path.basename(sourcePath);
    const targetPath = path.join(targetDir, fileName);

    // Check if file already exists in target
    let finalTargetPath = targetPath;
    let counter = 1;
    while (true) {
      try {
        await fs.access(finalTargetPath);
        // File exists, try with counter
        const ext = path.extname(fileName);
        const base = path.basename(fileName, ext);
        finalTargetPath = path.join(targetDir, `${base}_${counter}${ext}`);
        counter++;
      } catch {
        // File doesn't exist, we can use this path
        break;
      }
    }

    await fs.copyFile(sourcePath, finalTargetPath);
    return finalTargetPath;
  } catch (error) {
    console.error(`Error copying PDF ${sourcePath}:`, error.message);
    return null;
  }
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
PDF OCR Finder - Find and move PDFs without extractable text

Usage:
  pdf-ocr-finder <input-directory> [options]

Options:
  --output <dir>          Directory to copy OCR PDFs to (default: ./ocr-scans)
  --min-text <number>     Minimum characters to consider as having text (default: 100)
  --dry-run               Show what would be copied without actually copying
  --list-only             Only list OCR PDFs without copying them
  --help, -h              Show this help message

Examples:
  pdf-ocr-finder ~/Documents/PDFs
  pdf-ocr-finder ~/Documents/PDFs --output ~/OCR-Scans
  pdf-ocr-finder ~/Documents/PDFs --dry-run
  pdf-ocr-finder ~/Documents/PDFs --list-only --min-text 50
`);
    process.exit(0);
  }

  const inputDir = args[0];
  const outputDir = args.includes('--output')
    ? args[args.indexOf('--output') + 1]
    : path.join(process.cwd(), 'ocr-scans');
  const dryRun = args.includes('--dry-run');
  const listOnly = args.includes('--list-only');

  if (args.includes('--min-text')) {
    const minTextIndex = args.indexOf('--min-text');
    const minTextValue = parseInt(args[minTextIndex + 1], 10);
    if (!isNaN(minTextValue)) {
      MIN_TEXT_LENGTH = minTextValue;
    }
  }

  console.log(`Scanning for PDFs in: ${inputDir}`);
  console.log(`Minimum text length: ${MIN_TEXT_LENGTH} characters`);
  if (!listOnly) {
    console.log(`Output directory: ${outputDir}`);
  }
  if (dryRun) {
    console.log('DRY RUN - No files will be moved');
  }
  if (listOnly) {
    console.log('LIST ONLY - No files will be moved');
  }
  console.log('');

  const pdfs = await findPDFs(inputDir);
  console.log(`Found ${pdfs.length} PDF files\n`);

  const ocrPDFs = [];
  let processed = 0;
  let copied = 0;

  for (const pdfPath of pdfs) {
    processed++;
    const result = await hasExtractableText(pdfPath);

    if (result.hasText === false) {
      ocrPDFs.push({ path: pdfPath, ...result });
      console.log(`[${processed}/${pdfs.length}] OCR SCAN: ${pdfPath}`);
      console.log(`  Text length: ${result.textLength} characters, Pages: ${result.totalPages}`);

      // Copy immediately if not in list-only mode
      if (!listOnly) {
        if (dryRun) {
          console.log(`  [DRY RUN] Would copy to: ${outputDir}`);
        } else {
          const newPath = await copyPDF(pdfPath, outputDir);
          if (newPath) {
            copied++;
            console.log(`  ✓ Copied to: ${newPath}`);
          }
        }
      }
    } else if (result.hasText === null) {
      console.log(`[${processed}/${pdfs.length}] ERROR: ${pdfPath}`);
      console.log(`  ${result.error}`);
    } else {
      console.log(`[${processed}/${pdfs.length}] Has text: ${pdfPath} (${result.textLength} chars)`);
    }
  }

  console.log(`\n=== Summary ===`);
  console.log(`Total PDFs scanned: ${pdfs.length}`);
  console.log(`OCR scans found: ${ocrPDFs.length}`);

  if (!listOnly && !dryRun && copied > 0) {
    console.log(`Successfully copied: ${copied} PDFs`);
  }

  if (ocrPDFs.length > 0 && listOnly) {
    console.log(`\nOCR PDF list:`);
    ocrPDFs.forEach((pdf, i) => {
      console.log(`${i + 1}. ${pdf.path}`);
    });
  }
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
