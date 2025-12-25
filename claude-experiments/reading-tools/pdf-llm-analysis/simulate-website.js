import fs from "fs";
import { execSync } from "child_process";

// Simulate exactly what the website does

// 1. Load index (same logic as readings.js)
const indexPath = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/reading-tools/pdf-indexer/pdf-index-fixed.json";
const data = JSON.parse(fs.readFileSync(indexPath, "utf8"));
const pdfIndex = {};
for (const entry of data) {
  if (entry.hash) {
    const existing = pdfIndex[entry.hash];
    const isDupe = entry.is_duplicate || false;

    // If we already have an entry for this hash, prefer the non-duplicate one
    if (existing && !existing.isDuplicate && isDupe) {
      continue; // Keep the existing non-duplicate entry
    }

    pdfIndex[entry.hash] = {
      title: entry.preferred_title || entry.ocr_title || entry.title || null,
      author: entry.preferred_author || entry.ocr_author || entry.author || null,
      pageCount: entry.totalPages || null,
      creationDate: entry.creationDate || null,
      isDuplicate: isDupe,
    };
  }
}

console.log("Index entries:", Object.keys(pdfIndex).length);
console.log("Index entries with isDuplicate=true:", Object.values(pdfIndex).filter(e => e.isDuplicate === true).length);
console.log("Index entries with isDuplicate=false:", Object.values(pdfIndex).filter(e => e.isDuplicate === false).length);

// 2. Get S3 PDFs
const s3Output = execSync("aws s3 ls s3://jimmyhmiller-bucket/pdfs/ --recursive", { encoding: 'utf8' });
const s3PDFs = [];
for (const line of s3Output.split("\n")) {
  const match = line.match(/pdfs\/([^/]+)\/([a-f0-9]{64})\.pdf/);
  if (match) s3PDFs.push({ category: match[1], hash: match[2] });
}

console.log("\nS3 PDFs:", s3PDFs.length);

// 3. Filter like the website does
const excludeCategories = ['garbage', 'not-article', 'books', 'ai'];
let skippedCategory = 0;
let skippedDuplicate = 0;
let notInIndex = 0;
let showing = 0;

for (const pdf of s3PDFs) {
  if (excludeCategories.includes(pdf.category)) {
    skippedCategory++;
    continue;
  }

  const metadata = pdfIndex[pdf.hash] || {};

  if (metadata.isDuplicate) {
    skippedDuplicate++;
    continue;
  }

  if (!pdfIndex[pdf.hash]) {
    notInIndex++;
  }

  showing++;
}

console.log("\nFiltering:");
console.log("  Skipped (excluded category):", skippedCategory);
console.log("  Skipped (duplicate):", skippedDuplicate);
console.log("  Not in index:", notInIndex);
console.log("  SHOWING:", showing);
