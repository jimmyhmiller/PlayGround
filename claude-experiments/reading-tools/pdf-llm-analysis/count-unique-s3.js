import fs from "fs";
import { execSync } from "child_process";

const index = JSON.parse(fs.readFileSync("../pdf-indexer/pdf-index-fixed.json", "utf8"));

// Get S3 hashes
const s3Output = execSync("aws s3 ls s3://jimmyhmiller-bucket/pdfs/ --recursive", { encoding: 'utf8' });
const s3Entries = [];
for (const line of s3Output.split("\n")) {
  const match = line.match(/pdfs\/([^/]+)\/([a-f0-9]{64})\.pdf/);
  if (match) s3Entries.push({ folder: match[1], hash: match[2] });
}

const excluded = ["garbage", "not-article", "books", "ai"];
const s3NonExcluded = s3Entries.filter(e => !excluded.includes(e.folder));

console.log("S3 files:", s3Entries.length);
console.log("S3 non-excluded:", s3NonExcluded.length);

// How many of these are non-duplicates in index?
const nonDupeHashes = new Set(
  index.filter(e => e.is_duplicate !== true).map(e => e.hash)
);

console.log("Non-duplicate hashes in index:", nonDupeHashes.size);

const s3NonDupes = s3NonExcluded.filter(e => nonDupeHashes.has(e.hash));
console.log("S3 non-excluded & non-duplicate:", s3NonDupes.length);

// Show breakdown by folder
const byFolder = {};
for (const e of s3NonDupes) {
  byFolder[e.folder] = (byFolder[e.folder] || 0) + 1;
}
console.log("\nBy folder:");
Object.entries(byFolder).sort((a,b) => b[1]-a[1]).forEach(([f, c]) => console.log(`  ${f}: ${c}`));
