import fs from "fs";
import { execSync } from "child_process";

const index = JSON.parse(fs.readFileSync("../pdf-indexer/pdf-index-fixed.json", "utf8"));

// Get S3 hashes
const s3Output = execSync("aws s3 ls s3://jimmyhmiller-bucket/pdfs/ --recursive", { encoding: 'utf8' });
const s3Hashes = new Set();
for (const line of s3Output.split("\n")) {
  const match = line.match(/([a-f0-9]{64})\.pdf/);
  if (match) s3Hashes.add(match[1]);
}

console.log("S3 hashes:", s3Hashes.size);

// Find entries marked as duplicate that are in S3
const s3Dupes = index.filter(e => e.is_duplicate === true && s3Hashes.has(e.hash));
console.log("S3 entries marked as duplicate:", s3Dupes.length);

// For each S3 dupe, find what it's a duplicate OF
console.log("\nSample S3 duplicates:");
for (const dupe of s3Dupes.slice(0, 10)) {
  const original = index.find(e => e.hash === dupe.duplicate_of);
  const origInS3 = original && s3Hashes.has(original.hash);
  console.log(`\n  Dupe: ${dupe.preferred_title || dupe.ocr_title || dupe.title}`);
  console.log(`    hash: ${dupe.hash.slice(0, 16)}... (in S3: true)`);
  if (original) {
    console.log(`  Original: ${original.preferred_title || original.ocr_title || original.title}`);
    console.log(`    hash: ${original.hash.slice(0, 16)}... (in S3: ${origInS3})`);
  } else {
    console.log(`  Original: NOT FOUND (hash: ${dupe.duplicate_of?.slice(0, 16)})`);
  }
}
