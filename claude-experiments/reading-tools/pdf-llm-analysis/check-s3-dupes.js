import fs from "fs";
import { execSync } from "child_process";

const index = JSON.parse(fs.readFileSync("../pdf-indexer/pdf-index-fixed.json", "utf8"));

// Build hash lookup
const byHash = {};
for (const e of index) {
  byHash[e.hash] = e;
}

// Get S3 hashes
const s3Output = execSync("aws s3 ls s3://jimmyhmiller-bucket/pdfs/ --recursive").toString();
const s3Hashes = s3Output.split("\n")
  .filter(line => line.includes(".pdf"))
  .map(line => {
    const match = line.match(/pdfs\/([^/]+)\/([a-f0-9]+)\.pdf/);
    if (match) return { folder: match[1], hash: match[2] };
    return null;
  })
  .filter(Boolean);

console.log("S3 PDFs:", s3Hashes.length);

const excluded = ["garbage", "not-article", "books", "ai"];
let inIndex = 0;
let notInIndex = 0;
let isDuplicate = 0;
let isExcludedFolder = 0;
let shouldShow = 0;

for (const s3 of s3Hashes) {
  if (excluded.includes(s3.folder)) {
    isExcludedFolder++;
    continue;
  }

  const entry = byHash[s3.hash];
  if (!entry) {
    notInIndex++;
  } else {
    inIndex++;
    if (entry.is_duplicate === true) {
      isDuplicate++;
    } else {
      shouldShow++;
    }
  }
}

console.log("In excluded folders:", isExcludedFolder);
console.log("In index:", inIndex);
console.log("Not in index:", notInIndex);
console.log("Marked as duplicate:", isDuplicate);
console.log("Should show:", shouldShow);
