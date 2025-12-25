import fs from "fs";
const index = JSON.parse(fs.readFileSync("../pdf-indexer/pdf-index-fixed.json", "utf8"));

const nonDupes = index.filter(e => e.is_duplicate !== true);
console.log("Non-duplicate entries:", nonDupes.length);

// Check excluded categories
const excluded = ["garbage", "not-article", "books", "ai"];
let excludedCount = 0;
const byFolder = {};

for (const e of nonDupes) {
  const parts = e.path.split("/");
  const folder = parts[parts.length - 2] || "unknown";
  byFolder[folder] = (byFolder[folder] || 0) + 1;
  if (excluded.includes(folder)) excludedCount++;
}

console.log("In excluded categories:", excludedCount);
console.log("After category exclusion:", nonDupes.length - excludedCount);

console.log("\nBy folder:");
Object.entries(byFolder).sort((a,b) => b[1] - a[1]).forEach(([f, c]) => {
  const ex = excluded.includes(f) ? " (EXCLUDED)" : "";
  console.log(`  ${f}: ${c}${ex}`);
});
