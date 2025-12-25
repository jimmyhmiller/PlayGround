#!/usr/bin/env node

/**
 * Deduplicate PDF index by merging entries with same/similar titles.
 * Picks the "best" entry from each duplicate group.
 */

import { readFile, writeFile } from 'fs/promises';
import { execSync } from 'child_process';

// Get S3 hashes
function getS3Hashes() {
  try {
    const s3Output = execSync("aws s3 ls s3://jimmyhmiller-bucket/pdfs/ --recursive", { encoding: 'utf8' });
    const hashes = new Set();
    for (const line of s3Output.split("\n")) {
      const match = line.match(/([a-f0-9]{64})\.pdf/);
      if (match) hashes.add(match[1]);
    }
    return hashes;
  } catch (e) {
    console.error("Warning: Could not fetch S3 hashes:", e.message);
    return new Set();
  }
}

function getDisplayTitle(entry) {
  return entry.preferred_title || entry.ocr_title || entry.title || entry.fileName || '';
}

function normalizeTitle(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Score an entry - higher is better
 * Prefers: in S3, more pages, better metadata, certain folders
 */
function scoreEntry(entry, s3Hashes) {
  let score = 0;

  // STRONGLY prefer entries that are in S3 (most important!)
  if (s3Hashes.has(entry.hash)) score += 10000;

  // Prefer entries with more pages (likely more complete)
  score += entry.totalPages || 0;

  // Prefer entries with preferred_title set
  if (entry.preferred_title) score += 100;

  // Prefer entries with good metadata
  if (entry.preferred_source === 'embedded') score += 50;
  if (entry.preferred_source === 'ocr') score += 30;

  // Prefer certain folders
  const path = entry.path || '';
  if (path.includes('/foc-covered/')) score += 200;  // Already reviewed
  if (path.includes('/interesting/')) score += 150;
  if (path.includes('/historical/')) score += 100;
  if (path.includes('/foc-potential/')) score += 50;

  // Prefer cleaner filenames (not hash-based)
  if (!entry.fileName.match(/^[a-f0-9]{20,}/i)) score += 20;

  return score;
}

async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Deduplicate PDF Index

Usage:
  node deduplicate-index.js <index-file> [options]

Options:
  --dry-run     Show what would be done without making changes
  --output FILE Write deduplicated index to FILE (default: overwrite input)

The script will:
  1. Group entries by normalized title
  2. Pick the "best" entry from each duplicate group
  3. Mark other entries with is_duplicate=true and duplicate_of=<hash>
  4. Output the updated index
    `);
    process.exit(0);
  }

  const indexPath = args[0] || '../pdf-indexer/pdf-index-fixed.json';
  const dryRun = args.includes('--dry-run');

  const outputIndex = args.indexOf('--output');
  const outputPath = outputIndex !== -1 ? args[outputIndex + 1] : indexPath;

  console.log('='.repeat(60));
  console.log('Deduplicate PDF Index');
  console.log('='.repeat(60));
  console.log(`Input: ${indexPath}`);
  console.log(`Output: ${outputPath}`);
  console.log(`Dry run: ${dryRun}`);
  console.log('');

  const indexData = JSON.parse(await readFile(indexPath, 'utf-8'));
  console.log(`Total entries: ${indexData.length}`);

  // Get S3 hashes to prefer entries that are already uploaded
  console.log('Fetching S3 hashes...');
  const s3Hashes = getS3Hashes();
  console.log(`S3 PDFs: ${s3Hashes.size}`);
  console.log('');

  // Group by normalized title
  const byTitle = {};
  for (let i = 0; i < indexData.length; i++) {
    const entry = indexData[i];
    entry._index = i;  // Track original index
    const title = normalizeTitle(getDisplayTitle(entry));
    if (!title || title.length < 5) continue;

    if (!byTitle[title]) byTitle[title] = [];
    byTitle[title].push(entry);
  }

  // Process duplicates
  let dupeGroupCount = 0;
  let markedAsDupe = 0;

  for (const [title, entries] of Object.entries(byTitle)) {
    if (entries.length <= 1) continue;

    dupeGroupCount++;

    // Score each entry and pick the best
    const scored = entries.map(e => ({ entry: e, score: scoreEntry(e, s3Hashes) }));
    scored.sort((a, b) => b.score - a.score);

    const best = scored[0].entry;

    if (dryRun && dupeGroupCount <= 10) {
      console.log(`\nGroup: "${getDisplayTitle(best).slice(0, 50)}..."`);
      console.log(`  Keep: ${best.fileName} (score: ${scored[0].score})`);
      for (let i = 1; i < scored.length; i++) {
        console.log(`  Dupe: ${scored[i].entry.fileName} (score: ${scored[i].score})`);
      }
    }

    // Mark duplicates
    for (let i = 1; i < scored.length; i++) {
      const dupe = scored[i].entry;
      indexData[dupe._index].is_duplicate = true;
      indexData[dupe._index].duplicate_of = best.hash;
      markedAsDupe++;
    }
  }

  console.log(`\nDuplicate groups: ${dupeGroupCount}`);
  console.log(`Entries marked as duplicate: ${markedAsDupe}`);
  console.log(`Unique entries: ${indexData.length - markedAsDupe}`);

  if (dryRun) {
    console.log('\nDry run - no changes made');
    if (dupeGroupCount > 10) {
      console.log(`(Showing first 10 of ${dupeGroupCount} groups)`);
    }
  } else {
    // Clean up temp fields
    for (const entry of indexData) {
      delete entry._index;
    }

    await writeFile(outputPath, JSON.stringify(indexData, null, 2));
    console.log(`\nIndex updated: ${outputPath}`);
  }
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
