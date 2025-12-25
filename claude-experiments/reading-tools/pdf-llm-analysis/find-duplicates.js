#!/usr/bin/env node

/**
 * Find duplicate PDFs by similar titles
 */

import { readFile, writeFile } from 'fs/promises';

// Levenshtein distance
function levenshtein(a, b) {
  if (!a || !b) return Math.max(a?.length || 0, b?.length || 0);
  const matrix = [];
  for (let i = 0; i <= b.length; i++) matrix[i] = [i];
  for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j] + 1);
      }
    }
  }
  return matrix[b.length][a.length];
}

function similarity(a, b) {
  if (!a || !b) return 0;
  const longer = a.length > b.length ? a : b;
  return (longer.length - levenshtein(a.toLowerCase(), b.toLowerCase())) / longer.length;
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

async function main() {
  const args = process.argv.slice(2);
  const indexPath = args[0] || '../pdf-indexer/pdf-index-fixed.json';
  const threshold = parseFloat(args[1]) || 0.85;

  console.log('Finding duplicates...');
  console.log(`Index: ${indexPath}`);
  console.log(`Similarity threshold: ${threshold}`);
  console.log('');

  const indexData = JSON.parse(await readFile(indexPath, 'utf-8'));

  // Get titles
  const entries = indexData.map((e, i) => ({
    index: i,
    hash: e.hash,
    fileName: e.fileName,
    path: e.path,
    title: getDisplayTitle(e),
    normalized: normalizeTitle(getDisplayTitle(e)),
    totalPages: e.totalPages,
  })).filter(e => e.normalized.length > 5);

  console.log(`Entries with titles: ${entries.length}`);

  // Find exact duplicates first
  const byNormalized = {};
  for (const entry of entries) {
    if (!byNormalized[entry.normalized]) byNormalized[entry.normalized] = [];
    byNormalized[entry.normalized].push(entry);
  }

  const exactDupes = Object.entries(byNormalized).filter(([_, entries]) => entries.length > 1);
  console.log(`\nExact title duplicates: ${exactDupes.length} groups`);

  // Show exact duplicates
  console.log('\n' + '='.repeat(60));
  console.log('EXACT DUPLICATES (same normalized title):');
  console.log('='.repeat(60));

  let totalDupeCount = 0;
  for (const [title, dupes] of exactDupes.slice(0, 20)) {
    console.log(`\n"${dupes[0].title.slice(0, 70)}${dupes[0].title.length > 70 ? '...' : ''}"`);
    for (const d of dupes) {
      const folder = d.path.split('/').slice(-2, -1)[0];
      console.log(`  [${d.totalPages}pp] ${folder}/${d.fileName.slice(0, 40)}`);
    }
    totalDupeCount += dupes.length - 1;
  }

  if (exactDupes.length > 20) {
    console.log(`\n... and ${exactDupes.length - 20} more groups`);
  }

  console.log(`\nTotal duplicate files: ${totalDupeCount}`);

  // Write duplicates to JSON for further processing
  const dupeReport = exactDupes.map(([normalized, dupes]) => ({
    title: dupes[0].title,
    normalized,
    count: dupes.length,
    entries: dupes.map(d => ({
      hash: d.hash,
      path: d.path,
      fileName: d.fileName,
      totalPages: d.totalPages,
    }))
  }));

  await writeFile('duplicates-report.json', JSON.stringify(dupeReport, null, 2));
  console.log('\nFull report written to: duplicates-report.json');
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
