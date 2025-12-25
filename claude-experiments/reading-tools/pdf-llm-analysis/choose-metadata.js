#!/usr/bin/env node

/**
 * Compare embedded metadata vs OCR-extracted metadata using LLM.
 * For each PDF with both, ask the LLM which title/author pair is better.
 */

import { readFile, writeFile } from 'fs/promises';
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed'
});

/**
 * Retry function with exponential backoff
 */
async function withRetry(fn, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      const isRetryable = ['ECONNREFUSED', 'ECONNRESET', 'ETIMEDOUT'].some(
        code => error.message?.includes(code) || error.code === code
      );

      if (!isRetryable || attempt === maxRetries - 1) {
        throw error;
      }

      const delay = Math.pow(2, attempt) * 1000;
      console.log(`  Retry ${attempt + 1}/${maxRetries} in ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

/**
 * Ask LLM to choose between two metadata pairs
 */
async function chooseMetadata(embedded, ocr) {
  const prompt = `You are comparing two sets of metadata for the same PDF document. Choose which one is better.

EMBEDDED METADATA (from PDF file):
  Title: "${embedded.title || '(none)'}"
  Author: "${embedded.author || '(none)'}"

OCR METADATA (extracted from page images):
  Title: "${ocr.title || '(none)'}"
  Author: "${ocr.author || '(none)'}"

Rules for choosing:
1. Prefer metadata that has BOTH title and author over one that's missing either
2. Prefer descriptive titles over generic ones (e.g., "Machine Learning Survey" is better than "Document1" or "Microsoft Word - paper.doc")
3. Prefer author names over "Unknown", empty, or software names (e.g., "Adobe Acrobat")
4. If the title looks like a filename, path, or software-generated string, prefer the other
5. If both are roughly equal quality, prefer "embedded" (it's the original)
6. If both are bad/missing, return "neither"

Respond with ONLY a JSON object:
{
  "choice": "embedded" | "ocr" | "neither",
  "reason": "brief explanation"
}`;

  const response = await withRetry(async () => {
    return await client.chat.completions.create({
      model: 'qwen-vl',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 200
    });
  });

  const text = response.choices[0]?.message?.content || '';

  // Extract JSON from response
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error('No JSON in response');
  }

  return JSON.parse(jsonMatch[0]);
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
Choose Best Metadata using LLM

Usage:
  node choose-metadata.js <index-file> [options]

Options:
  --dry-run      Show what would be processed without making changes
  --parallel N   Process N entries in parallel (default: 1)

The script will:
  1. Find PDFs that have BOTH embedded metadata AND ocr_title/ocr_author
  2. Ask the LLM to choose which metadata pair is better
  3. Store the decision in "preferred_source" field ("embedded", "ocr", or "neither")
  4. Add "preferred_title" and "preferred_author" fields with the chosen values
    `);
    process.exit(0);
  }

  const indexPath = args[0];
  const dryRun = args.includes('--dry-run');

  // Parse --parallel option
  const parallelIndex = args.indexOf('--parallel');
  const parallel = parallelIndex !== -1 && args[parallelIndex + 1]
    ? parseInt(args[parallelIndex + 1])
    : 1;

  console.log('='.repeat(60));
  console.log('Choose Best Metadata');
  console.log('='.repeat(60));
  console.log(`Index file: ${indexPath}`);
  console.log(`Dry run: ${dryRun}`);
  console.log(`Parallel workers: ${parallel}`);
  console.log('');

  // Read index
  const indexData = JSON.parse(await readFile(indexPath, 'utf-8'));

  if (!Array.isArray(indexData)) {
    throw new Error('Index file is not an array');
  }

  // Find entries that have both embedded and OCR metadata
  const toProcess = indexData.filter(entry => {
    // Must have OCR data
    if (!entry.ocr_title && !entry.ocr_author) return false;

    // Must have embedded metadata OR we want to compare even if embedded is empty
    // Actually, let's process any entry that has ocr_title, so we can set preferred_*
    return entry.ocr_title !== undefined;
  }).filter(entry => {
    // Skip if already processed
    return entry.preferred_source === undefined;
  });

  console.log(`Total entries: ${indexData.length}`);
  console.log(`Entries to process: ${toProcess.length}`);
  console.log('');

  if (dryRun) {
    console.log('Dry run - showing first 10 entries that would be processed:');
    toProcess.slice(0, 10).forEach((entry, i) => {
      console.log(`\n${i + 1}. ${entry.fileName}`);
      console.log(`   Embedded: "${entry.title}" by "${entry.author}"`);
      console.log(`   OCR:      "${entry.ocr_title}" by "${entry.ocr_author}"`);
    });
    process.exit(0);
  }

  let processed = 0;
  let errors = 0;
  const choices = { embedded: 0, ocr: 0, neither: 0 };

  // Process a single entry
  async function processEntry(entry, entryNum) {
    const idx = indexData.indexOf(entry);
    const prefix = `[${entryNum}/${toProcess.length}]`;

    console.log(`${prefix} ${entry.fileName}`);

    try {
      const result = await chooseMetadata(
        { title: entry.title, author: entry.author },
        { title: entry.ocr_title, author: entry.ocr_author }
      );

      console.log(`${prefix} → ${result.choice}: ${result.reason}`);

      // Update entry
      indexData[idx].preferred_source = result.choice;

      if (result.choice === 'embedded') {
        indexData[idx].preferred_title = entry.title;
        indexData[idx].preferred_author = entry.author;
        choices.embedded++;
      } else if (result.choice === 'ocr') {
        indexData[idx].preferred_title = entry.ocr_title;
        indexData[idx].preferred_author = entry.ocr_author;
        choices.ocr++;
      } else {
        indexData[idx].preferred_title = null;
        indexData[idx].preferred_author = null;
        choices.neither++;
      }

      processed++;
      return true;

    } catch (error) {
      console.error(`${prefix} Error: ${error.message}`);
      errors++;
      return false;
    }
  }

  // Process in batches for parallel execution
  for (let i = 0; i < toProcess.length; i += parallel) {
    const batch = toProcess.slice(i, i + parallel);
    const batchPromises = batch.map((entry, j) => processEntry(entry, i + j + 1));

    const results = await Promise.all(batchPromises);

    // Save after each batch if any updates were made
    if (results.some(r => r)) {
      await writeFile(indexPath, JSON.stringify(indexData, null, 2));
      console.log(`  [Saved progress: ${processed} processed so far]`);
    }
  }

  // Summary
  console.log('='.repeat(60));
  console.log('Summary:');
  console.log(`  Processed: ${processed}`);
  console.log(`  Errors: ${errors}`);
  console.log(`  Chose embedded: ${choices.embedded}`);
  console.log(`  Chose OCR: ${choices.ocr}`);
  console.log(`  Neither good: ${choices.neither}`);
  console.log('='.repeat(60));
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
