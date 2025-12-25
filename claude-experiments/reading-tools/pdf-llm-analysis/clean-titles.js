#!/usr/bin/env node

/**
 * Clean up titles with OCR mistakes, encoding issues, and typos using local LLM.
 */

import { readFile, writeFile } from 'fs/promises';
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed'
});

async function cleanTitle(title, author) {
  if (!title) return null;

  const prompt = `Fix any OCR errors, encoding issues, or typos in this academic paper title and author. Common issues include:
- Wrong quotes or special chars (e.g., G"odel should be Gödel)
- LaTeX artifacts (e.g., \\' or \\")
- Garbled characters from bad encoding

IMPORTANT:
- Do NOT change intentional capitalization or acronyms (e.g., STEPS, LISP, SQL should stay uppercase)
- Do NOT change title case or style choices
- Only fix actual errors like encoding problems or OCR mistakes

Title: "${title}"
Author: "${author || 'Unknown'}"

Return ONLY a JSON object with the cleaned versions:
{
  "title": "cleaned title here",
  "author": "cleaned author here",
  "changed": true/false
}

If nothing needs fixing, set changed to false and return the originals.`;

  const response = await client.chat.completions.create({
    model: 'qwen-vl',
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.1,
    max_tokens: 300
  });

  const text = response.choices[0]?.message?.content || '';
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return null;

  return JSON.parse(jsonMatch[0]);
}

async function main() {
  const args = process.argv.slice(2);
  const indexPath = args[0] || '../pdf-indexer/pdf-index-fixed.json';
  const dryRun = args.includes('--dry-run');
  const parallel = parseInt(args.find(a => a.startsWith('--parallel='))?.split('=')[1] || '1');

  console.log('='.repeat(60));
  console.log('Clean Titles');
  console.log('='.repeat(60));
  console.log(`Index: ${indexPath}`);
  console.log(`Dry run: ${dryRun}`);
  console.log(`Parallel: ${parallel}`);
  console.log('');

  const indexData = JSON.parse(await readFile(indexPath, 'utf-8'));

  // Find entries with preferred_title that haven't been cleaned yet
  const toProcess = indexData.filter(e => e.preferred_title && !e.unmodified_preferred_title);
  console.log(`Entries to check: ${toProcess.length}`);

  let cleaned = 0;
  let unchanged = 0;
  let errors = 0;

  async function processEntry(entry, num) {
    const idx = indexData.indexOf(entry);
    const title = entry.preferred_title;
    const author = entry.preferred_author;

    try {
      const result = await cleanTitle(title, author);

      if (result && result.changed) {
        console.log(`[${num}] FIXED:`);
        console.log(`  Before: "${title}"`);
        console.log(`  After:  "${result.title}"`);
        if (author !== result.author) {
          console.log(`  Author: "${author}" → "${result.author}"`);
        }

        if (!dryRun) {
          // Back up originals
          indexData[idx].unmodified_preferred_title = title;
          indexData[idx].unmodified_preferred_author = author;
          // Update in place
          indexData[idx].preferred_title = result.title;
          indexData[idx].preferred_author = result.author;
        }
        cleaned++;
        return true;
      } else {
        unchanged++;
        return false;
      }
    } catch (err) {
      console.error(`[${num}] Error: ${err.message}`);
      errors++;
      return false;
    }
  }

  // Process in batches
  for (let i = 0; i < toProcess.length; i += parallel) {
    const batch = toProcess.slice(i, i + parallel);
    const promises = batch.map((e, j) => processEntry(e, i + j + 1));
    const results = await Promise.all(promises);

    if (!dryRun && results.some(r => r)) {
      await writeFile(indexPath, JSON.stringify(indexData, null, 2));
    }

    // Progress
    if ((i + parallel) % 50 === 0 || i + parallel >= toProcess.length) {
      console.log(`  Progress: ${Math.min(i + parallel, toProcess.length)}/${toProcess.length}`);
    }
  }

  console.log('');
  console.log('='.repeat(60));
  console.log(`Cleaned: ${cleaned}`);
  console.log(`Unchanged: ${unchanged}`);
  console.log(`Errors: ${errors}`);
  console.log('='.repeat(60));
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
