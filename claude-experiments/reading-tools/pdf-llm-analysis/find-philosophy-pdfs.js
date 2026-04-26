#!/usr/bin/env node

import { readFile, unlink, copyFile, mkdir, stat } from 'fs/promises';
import { dirname, join, basename, resolve } from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import { homedir } from 'os';
import OpenAI from 'openai';
import { pdfToImages, imageToBase64, retryWithBackoff } from './extract-metadata.js';

const execPromise = promisify(exec);

const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed',
});

const MODEL = process.env.LLM_MODEL || 'qwen3';

const PHILOSOPHY_PROMPT = `You are classifying scanned pages from a document. Decide if it is a work of philosophy that belongs in a personal philosophy library.

INCLUDE:
- Journal articles in any subfield: metaphysics, epistemology, ethics, philosophy of mind, philosophy of language, philosophy of science, logic, political philosophy, aesthetics, phenomenology, philosophy of religion, philosophy of mathematics
- Philosophy books and monographs (analytic, continental, history of philosophy, ancient/medieval/modern/contemporary)
- Primary sources by philosophers (Plato, Aristotle, Kant, Hegel, Heidegger, Wittgenstein, Quine, Lewis, Anscombe, Foucault, Deleuze, etc.)
- Essays, lectures, and dissertations on philosophical topics
- Philosophical theology / philosophy of religion that engages substantively with philosophical argumentation
- History of philosophy and commentaries on philosophical works

EXCLUDE:
- Computer science / programming research papers, technical books, software documentation
- Pure math, physics, or empirical-science papers without philosophical content
- Resumes, CVs, invoices, receipts, tax forms, financial documents
- Flight manuals, aviation documents, product manuals, user guides
- Flyers, advertisements, promotional materials, order forms
- Personal correspondence, contracts, legal documents
- Devotional/sermon material or popular religious writing without philosophical argumentation
- Self-help / pop-psychology that merely references philosophers in passing

Return ONLY a JSON object in this exact format:
{
  "is_philosophy": true|false,
  "confidence": "high|medium|low",
  "reason": "<one short sentence>"
}

Do not include any other text.`;

async function classifyPhilosophy(pdfPath, numPages = 3, dpi = 100, quiet = true) {
  const log = quiet ? () => {} : console.log;
  const imagePaths = await pdfToImages(pdfPath, numPages, dpi);

  try {
    const imageContents = await Promise.all(
      imagePaths.map(async (p) => ({
        type: 'image_url',
        image_url: { url: `data:image/png;base64,${await imageToBase64(p)}` },
      }))
    );

    const messages = [
      {
        role: 'user',
        content: [{ type: 'text', text: PHILOSOPHY_PROMPT }, ...imageContents],
      },
    ];

    log(`Sending ${imagePaths.length} images to ${MODEL}...`);
    const response = await retryWithBackoff(
      () =>
        client.chat.completions.create({
          model: MODEL,
          messages,
          max_tokens: 4096,
          temperature: 0.1,
        }),
      3,
      1000
    );

    const responseText = (response.choices[0].message.content || '').trim();
    log('Raw response:', responseText);

    const jsonMatch = responseText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error(`No JSON in model output. Got: ${responseText.slice(0, 500)}`);
    }
    const parsed = JSON.parse(jsonMatch[0]);
    return {
      is_philosophy: !!parsed.is_philosophy,
      confidence: parsed.confidence || 'low',
      reason: parsed.reason || '',
    };
  } finally {
    for (const p of imagePaths) {
      try { await unlink(p); } catch {}
    }
  }
}

async function pathExists(p) {
  try { await stat(p); return true; } catch { return false; }
}

async function findPdfs(searchDirs) {
  const args = ['--type', 'f', '--extension', 'pdf', '.', ...searchDirs];
  const cmd = `fd ${args.map((a) => `"${a.replace(/"/g, '\\"')}"`).join(' ')}`;
  const { stdout } = await execPromise(cmd, { maxBuffer: 200 * 1024 * 1024 });
  return stdout.split('\n').map((s) => s.trim()).filter(Boolean);
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Vision-based philosophy PDF finder.

Usage:
  node find-philosophy-pdfs.js [options] [search-dir...]

Options:
  --output DIR     Destination directory (default: ~/Documents/Code/readings/philosophy_classified)
  --pages N        Pages to analyze per PDF (default: 3)
  --dpi N          Render DPI (default: 100)
  --single PATH    Classify one PDF and print the result, do not copy
  --no-copy        Classify but do not copy files
  --help, -h

Env:
  LLM_MODEL        Model name to send (default: qwen3)
`);
    process.exit(0);
  }

  const getOpt = (name) => {
    const i = args.indexOf(name);
    return i !== -1 ? args[i + 1] : undefined;
  };

  const outputDir = resolve(
    getOpt('--output') ||
      join(homedir(), 'Documents/Code/readings/philosophy_classified')
  );
  const numPages = parseInt(getOpt('--pages') || '3', 10);
  const dpi = parseInt(getOpt('--dpi') || '100', 10);
  const single = getOpt('--single');
  const noCopy = args.includes('--no-copy');

  if (single) {
    console.log(`Classifying single PDF: ${single}`);
    const r = await classifyPhilosophy(single, numPages, dpi, false);
    console.log('\nResult:', JSON.stringify(r, null, 2));
    return;
  }

  const positional = args.filter((a, i) => {
    if (a.startsWith('--')) return false;
    const prev = args[i - 1];
    if (['--output', '--pages', '--dpi', '--single'].includes(prev)) return false;
    return true;
  });

  const searchDirs = positional.length
    ? positional.map((p) => resolve(p))
    : [
        join(homedir(), 'Documents'),
        join(homedir(), 'Downloads'),
        join(homedir(), 'Desktop'),
      ];

  await mkdir(outputDir, { recursive: true });

  console.log('='.repeat(60));
  console.log('Philosophy PDF finder (vision)');
  console.log(`  Model:       ${MODEL}`);
  console.log(`  Search dirs: ${searchDirs.join(', ')}`);
  console.log(`  Output dir:  ${outputDir}`);
  console.log(`  Pages/PDF:   ${numPages}`);
  console.log(`  DPI:         ${dpi}`);
  console.log('='.repeat(60));

  const pdfs = await findPdfs(searchDirs);
  console.log(`Found ${pdfs.length} PDFs to consider\n`);

  let scanned = 0, copied = 0, skipped = 0, errors = 0, alreadyExists = 0;

  for (const pdf of pdfs) {
    scanned++;
    const fname = basename(pdf);
    const dest = join(outputDir, fname);

    if (await pathExists(dest)) {
      alreadyExists++;
      if (scanned % 25 === 0) {
        console.log(`... ${scanned}/${pdfs.length} (copied ${copied}, skipped ${skipped}, exists ${alreadyExists}, errors ${errors})`);
      }
      continue;
    }

    try {
      const r = await classifyPhilosophy(pdf, numPages, dpi, true);
      if (r.is_philosophy) {
        copied++;
        console.log(`✓ [${scanned}/${pdfs.length}] ${fname}`);
        console.log(`    confidence=${r.confidence} reason=${r.reason}`);
        if (!noCopy) await copyFile(pdf, dest);
      } else {
        skipped++;
      }
    } catch (e) {
      errors++;
      console.error(`! [${scanned}/${pdfs.length}] ${fname}: ${e.message}`);
    }

    if (scanned % 10 === 0) {
      console.log(`... ${scanned}/${pdfs.length} (copied ${copied}, skipped ${skipped}, exists ${alreadyExists}, errors ${errors})`);
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log(`Scanned:  ${scanned}`);
  console.log(`Copied:   ${copied}`);
  console.log(`Skipped:  ${skipped}`);
  console.log(`Existing: ${alreadyExists}`);
  console.log(`Errors:   ${errors}`);
  console.log(`Output:   ${outputDir}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => { console.error('Fatal:', e); process.exit(1); });
}

export { classifyPhilosophy };
