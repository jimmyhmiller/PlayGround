#!/usr/bin/env node

import { readFileSync, writeFileSync, readdirSync, mkdirSync } from 'fs';
import { join, basename, resolve } from 'path';
import { execSync } from 'child_process';

const RUST_PROJECT_DIR = resolve(process.cwd());
const ION_EXAMPLES_DIR = join(RUST_PROJECT_DIR, 'ion-examples');
const FIXTURES_DIR = join(RUST_PROJECT_DIR, 'tests/fixtures/ion-examples');
const TS_IONGRAPH_DIR = '/Users/jimmyhmiller/Documents/Code/open-source/iongraph2';

// Ensure fixtures directory exists
try {
  mkdirSync(FIXTURES_DIR, { recursive: true });
} catch (e) {
  // Directory might already exist
}

console.log('🔍 Scanning ion-examples directory...\n');

// Get all JSON files
const jsonFiles = readdirSync(ION_EXAMPLES_DIR)
  .filter(f => f.endsWith('.json'))
  .sort();

console.log(`Found ${jsonFiles.length} JSON files\n`);

let totalFixtures = 0;
let successCount = 0;
let failCount = 0;
const manifest = [];

for (const jsonFile of jsonFiles) {
  const jsonPath = join(ION_EXAMPLES_DIR, jsonFile);
  const baseName = basename(jsonFile, '.json');

  console.log(`\n📁 Processing ${jsonFile}...`);

  try {
    // Parse JSON to count functions and passes
    const data = JSON.parse(readFileSync(jsonPath, 'utf-8'));
    const numFunctions = data.functions.length;

    console.log(`   Functions: ${numFunctions}`);

    for (let funcIdx = 0; funcIdx < numFunctions; funcIdx++) {
      const func = data.functions[funcIdx];
      const numPasses = func.passes.length;

      console.log(`   Function ${funcIdx} "${func.name || 'unnamed'}": ${numPasses} passes`);

      for (let passIdx = 0; passIdx < numPasses; passIdx++) {
        const pass = func.passes[passIdx];
        const fixtureBaseName = `ts-${baseName}-func${funcIdx}-pass${passIdx}`;
        const fixturePath = join(FIXTURES_DIR, `${fixtureBaseName}.svg`);

        totalFixtures++;

        try {
          // Generate TypeScript SVG using the TS implementation with absolute paths
          const absoluteJsonPath = resolve(jsonPath);
          const absoluteFixturePath = resolve(fixturePath);
          const cmd = `cd ${TS_IONGRAPH_DIR} && node generate-svg-function.mjs "${absoluteJsonPath}" ${funcIdx} ${passIdx} "${absoluteFixturePath}"`;
          execSync(cmd, { stdio: 'pipe' });

          successCount++;

          // Read the generated SVG to get dimensions
          const svgContent = readFileSync(fixturePath, 'utf-8');
          const widthMatch = svgContent.match(/width="([^"]+)"/);
          const heightMatch = svgContent.match(/height="([^"]+)"/);

          manifest.push({
            file: jsonFile,
            baseName: baseName,
            funcIndex: funcIdx,
            funcName: func.name || 'unnamed',
            passIndex: passIdx,
            passName: pass.name || 'unnamed',
            blocks: pass.mir.blocks.length,
            fixtureName: `${fixtureBaseName}.svg`,
            width: widthMatch ? widthMatch[1] : 'unknown',
            height: heightMatch ? heightMatch[1] : 'unknown',
          });

          console.log(`      ✓ Pass ${passIdx} "${pass.name || 'unnamed'}" (${pass.mir.blocks.length} blocks)`);
        } catch (error) {
          failCount++;
          console.error(`      ✗ Failed to generate pass ${passIdx}:`, error.message);
        }
      }
    }
  } catch (error) {
    console.error(`   ✗ Failed to process ${jsonFile}:`, error.message);
  }
}

// Write manifest
const manifestPath = join(FIXTURES_DIR, 'manifest.json');
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

console.log('\n' + '='.repeat(60));
console.log('📊 SUMMARY');
console.log('='.repeat(60));
console.log(`Total fixtures generated: ${successCount}/${totalFixtures}`);
console.log(`Success: ${successCount}`);
console.log(`Failed: ${failCount}`);
console.log(`\nManifest written to: ${manifestPath}`);
console.log(`Fixtures directory: ${FIXTURES_DIR}`);
console.log('='.repeat(60));

if (failCount > 0) {
  process.exit(1);
}
