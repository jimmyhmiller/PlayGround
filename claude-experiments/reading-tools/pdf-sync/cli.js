#!/usr/bin/env node

import { Command } from 'commander';
import { PDFSync } from './sync.js';
import path from 'path';
import fs from 'fs';

const program = new Command();

program
  .name('pdf-sync')
  .description('Sync PDFs to S3 with hash-based naming and folder structure preservation')
  .version('1.0.0');

program
  .command('sync')
  .description('Sync PDFs from a directory to S3')
  .argument('<directory>', 'Source directory containing PDFs')
  .option('-b, --bucket <bucket>', 'S3 bucket name', 'jimmyhmiller-bucket')
  .option('-p, --prefix <prefix>', 'S3 key prefix/folder', 'pdfs/')
  .option('--dry-run', 'Preview what would be synced without uploading', false)
  .option('-s, --state-file <file>', 'Path to state file', '.pdf-sync-state.json')
  .option('-e, --exclude <folders>', 'Comma-separated list of folder names to exclude', '')
  .action(async (directory, options) => {
    try {
      const absolutePath = path.resolve(directory);

      // Parse comma-separated exclude list
      const excludeFolders = options.exclude
        ? options.exclude.split(',').map(f => f.trim()).filter(f => f.length > 0)
        : [];

      const syncer = new PDFSync({
        bucket: options.bucket,
        prefix: options.prefix,
        dryRun: options.dryRun,
        stateFile: options.stateFile,
        excludeFolders
      });

      await syncer.sync(absolutePath);
    } catch (error) {
      console.error(`\n❌ Error: ${error.message}`);
      process.exit(1);
    }
  });

program
  .command('reset')
  .description('Reset the sync state (clear tracking of uploaded files)')
  .option('-s, --state-file <file>', 'Path to state file', '.pdf-sync-state.json')
  .action(async (options) => {
    try {
      if (fs.existsSync(options.stateFile)) {
        fs.unlinkSync(options.stateFile);
        console.log(`✓ State file deleted: ${options.stateFile}`);
      } else {
        console.log(`No state file found at: ${options.stateFile}`);
      }
    } catch (error) {
      console.error(`Error deleting state file: ${error.message}`);
      process.exit(1);
    }
  });

program.parse();
