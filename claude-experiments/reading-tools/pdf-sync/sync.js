import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { S3Client, PutObjectCommand, HeadObjectCommand } from '@aws-sdk/client-s3';

export class PDFSync {
  constructor(options = {}) {
    this.bucket = options.bucket || 'jimmyhmiller-bucket';
    this.prefix = options.prefix || 'pdfs/';
    this.dryRun = options.dryRun || false;
    this.stateFile = options.stateFile || '.pdf-sync-state.json';
    this.excludeFolders = options.excludeFolders || [];
    this.s3Client = new S3Client({});
    this.state = this.loadState();
  }

  loadState() {
    try {
      if (fs.existsSync(this.stateFile)) {
        const data = fs.readFileSync(this.stateFile, 'utf8');
        return JSON.parse(data);
      }
    } catch (error) {
      console.warn(`Warning: Could not load state file: ${error.message}`);
    }
    return { uploaded: {} };
  }

  saveState() {
    try {
      fs.writeFileSync(this.stateFile, JSON.stringify(this.state, null, 2));
    } catch (error) {
      console.error(`Error saving state file: ${error.message}`);
    }
  }

  calculateHash(filePath) {
    const fileBuffer = fs.readFileSync(filePath);
    const hashSum = crypto.createHash('sha256');
    hashSum.update(fileBuffer);
    return hashSum.digest('hex');
  }

  async findPDFs(dir, baseDir = dir) {
    const results = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        // Skip excluded folders
        if (this.excludeFolders.includes(entry.name)) {
          console.log(`⊘ Skipping excluded folder: ${path.relative(baseDir, fullPath)}`);
          continue;
        }
        results.push(...await this.findPDFs(fullPath, baseDir));
      } else if (entry.isFile() && path.extname(entry.name).toLowerCase() === '.pdf') {
        const relativePath = path.relative(baseDir, fullPath);
        const relativeDir = path.dirname(relativePath);

        // Only include files in nested folders, not root
        if (relativeDir !== '.') {
          results.push({
            fullPath,
            relativePath,
            relativeDir,
            fileName: entry.name
          });
        }
      }
    }

    return results;
  }

  async fileExistsInS3(key) {
    try {
      await this.s3Client.send(new HeadObjectCommand({
        Bucket: this.bucket,
        Key: key
      }));
      return true;
    } catch (error) {
      if (error.name === 'NotFound' || error.$metadata?.httpStatusCode === 404) {
        return false;
      }
      throw error;
    }
  }

  async uploadFile(localPath, s3Key) {
    if (this.dryRun) {
      console.log(`[DRY RUN] Would upload: ${localPath} -> s3://${this.bucket}/${s3Key}`);
      return true;
    }

    try {
      const fileContent = fs.readFileSync(localPath);

      await this.s3Client.send(new PutObjectCommand({
        Bucket: this.bucket,
        Key: s3Key,
        Body: fileContent,
        ContentType: 'application/pdf',
        ACL: 'public-read'
      }));

      console.log(`✓ Uploaded: ${localPath} -> s3://${this.bucket}/${s3Key}`);
      return true;
    } catch (error) {
      console.error(`✗ Error uploading ${localPath}: ${error.message}`);
      return false;
    }
  }

  async sync(sourceDir) {
    if (!fs.existsSync(sourceDir)) {
      throw new Error(`Source directory does not exist: ${sourceDir}`);
    }

    console.log(`\n📁 Scanning directory: ${sourceDir}`);
    const pdfs = await this.findPDFs(sourceDir);
    console.log(`Found ${pdfs.length} PDF files\n`);

    const stats = {
      total: pdfs.length,
      uploaded: 0,
      skipped: 0,
      errors: 0,
      duplicates: 0
    };

    // Track seen hashes per folder (not globally)
    const seenHashesByFolder = new Map();

    for (const pdf of pdfs) {
      const hash = this.calculateHash(pdf.fullPath);
      // relativeDir always exists since we only include files in nested folders
      const s3Key = `${this.prefix}${pdf.relativeDir}/${hash}.pdf`;

      // Check for duplicates within the same folder only
      if (!seenHashesByFolder.has(pdf.relativeDir)) {
        seenHashesByFolder.set(pdf.relativeDir, new Set());
      }
      const folderHashes = seenHashesByFolder.get(pdf.relativeDir);

      if (folderHashes.has(hash)) {
        console.log(`⊗ Duplicate in folder (same hash): ${pdf.relativePath}`);
        stats.duplicates++;
        stats.skipped++;
        continue;
      }
      folderHashes.add(hash);

      // Check if already uploaded in previous runs
      if (this.state.uploaded[s3Key]) {
        console.log(`→ Already synced: ${pdf.relativePath} (${hash}.pdf)`);
        stats.skipped++;
        continue;
      }

      // Check if file exists in S3 (for cases where state file was lost)
      if (!this.dryRun && await this.fileExistsInS3(s3Key)) {
        console.log(`→ Already in S3: ${pdf.relativePath} (${hash}.pdf)`);
        this.state.uploaded[s3Key] = {
          originalPath: pdf.relativePath,
          hash,
          uploadedAt: new Date().toISOString()
        };
        stats.skipped++;
        continue;
      }

      // Upload the file
      const success = await this.uploadFile(pdf.fullPath, s3Key);
      if (success) {
        stats.uploaded++;
        if (!this.dryRun) {
          this.state.uploaded[s3Key] = {
            originalPath: pdf.relativePath,
            hash,
            uploadedAt: new Date().toISOString()
          };
        }
      } else {
        stats.errors++;
      }
    }

    // Save state after sync
    if (!this.dryRun) {
      this.saveState();
    }

    // Print summary
    console.log('\n' + '='.repeat(50));
    console.log('Sync Summary:');
    console.log('='.repeat(50));
    console.log(`Total files found:     ${stats.total}`);
    console.log(`Uploaded:              ${stats.uploaded}`);
    console.log(`Skipped (already synced): ${stats.skipped}`);
    console.log(`Duplicates:            ${stats.duplicates}`);
    console.log(`Errors:                ${stats.errors}`);
    console.log('='.repeat(50));

    if (this.dryRun) {
      console.log('\n⚠️  This was a DRY RUN - no files were actually uploaded');
    }

    return stats;
  }
}
