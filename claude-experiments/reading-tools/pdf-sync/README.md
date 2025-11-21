# PDF Sync to S3

A Node.js CLI tool that syncs PDFs to Amazon S3 with hash-based naming and folder structure preservation.

## Features

- **Hash-based naming**: Files are renamed using SHA-256 hashes to ensure uniqueness
- **Folder structure preservation**: Maintains the original directory hierarchy in S3
- **Per-folder duplicate detection**: Skips duplicate files within the same folder (same hash can exist in different folders)
- **Smart sync**: Tracks uploaded files to avoid re-uploading
- **Dry-run mode**: Preview what would be synced without actually uploading
- **Public access**: All uploaded PDFs are publicly accessible
- **Configurable**: Customize S3 bucket and prefix

## Prerequisites

- Node.js 18+ (uses ES modules)
- AWS credentials configured (via `~/.aws/credentials`, environment variables, or IAM role)
- S3 bucket with appropriate permissions

## Installation

```bash
cd pdf-sync
npm install
```

## Usage

### Sync PDFs to S3

```bash
# Basic usage (uses default bucket and prefix)
node cli.js sync /path/to/pdfs

# With custom bucket and prefix
node cli.js sync /path/to/pdfs --bucket my-bucket --prefix documents/

# Dry run to preview what would be synced
node cli.js sync /path/to/pdfs --dry-run

# With custom state file location
node cli.js sync /path/to/pdfs --state-file /path/to/.state.json

# Exclude specific folders
node cli.js sync /path/to/pdfs --exclude "temp,drafts,archive"
```

### Options

- `-b, --bucket <bucket>` - S3 bucket name (default: `jimmyhmiller-bucket`)
- `-p, --prefix <prefix>` - S3 key prefix/folder (default: `pdfs/`)
- `-e, --exclude <folders>` - Comma-separated list of folder names to exclude
- `--dry-run` - Preview without uploading
- `-s, --state-file <file>` - Path to state file (default: `.pdf-sync-state.json`)

### Reset Sync State

If you need to clear the tracking of uploaded files:

```bash
node cli.js reset
```

## How It Works

1. **Scans** the source directory recursively for PDF files in nested folders (root-level PDFs are skipped)
2. **Calculates** SHA-256 hash for each PDF
3. **Preserves** folder structure by mapping relative paths to S3 keys
4. **Renames** files using their hash (e.g., `abc123...xyz.pdf`)
5. **Checks** if file already exists (via state file or S3 HEAD request)
6. **Uploads** new files with public-read ACL
7. **Tracks** uploaded files in a state file to avoid re-checking S3

**Important**: Only PDFs in subdirectories are synced. PDFs in the root folder are ignored.

### Example

Given this directory structure:
```
/source/
  ├── root-file.pdf         (SKIPPED - in root)
  ├── category1/
  │   ├── document1.pdf     (SYNCED)
  │   └── document2.pdf     (SYNCED)
  └── category2/
      └── document3.pdf     (SYNCED)
```

Files will be uploaded to S3 as:
```
s3://bucket/pdfs/category1/abc123...def.pdf
s3://bucket/pdfs/category1/456789...ghi.pdf
s3://bucket/pdfs/category2/xyz987...uvw.pdf
```

Note: `root-file.pdf` is not synced because it's in the root directory.

### Excluding Folders

Use the `--exclude` option to skip specific folders:

```bash
node cli.js sync /path/to/pdfs --exclude "drafts,temp,archive"
```

Given this structure with exclusions:
```
/source/
  ├── published/
  │   └── paper.pdf         (SYNCED)
  ├── drafts/
  │   └── draft.pdf         (SKIPPED - excluded)
  └── archive/
      └── old.pdf           (SKIPPED - excluded)
```

Only `published/paper.pdf` will be synced. The `drafts` and `archive` folders are completely skipped.

### Duplicate Detection (Per-Folder)

Duplicate detection works **per folder**, not globally. This means the same file (same hash) can exist in multiple folders:

```
/source/
  ├── category1/
  │   ├── report.pdf        (hash: abc123) → Uploaded
  │   └── report-copy.pdf   (hash: abc123) → SKIPPED (duplicate in same folder)
  └── category2/
      └── report.pdf        (hash: abc123) → Uploaded (different folder, allowed)
```

Result in S3:
```
s3://bucket/pdfs/category1/abc123.pdf  ✓
s3://bucket/pdfs/category2/abc123.pdf  ✓ (same hash, different folder)
```

This allows you to have the same PDF in different categories/folders while still preventing duplicates within each folder.

## State File

The tool maintains a `.pdf-sync-state.json` file that tracks:
- S3 keys of uploaded files
- Original file paths
- File hashes
- Upload timestamps

This prevents unnecessary S3 API calls on subsequent runs.

## AWS Credentials

Ensure you have AWS credentials configured. The tool uses the AWS SDK default credential provider chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. Shared credentials file (`~/.aws/credentials`)
3. IAM role (if running on EC2, ECS, Lambda, etc.)

## Required S3 Permissions

Your AWS credentials need the following S3 permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:HeadObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket/*"
    }
  ]
}
```

## Output Example

```
📁 Scanning directory: /Users/jimmy/Documents/pdfs
Found 15 PDF files

✓ Uploaded: category1/report.pdf -> s3://my-bucket/pdfs/category1/a3f5...b2c1.pdf
→ Already synced: category1/summary.pdf (d4e6...c7f2.pdf)
⊗ Duplicate (same hash): category2/report-copy.pdf
...

==================================================
Sync Summary:
==================================================
Total files found:     15
Uploaded:              8
Skipped (already synced): 5
Duplicates:            2
Errors:                0
==================================================
```

## License

MIT
