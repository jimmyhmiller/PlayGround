# PDF OCR Finder

Find and copy PDFs without extractable text (OCR scans/image-only PDFs).

## Overview

This tool scans a directory of PDFs and identifies files that don't contain extractable text - typically scanned documents that haven't been OCR'd. It can optionally copy these files to a separate directory for batch OCR processing.

## Installation

```bash
npm install
```

## Usage

```bash
# Basic usage - scan directory
./index.js ~/Documents/PDFs

# Specify custom output directory
./index.js ~/Documents/PDFs --output ~/OCR-Scans

# List OCR PDFs without moving them
./index.js ~/Documents/PDFs --list-only

# Preview what would be moved (dry run)
./index.js ~/Documents/PDFs --dry-run

# Adjust text threshold (default is 100 characters)
./index.js ~/Documents/PDFs --min-text 50
```

## Options

- `--output <dir>` - Directory to copy OCR PDFs to (default: ./ocr-scans)
- `--min-text <number>` - Minimum characters to consider as having text (default: 100)
- `--dry-run` - Show what would be copied without actually copying files
- `--list-only` - Only list OCR PDFs without copying them
- `--help, -h` - Show help message

## How It Works

1. Recursively scans the input directory for PDF files
2. For each PDF, extracts text using `pdf-parse`
3. Counts non-whitespace characters in the extracted text
4. PDFs with fewer characters than the threshold are considered OCR scans
5. Optionally copies identified OCR PDFs to the output directory

## Examples

**Find all OCR PDFs in your library:**
```bash
./index.js ~/PDFs --list-only
```

**Copy OCR PDFs to a separate folder:**
```bash
./index.js ~/PDFs --output ~/PDFs-Need-OCR
```

**Preview before copying:**
```bash
./index.js ~/PDFs --output ~/PDFs-Need-OCR --dry-run
```

## Notes

- The tool uses a default threshold of 100 characters to distinguish between PDFs with and without text
- Some PDFs may contain minimal metadata text but no actual content - adjust `--min-text` as needed
- Duplicate filenames in the target directory are handled by appending a counter
- The tool processes PDFs sequentially to avoid memory issues with large collections
