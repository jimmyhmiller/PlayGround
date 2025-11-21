# PDF LLM Analysis

Extract title and author metadata from PDFs using vision LLM (qwen-vl) for PDFs that don't have embedded metadata.

## Overview

This tool uses a local qwen-vl vision model (running via OpenAI-compatible API at localhost:8080) to extract title and author information from PDF files by analyzing the first few pages as images.

## Prerequisites

- Node.js (v18+)
- `pdftoppm` (part of poppler-utils, usually installed via Homebrew on macOS: `brew install poppler`)
- qwen-vl model running at `http://localhost:8080/v1` (OpenAI-compatible API)

## Installation

```bash
npm install
```

## Tools

### 1. extract-metadata.js - Single PDF Extraction

Extract title and author from a single PDF file.

**Usage:**
```bash
# Analyze first 3 pages (default)
node extract-metadata.js /path/to/paper.pdf

# Analyze first 5 pages for better accuracy
node extract-metadata.js /path/to/paper.pdf 5
```

**Output:**
```json
{
  "title": "The Declarative Imperative: Experiences and Conjectures in Distributed Logic",
  "author": "Joseph M. Hellerstein"
}
```

### 2. update-index.js - Batch Processing & Index Updates

Update a PDF index (created by pdf-indexer) with LLM-extracted metadata.

**Usage:**

```bash
# Process PDFs without metadata
node update-index.js ../pdf-indexer/pdf-index.json

# Verify ALL PDFs and add ocr_title for mismatches
node update-index.js ../pdf-indexer/pdf-index.json --verify

# Use more pages for better accuracy
node update-index.js ../pdf-indexer/pdf-index.json --pages 5

# Use stricter matching (90% similarity required)
node update-index.js ../pdf-indexer/pdf-index.json --verify --threshold 0.9
```

**Options:**
- `--verify` - Verify existing metadata and add `ocr_title`/`ocr_author` fields if titles don't match
- `--pages N` - Number of pages to analyze per PDF (default: 3)
- `--threshold T` - Similarity threshold for fuzzy matching (0-1, default: 0.8)

**Behavior:**

1. **Without `--verify` flag:**
   - Only processes PDFs where `metadataFound: false`
   - Adds `ocr_title` and `ocr_author` fields to these PDFs

2. **With `--verify` flag:**
   - Processes ALL PDFs in the index
   - Compares existing title with LLM-extracted title using fuzzy matching
   - If titles match (using prefix matching or 80%+ similarity), keeps existing metadata
   - If titles don't match, adds `ocr_title` and `ocr_author` fields

**Fuzzy Matching:**

The tool uses intelligent fuzzy matching that:
- Recognizes when one title is a prefix of another (e.g., "The Declarative Imperative" matches "The Declarative Imperative: Experiences and Conjectures in Distributed Logic")
- Uses Levenshtein distance for general similarity comparison
- Configurable threshold (default: 0.8 = 80% similarity)

**Example Output:**

```
Loading index from: pdf-index.json

Found 100 PDFs in index

45 PDFs need processing

[1/45] Processing: paper1.pdf
  Path: /path/to/paper1.pdf
  No existing metadata
  LLM title: "Deep Learning for Natural Language Processing"
  LLM author: "John Smith, Jane Doe"
  Action: Adding OCR metadata (no existing metadata)

[2/45] Processing: paper2.pdf
  Path: /path/to/paper2.pdf
  Existing title: "Neural Networks"
  Existing author: "Bob Jones"
  LLM title: "Neural Networks: A Comprehensive Introduction"
  LLM author: "Bob Jones"
  Match: One title is a prefix of the other
  Action: Titles match, keeping existing metadata

============================================================
Summary:
Total PDFs in index: 100
PDFs examined: 45
PDFs updated with OCR fields: 30
Titles matched: 10
Titles mismatched: 5
============================================================
```

**Updated Index Structure:**

PDFs without metadata get `ocr_title` and `ocr_author` fields:

```json
{
  "hash": "abc123...",
  "path": "/path/to/paper.pdf",
  "fileName": "paper.pdf",
  "title": null,
  "author": null,
  "metadataFound": false,
  "ocr_title": "Extracted Title from Vision LLM",
  "ocr_author": "Extracted Author(s)"
}
```

PDFs with mismatched metadata get additional OCR fields:

```json
{
  "hash": "def456...",
  "path": "/path/to/paper.pdf",
  "fileName": "paper.pdf",
  "title": "Wrong Title from PDF Metadata",
  "author": "Wrong Author",
  "metadataFound": true,
  "ocr_title": "Correct Title from Vision LLM",
  "ocr_author": "Correct Author(s)"
}
```

## How It Works

1. **PDF to Images**: Uses `pdftoppm` to convert the first N pages of the PDF to PNG images
   - Configurable resolution (default: 72 DPI for fast processing)
   - Lower DPI = smaller images, faster uploads, faster LLM processing
2. **Base64 Encoding**: Encodes images as base64 for transmission to the model
3. **Vision Analysis**: Sends images to qwen-vl model via OpenAI-compatible API
   - Includes automatic retry logic with exponential backoff (3 retries: 1s, 2s, 4s delays)
   - Handles connection errors gracefully (ECONNREFUSED, ETIMEDOUT, etc.)
4. **JSON Extraction**: Parses the model's response to extract title and author
5. **Index Update**: Updates the index file with new OCR fields

## Configuration

The qwen-vl API endpoint is configured in `extract-metadata.js`:

```javascript
const client = new OpenAI({
  baseURL: 'http://localhost:8080/v1',
  apiKey: 'not-needed',
});
```

Adjust the `baseURL` if your model is running on a different host/port.

## Performance Tips

- **Image Resolution (DPI)**: Lower DPI = faster processing and smaller images
  - 50-72 DPI: Very fast, good for most academic papers with clear titles (default: 72)
  - 100-150 DPI: Better quality, slower processing
  - Use `--dpi N` to adjust: `node update-index.js index.json --dpi 50`

- **Accuracy vs Speed**: More pages = better accuracy but slower processing
  - 2-3 pages: Fast, good for most academic papers
  - 5+ pages: Better for documents with complex layouts

- **Batch Processing**: The tool processes PDFs sequentially to avoid overwhelming the LLM API
  - For very large indexes, consider running on a subset first

- **Threshold Tuning**: Adjust `--threshold` based on your needs
  - 0.9 (90%): Strict matching, fewer false positives
  - 0.8 (80%): Default, balanced
  - 0.7 (70%): Loose matching, more matches but potential false positives

**Recommended settings for speed:**
```bash
# Fast processing (good for batch jobs)
node update-index.js index.json --pages 2 --dpi 50

# Balanced (default)
node update-index.js index.json --pages 3 --dpi 72

# High accuracy (slower)
node update-index.js index.json --pages 5 --dpi 150
```

## Error Handling & Resilience

The tool is designed to be resilient for long-running batch operations:

- **Automatic Retries**: Connection errors are automatically retried 3 times with exponential backoff
  - Retry delays: 1 second, 2 seconds, 4 seconds
  - Handles: ECONNREFUSED, ECONNRESET, ETIMEDOUT, and other network errors

- **Graceful Degradation**: If a PDF fails after all retries:
  - Error is logged to console only (not saved to index)
  - Processing continues with the next PDF
  - Failed PDFs will be retried on the next run

- **Incremental Saves**: The index file is saved after **each PDF** is processed
  - No progress is lost if the process is interrupted (Ctrl+C, crash, etc.)
  - You can safely stop and resume at any time

- **Resume Capability**: If processing is interrupted:
  - Successfully processed PDFs have `ocr_title`/`ocr_author` fields
  - Re-running without `--verify` will skip successfully processed PDFs
  - Failed PDFs will be retried (they are not marked in the index)
  - Re-running with `--verify` will re-verify all PDFs

## Troubleshooting

**Error: `pdftoppm` not found**
```bash
brew install poppler
```

**Error: Connection refused to localhost:8080**
- Make sure qwen-vl model server is running
- Check the port number in `extract-metadata.js`
- The tool will automatically retry 3 times with exponential backoff (1s, 2s, 4s)
- If retries fail, the error is logged to console and that PDF will be retried on the next run

**Poor extraction quality**
- Increase the number of pages analyzed: `--pages 5`
- Check if the PDF has unusual formatting or is scanned (might need OCR preprocessing)

## Integration with pdf-indexer

This tool is designed to work seamlessly with the pdf-indexer tool:

1. First, run pdf-indexer to create an initial index:
   ```bash
   cd ../pdf-indexer
   node index.js /path/to/pdfs
   ```

2. Then, enhance the index with LLM-extracted metadata:
   ```bash
   cd ../pdf-llm-analysis
   node update-index.js ../pdf-indexer/pdf-index.json
   ```

3. Optionally verify existing metadata:
   ```bash
   node update-index.js ../pdf-indexer/pdf-index.json --verify
   ```

## License

ISC
