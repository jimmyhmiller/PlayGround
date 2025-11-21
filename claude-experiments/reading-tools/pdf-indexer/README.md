# PDF Indexer

A Node.js tool that recursively scans directories for PDF files and builds a comprehensive JSON index with metadata including titles, authors, file hashes, and paths.

## Features

- **Recursive Directory Scanning**: Automatically finds all PDF files in a directory and its subdirectories
- **File Hashing**: Calculates SHA256 hash for each PDF to uniquely identify files
- **Metadata Extraction**: Extracts title, author, and other metadata from PDF files
- **Missing Metadata Tracking**: Flags PDFs where metadata couldn't be extracted for later processing
- **JSON Output**: Creates a structured JSON index file for easy searching and filtering

## Installation

```bash
npm install
```

## Usage

### Basic Usage

```bash
node index.js <directory> [output-file]
```

### Examples

Scan a directory and create `pdf-index.json`:
```bash
node index.js /path/to/pdfs
```

Scan a directory and specify output file:
```bash
node index.js /path/to/pdfs my-custom-index.json
```

Show help:
```bash
node index.js --help
```

### Install Globally (Optional)

To use the tool from anywhere:
```bash
npm install -g .
pdf-indexer /path/to/pdfs
```

## Output Format

The tool generates a JSON file with the following structure for each PDF:

```json
[
  {
    "hash": "a1b2c3d4e5f6...",
    "path": "/full/path/to/file.pdf",
    "fileName": "file.pdf",
    "title": "Document Title",
    "author": "Author Name",
    "metadataFound": true,
    "totalPages": 42,
    "creator": "Microsoft Word",
    "producer": "Adobe PDF Library",
    "creationDate": "D:20230101120000Z",
    "error": null,
    "processedAt": "2025-11-12T15:30:00.000Z"
  },
  {
    "hash": "f6e5d4c3b2a1...",
    "path": "/full/path/to/another.pdf",
    "fileName": "another.pdf",
    "title": null,
    "author": null,
    "metadataFound": false,
    "totalPages": 10,
    "creator": null,
    "producer": "Acrobat Distiller",
    "creationDate": null,
    "error": null,
    "processedAt": "2025-11-12T15:30:05.000Z"
  }
]
```

### Field Descriptions

- **hash**: SHA256 hash of the PDF file for deduplication
- **path**: Full path to the PDF file
- **fileName**: Name of the PDF file
- **title**: Extracted title from PDF metadata (null if not found)
- **author**: Extracted author from PDF metadata (null if not found)
- **metadataFound**: Boolean flag indicating if both title and author were found
- **totalPages**: Number of pages in the PDF
- **creator**: Application that created the original document
- **producer**: PDF producer/converter used
- **creationDate**: Creation date from PDF metadata
- **error**: Error message if processing failed (null if successful)
- **processedAt**: ISO timestamp when the PDF was processed

## Use Cases

### Find PDFs Without Metadata

```javascript
const index = require('./pdf-index.json');
const withoutMetadata = index.filter(pdf => !pdf.metadataFound);
console.log(`Found ${withoutMetadata.length} PDFs without metadata`);
```

### Find Duplicates by Hash

```javascript
const index = require('./pdf-index.json');
const hashes = {};
index.forEach(pdf => {
  if (!hashes[pdf.hash]) hashes[pdf.hash] = [];
  hashes[pdf.hash].push(pdf.path);
});

const duplicates = Object.values(hashes).filter(paths => paths.length > 1);
console.log('Duplicate PDFs:', duplicates);
```

### Search by Author

```javascript
const index = require('./pdf-index.json');
const byAuthor = index.filter(pdf =>
  pdf.author && pdf.author.toLowerCase().includes('smith')
);
```

## Error Handling

- Directories that can't be read are skipped with a warning
- PDFs that can't be processed are included in the index with `metadataFound: false` and an error message
- The tool continues processing even if individual PDFs fail

## Dependencies

- **pdf-parse**: For extracting text and metadata from PDF files
- **crypto**: Built-in Node.js module for hash calculation
- **fs/promises**: Built-in Node.js module for async file operations

## License

ISC
