# PDF Renamer

Rename PDF files based on extracted metadata (title and author) from the PDF indexer.

## Features

- Renames PDFs using format: `lastname-firstname--title-words.pdf`
- Hash-based file lookup (works even if files have been moved/renamed)
- Dry-run mode to preview changes
- Safe: won't overwrite existing files
- Handles multiple authors, special characters, edge cases

## Usage

```bash
# Preview changes (recommended first!)
node rename-pdfs.js --dry-run

# Actually rename the files
node rename-pdfs.js

# Or use npm scripts
npm run dry-run
npm run rename
```

## Options

```
--index <path>      Path to pdf-index.json (default: ../pdf-indexer/pdf-index.json)
--dir <path>        Directory containing PDFs to rename (default: from index)
--dry-run           Preview changes without actually renaming files
-h, --help          Show help message
```

## Examples

### Custom index and directory
```bash
node rename-pdfs.js --index ./my-index.json --dir /path/to/pdfs
```

### Just preview
```bash
node rename-pdfs.js --dry-run
```

## Naming Convention

- Format: `lastname-firstname--title-words.pdf`
- All lowercase with hyphens (kebab-case)
- Removes leading articles (the, a, an)
- Multiple authors: shows first 2 + "et-al" if more
- Max filename length: 120 characters

### Examples

- `negroponte-nicholas--idiosyncratic-systems-approach-to-interactive-graphics.pdf`
- `kay-alan-goldberg-adele--personal-dynamic-media.pdf`
- `dijkstra-edsger-w--humble-programmer.pdf`
- `barton-r-s--design-of-programming-language-processors-ii.pdf`

## Requirements

- Node.js
- An index file from pdf-indexer with extracted metadata
