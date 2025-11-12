# PDF Classification Scripts

Scripts for finding and organizing programming-related PDFs using LLM classification.

## Main Scripts (Use These)

### `find_programming_pdfs_final.sh`
The main script for finding and copying programming-related PDFs.

**Features:**
- Uses local GPT-120B via llama.cpp server at localhost:8080
- Respects .gitignore files (uses `fd` instead of `find`)
- Outputs to separate directory: `~/Documents/Code/readings/llm_classified/`
- Skips already-classified PDFs

**Usage:**
```bash
./find_programming_pdfs_final.sh

# Or specify custom output directory:
./find_programming_pdfs_final.sh /path/to/output
```

### `test_single_pdf_final.sh`
Test classification on a single PDF to verify behavior.

**Usage:**
```bash
./test_single_pdf_final.sh "path/to/file.pdf"
```

### `prompt.txt`
The classification prompt used by both scripts above. **Edit this file** to change classification criteria.

### `rename_pdfs.sh`
Renames PDFs to clean titles based on metadata (Title_by_Author.pdf format).

**Usage:**
```bash
./rename_pdfs.sh
```

## Legacy Scripts (Old Versions)

These were intermediate versions during development:

- `find_programming_pdfs.sh` - Keyword-based (fast but inaccurate)
- `find_programming_pdfs_content.sh` - Keyword search with content extraction
- `find_programming_pdfs_llm.sh` - Ollama version (obsolete)
- `find_programming_pdfs_llm_v2.sh` - JSON grammar version
- `find_programming_pdfs_local_llm.sh` - Early local LLM version
- `test_*.sh` - Various test scripts for different versions

## Requirements

- `pdftotext` (from poppler-utils): `brew install poppler`
- `fd` (respects .gitignore): `brew install fd`
- `jq` (JSON processing): `brew install jq`
- Local LLM server running at localhost:8080 (llama.cpp server)

## How It Works

1. Uses `fd` to find all PDFs (respecting .gitignore)
2. Extracts first 3 pages of text from each PDF
3. Sends text + prompt to local LLM with tool calling
4. LLM calls `classify_pdf` function with `is_programming: true/false`
5. Copies programming-related PDFs to output directory
