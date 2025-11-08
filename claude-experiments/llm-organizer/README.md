# LLM Organizer

An intelligent file organization system powered by Large Language Models (LLMs) that automatically analyzes, categorizes, and presents your files through a virtual FUSE filesystem.

## Features

- **Automatic File Analysis**: Watches directories and automatically extracts metadata from documents using LLM
- **Virtual Filesystem**: Mount a FUSE filesystem that presents different "views" of your files based on semantic queries
- **Smart Organization**: LLM-powered categorization, tagging, and entity extraction
- **Flexible Views**: Create custom views using natural language queries (e.g., "PDFs from 2024", "Work documents")
- **Document Support**: Built-in extractors for PDF, DOCX, TXT, JSON, and many text-based formats
- **Dynamic Analyzers**: LLM can generate custom analyzer scripts for unknown file types
- **Configurable LLM**: Works with any LLM endpoint (local models via Ollama, OpenAI, Anthropic, etc.)
- **Efficient Caching**: Response caching and SQLite-backed metadata storage

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  FUSE Virtual Filesystem                     │
│  /mount/views/work/    ← Semantic organization              │
│  /mount/views/2024/    ← Dynamic views from queries         │
│  /mount/all/           ← All tracked files                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   File Watcher (notify)                      │
│  Monitors directories → Triggers analysis on changes         │
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────▼──────────┐   ┌──────────────────────────────┐
│  Document Analyzer     │   │  LLM Integration             │
│  - PDF extraction      │──▶│  - Summarization             │
│  - DOCX parsing        │   │  - Tag extraction            │
│  - Text extraction     │   │  - Category assignment       │
│  - Type detection      │   │  - Entity recognition        │
└────────────────────────┘   └────────────┬─────────────────┘
                                          │
┌─────────────────────────────────────────▼─────────────────┐
│                SQLite Metadata Store                       │
│  - File records (hash, path, content)                     │
│  - LLM-generated metadata (tags, categories, entities)    │
│  - View definitions and file mappings                     │
│  - Dynamic analyzer registry                              │
└───────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Rust 1.70 or later
- FUSE support:
  - **macOS**: Install [macFUSE](https://osxfuse.github.io/)
  - **Linux**: Install `fuse3` package
- An LLM endpoint (local or remote)

### Building from Source

```bash
git clone https://github.com/yourusername/llm-organizer
cd llm-organizer
cargo build --release
```

The binary will be at `target/release/llm-organizer`.

## Configuration

On first run, a default configuration file will be created at `~/.config/llm-organizer/config.toml`.

### Example Configuration

```toml
[llm]
endpoint = "http://localhost:8080"  # Your LLM endpoint
model = "mistral"                    # Optional model name
max_tokens = 2048
temperature = 0.7
timeout_secs = 30

[filesystem]
watch_dirs = ["/Users/you/Documents"]  # Directories to watch
mount_point = "/Users/you/.local/share/llm-organizer/mount"
ignore_extensions = [".tmp", ".swp", ".DS_Store"]

[database]
path = "/Users/you/.local/share/llm-organizer/metadata.db"
wal_mode = true

[organization]
prompt = """
Organize files by topic, project, and date.
Extract key information like authors, dates, and subjects.
Tag documents with relevant keywords.
Focus on professional and academic content.
"""
auto_analyze = true
cache_ttl_secs = 3600  # 1 hour
```

## Usage

### Quick Start

1. **Configure your LLM endpoint** in `~/.config/llm-organizer/config.toml`

2. **Run the organizer** (watcher + FUSE mount):
   ```bash
   llm-organizer run
   ```

3. **Create a view**:
   ```bash
   llm-organizer create-view "work-docs" "Work-related documents from 2024"
   ```

4. **Access your organized files**:
   ```bash
   cd ~/.local/share/llm-organizer/mount/views/work-docs
   ls -la
   ```

### Commands

#### Start File Watcher

Watches configured directories and automatically analyzes new/modified files:

```bash
llm-organizer watch
```

#### Mount Filesystem

Mount the virtual filesystem at the configured mount point:

```bash
llm-organizer mount
```

Or specify a custom mount point:

```bash
llm-organizer mount -m /tmp/my-organized-files
```

#### Run Both (Recommended)

Run watcher and mount filesystem simultaneously:

```bash
llm-organizer run
```

#### Create a View

Create a new view from a natural language query:

```bash
llm-organizer create-view <name> <query>

# Examples:
llm-organizer create-view "pdfs-2024" "PDF documents from 2024"
llm-organizer create-view "work" "Work-related documents and emails"
llm-organizer create-view "recipes" "Cooking recipes and food-related content"
```

#### List Views

Show all available views:

```bash
llm-organizer list-views
```

#### Analyze a Single File

Manually analyze a specific file:

```bash
llm-organizer analyze /path/to/document.pdf
```

This will show:
- File type and size
- LLM-generated summary
- Extracted tags
- Assigned categories
- Recognized entities (people, dates, locations, etc.)

### Filesystem Structure

When mounted, the filesystem has this structure:

```
mount/
├── views/               # Virtual directories (views)
│   ├── work-docs/      # Files matching "work documents" query
│   ├── pdfs-2024/      # Files matching "PDFs from 2024"
│   └── recipes/        # Files matching "recipes" query
└── all/                # All tracked files
```

All files in the virtual filesystem are read-only and point to the original files on disk.

## How It Works

### 1. File Watching

The watcher monitors configured directories for file changes. When a file is created or modified:

1. File type is detected using magic bytes and extensions
2. Content is extracted using format-specific parsers
3. File information is stored in SQLite

### 2. LLM Analysis

For each file, the LLM performs:

- **Summarization**: 2-3 sentence summary of content
- **Tag Extraction**: 3-7 relevant keywords
- **Category Assignment**: 1-3 categories based on organization prompt
- **Entity Recognition**: People, organizations, dates, locations

Results are cached to minimize LLM API calls.

### 3. View Creation

When you create a view with a query:

1. Query is sent to the LLM
2. LLM generates a SQL WHERE clause or filtering logic
3. Matching files are identified from metadata
4. View is populated and cached

### 4. FUSE Filesystem

The FUSE layer presents views as virtual directories:

- Inode management for files and directories
- Pass-through reads to original files
- Dynamic directory listings based on view definitions
- Read-only operations (no write support in v1)

## LLM Endpoint Setup

### Using Ollama (Local)

1. Install [Ollama](https://ollama.ai/)
2. Pull a model:
   ```bash
   ollama pull mistral
   ```
3. Run Ollama server:
   ```bash
   ollama serve
   ```
4. Configure endpoint in config.toml:
   ```toml
   [llm]
   endpoint = "http://localhost:11434/api/generate"
   ```

### Using OpenAI

```toml
[llm]
endpoint = "https://api.openai.com/v1/completions"
# Add API key to request headers (requires code modification)
```

### Using Local llama.cpp Server

```bash
# Start llama.cpp server
./server -m models/mistral-7b.gguf --port 8080

# Configure
[llm]
endpoint = "http://localhost:8080/completion"
```

## Advanced Features

### Dynamic Analyzer Generation

For unknown file types, the LLM can generate custom analyzer scripts:

```rust
// Automatically happens when encountering unknown formats
// Scripts are saved to ~/.config/llm-organizer/analyzers/
```

The system will:
1. Send sample bytes to the LLM
2. LLM generates a Rust/Python/Shell script
3. Script is compiled/saved
4. Future files of that type use the generated analyzer

### Custom Organization Prompts

Tailor the organization strategy to your needs by editing the `organization.prompt` in config.toml:

```toml
[organization]
prompt = """
Organize files into these categories:
- Research: Academic papers and studies
- Personal: Letters, journals, personal documents
- Finance: Invoices, receipts, tax documents
- Projects: Work projects organized by client name

Extract dates in YYYY-MM-DD format.
Identify company names and people.
Tag with technology keywords if relevant.
"""
```

### View Queries

Examples of natural language queries for views:

- "All PDF files"
- "Documents from 2024"
- "Files containing the word 'budget'"
- "Work emails and documents"
- "Research papers about machine learning"
- "Personal photos and memories"
- "Tax documents and receipts"

## Troubleshooting

### FUSE Mount Fails

**macOS**:
```bash
# Install macFUSE
brew install --cask macfuse
# Reboot may be required
```

**Linux**:
```bash
sudo apt-get install fuse3
# Add user to fuse group
sudo usermod -a -G fuse $USER
```

### LLM Connection Errors

Check your LLM endpoint:
```bash
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'
```

### Database Issues

Reset the database:
```bash
rm ~/.local/share/llm-organizer/metadata.db
llm-organizer run  # Will recreate schema
```

### File Not Appearing in Views

1. Check if file was analyzed:
   ```bash
   llm-organizer analyze /path/to/file
   ```

2. Verify view query matches:
   ```bash
   llm-organizer list-views
   ```

3. Check logs:
   ```bash
   llm-organizer run -v  # Verbose mode
   ```

## Performance

- **Caching**: LLM responses cached for 1 hour (configurable)
- **Incremental Analysis**: Only new/modified files are analyzed
- **SQLite WAL Mode**: Concurrent reads during writes
- **Debouncing**: Rapid file changes are debounced (2 sec default)

## Limitations

- **Read-Only**: v1 does not support writing through FUSE
- **Text-Based**: Best for documents; limited support for binary formats
- **LLM Dependent**: Quality depends on your LLM endpoint
- **No Encryption**: Metadata stored in plaintext SQLite
- **Single Machine**: No distributed/network filesystem support

## Roadmap

- [ ] Write support for file organization
- [ ] Multi-modal LLM support (images, videos)
- [ ] Web UI for view management
- [ ] Vector embeddings for semantic search
- [ ] Conflict resolution for file moves
- [ ] Plugin system for custom analyzers
- [ ] Cloud storage integration (S3, GCS)
- [ ] Real-time view updates via inotify

## Contributing

Contributions welcome! Please open an issue or PR.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [fuser](https://github.com/cberner/fuser) for FUSE support
- Uses [notify](https://github.com/notify-rs/notify) for file watching
- Document parsing via [pdf-extract](https://github.com/jrmuizel/pdf-extract) and [docx-rs](https://github.com/bokuweb/docx-rs)
