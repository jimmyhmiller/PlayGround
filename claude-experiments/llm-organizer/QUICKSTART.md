# Quick Start Guide

Get up and running with LLM Organizer in 5 minutes.

## Prerequisites

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Install FUSE**:

   **macOS**:
   ```bash
   brew install --cask macfuse
   # Reboot may be required after first install
   ```

   **Linux (Ubuntu/Debian)**:
   ```bash
   sudo apt-get install fuse3
   ```

3. **Set up a local LLM** (we'll use Ollama):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull a model (choose one)
   ollama pull mistral      # Recommended, 7B parameters
   ollama pull llama2       # Alternative, 7B parameters
   ollama pull phi          # Smaller, 2.7B parameters
   ```

## Installation

1. **Clone and build**:
   ```bash
   git clone https://github.com/yourusername/llm-organizer
   cd llm-organizer
   cargo build --release
   ```

2. **Install the binary** (optional):
   ```bash
   sudo cp target/release/llm-organizer /usr/local/bin/
   ```

## Configuration

1. **Run once to generate default config**:
   ```bash
   llm-organizer analyze --help  # Any command will create config
   ```

2. **Edit the configuration**:
   ```bash
   # macOS/Linux
   nano ~/.config/llm-organizer/config.toml
   ```

3. **Update the LLM endpoint** (if using Ollama):
   ```toml
   [llm]
   endpoint = "http://localhost:11434/api/generate"
   model = "mistral"  # or whichever model you pulled
   ```

4. **Set directories to watch**:
   ```toml
   [filesystem]
   watch_dirs = [
       "/Users/yourname/Documents",  # macOS
       # "/home/yourname/Documents",  # Linux
   ]
   ```

## First Run

### Test with a Single File

1. **Create a test document**:
   ```bash
   echo "This is a test document about Rust programming. It contains code examples and explanations." > /tmp/test-doc.txt
   ```

2. **Analyze it**:
   ```bash
   llm-organizer analyze /tmp/test-doc.txt
   ```

   You should see:
   ```
   Analyzing file: /tmp/test-doc.txt

   Analysis Results:
   ================
   File Type: text/plain
   Size: 95 bytes

   Summary:
   A brief document providing information and code examples related to Rust programming.

   Tags: rust, programming, code, examples
   Categories: Programming/Documentation

   Entities:
   {
     "people": [],
     "organizations": [],
     "dates": [],
     "locations": []
   }
   ```

### Start the Watcher

1. **In one terminal, start the file watcher**:
   ```bash
   llm-organizer watch -v
   ```

   You'll see:
   ```
   [INFO] Loaded configuration
   [INFO] Database initialized at /Users/you/.local/share/llm-organizer/metadata.db
   [INFO] LLM client initialized for endpoint: http://localhost:11434/api/generate
   [INFO] Starting file watcher...
   [INFO] Watching directory: /Users/you/Documents
   ```

2. **In another terminal, add a test file**:
   ```bash
   echo "Meeting notes from 2024-01-15: Discussed project timeline with Alice and Bob." > ~/Documents/meeting-notes.txt
   ```

3. **Watch the analysis happen** in the watcher terminal:
   ```
   [INFO] Processing file: /Users/you/Documents/meeting-notes.txt
   [DEBUG] Summary: Notes from a meeting on January 15, 2024 regarding project timeline.
   [DEBUG] Tags: ["meeting", "notes", "project", "timeline"]
   [DEBUG] Categories: ["Work/Meetings"]
   [INFO] Successfully analyzed and stored metadata for /Users/you/Documents/meeting-notes.txt
   ```

### Create and Use Views

1. **Create a view for work documents**:
   ```bash
   llm-organizer create-view "work" "Work-related documents and meetings"
   ```

2. **Create a view for 2024 files**:
   ```bash
   llm-organizer create-view "2024" "Documents from 2024"
   ```

3. **List your views**:
   ```bash
   llm-organizer list-views
   ```

   Output:
   ```
   Available views:
     - work (query: Work-related documents and meetings)
     - 2024 (query: Documents from 2024)
   ```

### Mount and Browse the Filesystem

1. **Mount the virtual filesystem** (in a new terminal):
   ```bash
   llm-organizer mount -v
   ```

   Or specify a custom location:
   ```bash
   mkdir /tmp/my-organized-files
   llm-organizer mount -m /tmp/my-organized-files
   ```

2. **Browse your organized files**:
   ```bash
   # In another terminal
   cd ~/.local/share/llm-organizer/mount  # or your custom mount point

   # See the structure
   ls -la
   # drwxr-xr-x  views/
   # drwxr-xr-x  all/

   # List files in the "work" view
   ls views/work/
   # meeting-notes.txt

   # List files in the "2024" view
   ls views/2024/
   # meeting-notes.txt

   # Read a file (it's the actual file, not a copy)
   cat views/work/meeting-notes.txt
   ```

### Run Everything Together

The easiest way is to run both watcher and mount:

```bash
llm-organizer run -v
```

This will:
- Start watching your configured directories
- Mount the virtual filesystem
- Keep both running until you Ctrl+C

## Common Workflows

### Organize Your Downloads Folder

1. **Add Downloads to watch list**:
   ```toml
   [filesystem]
   watch_dirs = [
       "/Users/you/Documents",
       "/Users/you/Downloads",  # Add this
   ]
   ```

2. **Create useful views**:
   ```bash
   llm-organizer create-view "receipts" "Receipts and invoices"
   llm-organizer create-view "pdfs" "PDF documents"
   llm-organizer create-view "this-week" "Files from the past 7 days"
   ```

3. **Browse organized downloads**:
   ```bash
   ls ~/.local/share/llm-organizer/mount/views/receipts/
   ```

### Organize Research Papers

1. **Customize the organization prompt**:
   ```toml
   [organization]
   prompt = """
   Organize academic papers and research documents.
   Extract: authors, publication year, venue (conference/journal), research area
   Categories: By research field and year
   Tags: methodology, key concepts, datasets used
   """
   ```

2. **Create views**:
   ```bash
   llm-organizer create-view "ml-papers" "Machine learning research papers"
   llm-organizer create-view "2024-research" "Research papers from 2024"
   llm-organizer create-view "citations-needed" "Papers I need to cite"
   ```

### Organize Photos with Metadata

1. **Watch your Photos folder**:
   ```toml
   [filesystem]
   watch_dirs = ["/Users/you/Pictures"]
   ```

2. **Create views by date/event**:
   ```bash
   llm-organizer create-view "2024-photos" "Photos from 2024"
   llm-organizer create-view "vacation" "Vacation and travel photos"
   llm-organizer create-view "family" "Family photos and gatherings"
   ```

## Tips & Tricks

### 1. Test Your LLM Endpoint

Before running the organizer, verify your LLM works:

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "prompt": "Say hello",
    "stream": false
  }'
```

### 2. Start with Verbose Mode

Always use `-v` when testing:
```bash
llm-organizer run -v
```

This helps you see what's happening and debug issues.

### 3. Analyze Existing Files

Manually analyze files that were added before you started watching:

```bash
# Single file
llm-organizer analyze ~/Documents/important.pdf

# Multiple files (bash loop)
for file in ~/Documents/*.pdf; do
    llm-organizer analyze "$file"
done
```

### 4. Check the Database

You can inspect the SQLite database directly:

```bash
sqlite3 ~/.local/share/llm-organizer/metadata.db

# In SQLite prompt:
.tables
SELECT * FROM views;
SELECT path, file_type FROM files LIMIT 10;
.quit
```

### 5. Unmount the Filesystem

To unmount on macOS:
```bash
umount ~/.local/share/llm-organizer/mount
```

On Linux:
```bash
fusermount -u ~/.local/share/llm-organizer/mount
```

Or just Ctrl+C the mount process.

### 6. Reset Everything

If something goes wrong, start fresh:

```bash
# Stop all running llm-organizer processes
pkill llm-organizer

# Remove database
rm ~/.local/share/llm-organizer/metadata.db

# Unmount if needed
umount ~/.local/share/llm-organizer/mount  # macOS
fusermount -u ~/.local/share/llm-organizer/mount  # Linux

# Start again
llm-organizer run
```

## Troubleshooting

### "Connection refused" when calling LLM

Make sure your LLM server is running:

```bash
# For Ollama
ollama serve

# For llama.cpp
./server -m path/to/model.gguf
```

### FUSE mount fails

**macOS**: You may need to go to System Preferences → Security & Privacy and allow the macFUSE kernel extension.

**Linux**: Make sure you're in the `fuse` group:
```bash
sudo usermod -a -G fuse $USER
# Log out and back in
```

### Files not appearing in views

1. Check if file was analyzed:
   ```bash
   llm-organizer analyze /path/to/file
   ```

2. Verify view query is correct:
   ```bash
   llm-organizer list-views
   ```

3. Check logs with verbose mode:
   ```bash
   llm-organizer run -v
   ```

### Slow performance

1. Reduce LLM timeout:
   ```toml
   [llm]
   timeout_secs = 10  # Default is 30
   ```

2. Use a faster model:
   ```bash
   ollama pull phi  # Smaller, faster model
   ```

3. Increase cache TTL:
   ```toml
   [organization]
   cache_ttl_secs = 86400  # 24 hours
   ```

## Next Steps

- Read the full [README.md](README.md) for more features
- Check [TODO.md](TODO.md) for known issues and planned features
- Customize your `organization.prompt` for your use case
- Create more views for your workflow
- Integrate with your existing file management

## Need Help?

- Check the [README.md](README.md) for detailed documentation
- Look at [TODO.md](TODO.md) for known issues
- Open an issue on GitHub
- Check logs with `-v` flag for debugging

Happy organizing! 🎉
