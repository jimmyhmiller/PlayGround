# Installation Guide

## System Requirements

### macOS

1. **Install macFUSE**:
   ```bash
   brew install --cask macfuse
   ```

   **Important**: After installation, you need to:
   - Go to System Settings → Privacy & Security
   - Scroll down to allow "System Extension" from developer "Benjamin Fleischer"
   - Reboot your system

   **Alternative manual installation**:
   - Download from https://osxfuse.github.io/
   - Install the DMG package
   - Reboot

2. **Set PKG_CONFIG_PATH** (if needed):
   ```bash
   export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
   ```

   Add to your `~/.zshrc` or `~/.bashrc`:
   ```bash
   echo 'export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.zshrc
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    libfuse3-dev \
    fuse3 \
    pkg-config \
    build-essential
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install -y \
    fuse3-devel \
    fuse3 \
    pkg-config \
    gcc
```

### Linux (Arch)

```bash
sudo pacman -S fuse3 pkg-config base-devel
```

## Rust Installation

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
rustc --version
cargo --version
```

## Building LLM Organizer

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/llm-organizer
   cd llm-organizer
   ```

2. **Build the project**:
   ```bash
   cargo build --release
   ```

   This will take a few minutes on first build as it downloads and compiles dependencies.

3. **Install the binary** (optional):
   ```bash
   sudo cp target/release/llm-organizer /usr/local/bin/
   ```

   Or add to your PATH:
   ```bash
   export PATH="$PATH:$(pwd)/target/release"
   ```

4. **Verify installation**:
   ```bash
   llm-organizer --help
   ```

## LLM Setup

You need an LLM endpoint. Here are your options:

### Option 1: Ollama (Recommended for Local)

1. **Install Ollama**:
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull a model**:
   ```bash
   # Recommended models (choose one):
   ollama pull mistral      # 7B, good balance of speed/quality
   ollama pull llama2       # 7B, alternative
   ollama pull phi          # 2.7B, faster but less capable
   ollama pull mixtral      # 47B, highest quality but slower
   ```

3. **Start Ollama** (runs automatically as service, or manually):
   ```bash
   ollama serve
   ```

4. **Test it**:
   ```bash
   curl -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{
       "model": "mistral",
       "prompt": "Hello, world!",
       "stream": false
     }'
   ```

5. **Configure LLM Organizer**:
   ```toml
   [llm]
   endpoint = "http://localhost:11434/api/generate"
   model = "mistral"
   ```

### Option 2: llama.cpp Server

1. **Build llama.cpp**:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```

2. **Download a model** (GGUF format):
   ```bash
   # Example: download from HuggingFace
   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
   ```

3. **Start the server**:
   ```bash
   ./server -m mistral-7b-instruct-v0.2.Q4_K_M.gguf --port 8080
   ```

4. **Configure LLM Organizer**:
   ```toml
   [llm]
   endpoint = "http://localhost:8080/completion"
   ```

### Option 3: OpenAI API

1. **Get an API key** from https://platform.openai.com/

2. **Configure** (note: code may need modification for auth headers):
   ```toml
   [llm]
   endpoint = "https://api.openai.com/v1/completions"
   model = "gpt-3.5-turbo"
   ```

   **Note**: Current implementation doesn't include API key handling. You'll need to modify `src/llm/mod.rs` to add authorization headers.

### Option 4: Anthropic Claude

Similar to OpenAI, requires code modification for API key handling.

## Post-Installation

1. **Run first-time setup**:
   ```bash
   llm-organizer --help
   ```

   This creates the default config at `~/.config/llm-organizer/config.toml`.

2. **Edit configuration**:
   ```bash
   # Copy example config
   cp config.example.toml ~/.config/llm-organizer/config.toml

   # Edit it
   nano ~/.config/llm-organizer/config.toml
   ```

3. **Test analysis**:
   ```bash
   echo "This is a test document." > /tmp/test.txt
   llm-organizer analyze /tmp/test.txt
   ```

## Troubleshooting Installation

### macFUSE Not Found

**Symptom**:
```
The system library `osxfuse` required by crate `fuser` was not found.
```

**Solution**:
```bash
# Install macFUSE
brew install --cask macfuse

# Reboot
sudo reboot

# Set PKG_CONFIG_PATH
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"

# Try building again
cargo clean
cargo build --release
```

### Linux: FUSE Not Found

**Symptom**:
```
The system library `fuse` required by crate `fuser` was not found.
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libfuse3-dev fuse3 pkg-config

# Fedora
sudo dnf install fuse3-devel fuse3 pkg-config

# Try building again
cargo build --release
```

### Permission Denied When Mounting

**Linux**:
```bash
# Add yourself to fuse group
sudo usermod -a -G fuse $USER

# Log out and back in, or:
newgrp fuse
```

**macOS**:
- Go to System Settings → Privacy & Security
- Allow kernel extension from Benjamin Fleischer
- Reboot

### Rust Compilation Errors

Update Rust to latest stable:
```bash
rustup update stable
cargo clean
cargo build --release
```

### LLM Connection Issues

Test your LLM endpoint separately:

**Ollama**:
```bash
ollama list  # Check if models are pulled
ollama serve  # Start server if not running
curl http://localhost:11434/api/generate -d '{"model":"mistral","prompt":"test"}'
```

**llama.cpp**:
```bash
# Check if server is running
curl http://localhost:8080/health

# Test completion
curl http://localhost:8080/completion -d '{"prompt":"test"}'
```

## Uninstallation

1. **Stop all processes**:
   ```bash
   pkill llm-organizer
   ```

2. **Unmount filesystem** (if mounted):
   ```bash
   # macOS
   umount ~/.local/share/llm-organizer/mount

   # Linux
   fusermount -u ~/.local/share/llm-organizer/mount
   ```

3. **Remove binary**:
   ```bash
   sudo rm /usr/local/bin/llm-organizer
   ```

4. **Remove data** (optional):
   ```bash
   rm -rf ~/.config/llm-organizer
   rm -rf ~/.local/share/llm-organizer
   ```

## Next Steps

After installation:

1. Read [QUICKSTART.md](QUICKSTART.md) for getting started
2. Check [README.md](README.md) for full documentation
3. Review [config.example.toml](config.example.toml) for configuration options
