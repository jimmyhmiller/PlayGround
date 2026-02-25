# Editor2

A WASM-plugin-based GUI editor for macOS using Metal/Skia rendering. Plugins are compiled to WebAssembly and loaded at runtime, allowing hot-reloading and sandboxed extensibility.

## Prerequisites

- Rust (stable)
- wasm32-wasip1 target: `rustup target add wasm32-wasip1`
- macOS (Metal rendering backend)

## Building

Build all WASM plugins:
```bash
cargo build-plugins
```

Build and run the editor:
```bash
cargo run -p editor2
```

Build the editor only:
```bash
cargo build -p editor2
```

## Project Structure

```
editor2/
  editor/            # Native host app (macOS, Metal/Skia)
  framework/         # Shared library used by both host and plugins
  headless_editor/   # Text buffer implementation
  plugins/
    code-editor/     # Code editing plugin with syntax support
    color-scheme/    # Color scheme management
    event-viewer/    # Event debugging/inspection
    multiple-widgets/# Multi-widget layout demo
    pane-editor/     # Pane-based editor layout
    process-test/    # External process integration (LSP, etc.)
    symbol-editor/   # Symbol/token editing
```

## Creating a New Plugin

1. Create a new crate under `plugins/`:
   ```bash
   cargo init --lib plugins/my-plugin
   ```

2. Set up `plugins/my-plugin/Cargo.toml`:
   ```toml
   [package]
   name = "my-plugin"
   version = "0.1.0"
   edition = "2021"

   [lib]
   crate-type = ['cdylib']

   [dependencies]
   framework = { path = "../../framework" }
   serde = { version = "1.0", features = ["derive"] }

   [features]
   default = []
   main = []
   ```

3. Add the crate to the workspace `Cargo.toml`:
   ```toml
   members = [
       # ... existing members
       "plugins/my-plugin",
   ]
   ```

4. Implement the `App` trait from `framework` and register with the `app!` macro.

5. Build: `cargo build-plugins`

## Usage

- **Cmd+O** — Open a file (creates a new code editor widget)
- **Drag & drop** — Drop `.wasm` files to load plugins, `.png` for images, or text files
- **Ctrl+drag** — Move widgets
- **Ctrl+Option+drag** — Resize widgets
- **Cmd+Ctrl+Option+click** — Delete a widget
- **Cmd+0** — Reset canvas zoom/scroll
