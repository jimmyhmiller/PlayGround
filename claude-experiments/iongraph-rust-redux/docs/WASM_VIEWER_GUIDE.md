# WASM Viewer Guide

## Overview

IonGraph now has **three** HTML rendering modes:

1. **Static HTML** (`--html`): Pre-rendered SVG, large files (~18MB)
2. **WASM HTML** (`--wasm`): Embedded JSON + WASM, medium files (~9MB)
3. **WASM Viewer** (`--viewer`): Standalone viewer, tiny files (~370KB) ✨ **NEW!**

## Comparison

| Feature | Static HTML | WASM HTML | WASM Viewer |
|---------|-------------|-----------|-------------|
| **File Size** (mega-complex) | 18MB | 9.2MB | **370KB** |
| **JSON Data** | Embedded (pre-rendered) | Embedded | **Drag-and-drop** |
| **Reusability** | One file per JSON | One file per JSON | **Universal viewer** |
| **First Load** | Instant | ~50ms WASM init | ~50ms WASM init |
| **Rendering** | Pre-rendered | On-demand | On-demand |
| **Best For** | Sharing specific graphs | Sharing multiple passes | **General purpose** |

## Quick Start: WASM Viewer

### 1. Generate the Viewer (One Time)

```bash
# Build WASM binary (if not done already)
wasm-pack build --target web --out-dir pkg

# Build iongraph binary
cargo build --release --bin iongraph

# Generate the universal viewer
./target/release/iongraph --viewer iongraph-viewer.html
```

This creates a **single 370KB HTML file** that can render any Ion JSON file!

### 2. Use the Viewer

Simply open the HTML file in your browser:

```bash
open iongraph-viewer.html
```

You'll see a drag-and-drop interface:

```
🎯 IonGraph WASM Viewer

Drag and drop an Ion JSON file here

or

[Choose File]

Supported: Ion JSON files from SpiderMonkey JIT compiler
```

### 3. Load a JSON File

**Option 1: Drag and Drop**
- Drag any `.json` file from your file system onto the window
- The viewer automatically parses and renders it

**Option 2: File Picker**
- Click the "Choose File" button
- Select a JSON file from the dialog

### 4. Navigate

Once loaded, you get the full interactive UI:
- Function selector dropdown
- Pass sidebar
- Keyboard shortcuts (`←/→`, `↑/↓`)
- "Load Different File" button to switch files

## Use Cases

### WASM Viewer (`--viewer`) - **Recommended for Most Uses**

Use when you want:
- ✅ A single viewer for all your JSON files
- ✅ Minimal file size (370KB vs 9-18MB)
- ✅ Easy sharing (one viewer.html for everyone)
- ✅ Quick iteration (just drag new JSON files)

Perfect for:
- Development and debugging
- Sharing a viewer with your team
- Exploring multiple JSON files
- Local analysis of compiler output

### WASM HTML (`--wasm`)

Use when you want:
- ✅ Self-contained file (JSON + viewer)
- ✅ Share a specific file with someone
- ✅ Medium file size (9MB vs 18MB)

Perfect for:
- Bug reports (single file with specific data)
- Archiving specific compilation results
- Sharing on file hosting (Dropbox, Drive, etc.)

### Static HTML (`--html`)

Use when you want:
- ✅ Instant display (no rendering time)
- ✅ Maximum compatibility (older browsers)
- ✅ No JavaScript required

Perfect for:
- Print/PDF export
- Offline viewing without WASM support
- Embedding in documentation

## File Size Breakdown

### WASM Viewer (370KB)
```
WASM binary (base64):  341KB (256KB raw)
JavaScript glue code:   11KB
HTML + CSS:             10KB
UI code:                 8KB
─────────────────────────────
Total:                 ~370KB
```

### WASM HTML (9.2MB for mega-complex)
```
IonJSON data:          ~9MB
WASM binary (base64):  341KB
JavaScript glue code:   11KB
HTML + CSS:             10KB
UI code:                 8KB
─────────────────────────────
Total:                 ~9.2MB
```

### Static HTML (18MB for mega-complex)
```
Pre-rendered SVGs:     ~18MB (526 passes)
JavaScript:             20KB
HTML + CSS:             10KB
─────────────────────────────
Total:                 ~18MB
```

## Features

### Drag-and-Drop
- ✅ Drag JSON files from file system
- ✅ Visual feedback (blue highlight on drag-over)
- ✅ Error handling for invalid files
- ✅ Loading indicator during parsing

### File Picker
- ✅ Standard file dialog
- ✅ Filters for `.json` files
- ✅ Multiple ways to load files

### Dynamic Loading
- ✅ Parse JSON in browser
- ✅ No server required
- ✅ Switch between files without reloading page
- ✅ "Load Different File" button while viewing

### Full Interactivity
- ✅ Same UI as other WASM modes
- ✅ Keyboard shortcuts
- ✅ SVG caching for instant navigation
- ✅ Function/pass switching

## Advanced Usage

### Sharing the Viewer

The viewer is completely self-contained. You can:

1. **Host it on a server**:
   ```bash
   # Copy to web server
   cp iongraph-viewer.html /var/www/html/
   # Access at http://yourserver/iongraph-viewer.html
   ```

2. **Share via email/Slack**:
   - Single 370KB file
   - Recipients can use it offline
   - Works on all modern browsers

3. **Check into repo**:
   ```bash
   git add iongraph-viewer.html
   git commit -m "Add IonGraph viewer"
   # Team members can use it to analyze any Ion JSON
   ```

### Workflow Example

```bash
# 1. Generate viewer once
./target/release/iongraph --viewer viewer.html

# 2. Run your compiler to generate Ion JSON files
./your-compiler --dump-ion output1.json
./your-compiler --dump-ion output2.json
./your-compiler --dump-ion output3.json

# 3. Open viewer
open viewer.html

# 4. Drag-and-drop each JSON to analyze
#    No need to regenerate HTML for each file!
```

## Technical Details

### How It Works

1. **WASM Initialization** (~50ms on page load):
   - Decode base64 WASM binary
   - Initialize WebAssembly module
   - Set up WASM functions

2. **File Loading**:
   - User drags/selects JSON file
   - FileReader API reads file content
   - JSON.parse() parses the data
   - Store in memory as `ION_DATA`

3. **Rendering**:
   - User selects function/pass
   - Call WASM `render_pass_svg()` with JSON data
   - Display SVG in viewport
   - Cache for instant re-display

### Browser Compatibility

Works in all modern browsers:
- ✅ Chrome/Edge 57+ (2017+)
- ✅ Firefox 52+ (2017+)
- ✅ Safari 11+ (2017+)
- ✅ Opera 44+ (2017+)

### Security

The viewer is completely client-side:
- ✅ No server communication
- ✅ All processing in browser
- ✅ Files never uploaded
- ✅ Works offline

### Performance

| Operation | Time |
|-----------|------|
| WASM init | ~50ms (once) |
| Load JSON (10MB) | ~500ms |
| Parse JSON | ~200ms |
| First render | ~100-200ms |
| Cached render | <1ms |

## Troubleshooting

### "Error parsing JSON"
- Make sure you're loading a valid Ion JSON file
- Check that it matches the SpiderMonkey format
- Try validating with `jq . your-file.json`

### Viewer doesn't load
- Make sure you built the WASM binary first
- Check browser console for errors
- Ensure you're using a modern browser

### Rendering is slow
- Large files (>20MB) may take 1-2 seconds to parse
- First render of each pass takes ~100-200ms
- Subsequent renders are instant (cached)

## Examples

### Generate All Three Formats

```bash
# Static HTML (18MB, instant display)
./target/release/iongraph --html mega-complex.json static.html

# WASM HTML (9.2MB, embedded data)
./target/release/iongraph --wasm mega-complex.json embedded.html

# WASM Viewer (370KB, drag-and-drop)
./target/release/iongraph --viewer viewer.html

# Compare sizes
ls -lh *.html
# static.html    18M
# embedded.html  9.2M
# viewer.html    370K  ← Smallest!
```

### Development Workflow

```bash
# One-time setup
wasm-pack build --target web --out-dir pkg
cargo build --release --bin iongraph
./target/release/iongraph --viewer dev-viewer.html

# Daily use
open dev-viewer.html
# Then just drag-and-drop new JSON files as you generate them!
```

## Recommendations

### For Development
→ Use **WASM Viewer** (`--viewer`)
- Generate once, use forever
- Quick iteration on JSON files
- Minimal disk usage

### For Sharing Specific Results
→ Use **WASM HTML** (`--wasm`)
- Self-contained file
- Medium size (9MB)
- Easy to email/share

### For Maximum Compatibility
→ Use **Static HTML** (`--html`)
- Works everywhere
- No WASM required
- Instant display

## Future Enhancements

Possible improvements:
1. **URL loading**: Load JSON from URL query parameter
2. **Recent files**: Remember recently loaded files
3. **LocalStorage**: Save viewed files for quick access
4. **Diff view**: Compare two JSON files side-by-side
5. **Export**: Save specific passes as PNG/PDF

## Conclusion

The WASM Viewer is the **recommended approach for most users**:

✅ **370KB** file size (50x smaller than static HTML)
✅ **Universal** - works with any Ion JSON file
✅ **Convenient** - drag-and-drop interface
✅ **Shareable** - single file for whole team
✅ **Offline** - no server required

Generate it once, use it forever! 🚀
