# Font Atlas Viewer

A visual SwiftUI application that demonstrates the SwiftFontAtlas library in action.

## Quick Start

```bash
cd FontAtlasViewer
swift run
```

This will open a macOS app window where you can:

## Features

### 🎛️ Interactive Controls
- **Font Selection**: Choose from SF Mono, Menlo, Helvetica, and more
- **Font Size**: Adjustable from 8pt to 72pt with live preview
- **Atlas Size**: 256×256, 512×512, or 1024×1024 pixels

### 🎨 Visual Atlas Display
- **Real-time visualization** of the font atlas texture
- **Zoom controls** (50% to 800%) with mouse wheel support
- **Grid overlay** showing character cell boundaries
- **Black background with white glyphs** for clear visibility

### ⚡ Live Rendering
- **ASCII Rendering**: Render all printable ASCII characters (32-126)
- **Custom Text**: Type any text and see it rendered into the atlas
- **Unicode Test**: Automatically test accented characters, Greek, arrows, symbols, and emoji
- **Clear Atlas**: Reset and start fresh

### 📊 Live Statistics
- Atlas dimensions and memory usage
- Glyph count and pixel utilization percentage
- Cell size and last operation timing
- Real-time performance metrics

### 📝 Activity Log
- Timestamped log of all operations
- Performance timing for each action
- Error reporting and status updates

## What You'll See

1. **Empty Atlas**: Initially shows a placeholder until you create an atlas
2. **Font Atlas Creation**: Black texture appears when you click "Create Atlas"
3. **Glyph Rendering**: White pixels appear as characters are rendered
4. **Efficient Packing**: See how the rectangle bin packing algorithm arranges glyphs
5. **Real-time Updates**: Statistics update as you add more characters

## Visual Features

### Atlas Visualization
- **Black background**: Represents empty atlas space
- **White pixels**: Show rendered glyph data
- **Zoom controls**: + / - / 1:1 buttons for detailed inspection
- **Grid overlay**: Blue lines showing character cell boundaries (when zoomed > 200%)

### Performance Monitoring
- **Memory usage**: Real-time memory consumption tracking
- **Utilization**: Percentage of atlas pixels actually used
- **Timing**: Millisecond precision for all operations
- **Glyph count**: Total characters rendered in the current atlas

## Try These Actions

1. **Start Simple**: Create a default atlas and render ASCII
2. **Test Fonts**: Try different fonts and sizes to see varying glyph shapes
3. **Unicode Fun**: Use the Unicode test to see emoji and special characters
4. **Performance**: Watch the millisecond timing for different operations
5. **Efficiency**: Observe how pixel utilization changes as you add glyphs
6. **Zoom In**: Get close to see individual glyph pixels and packing efficiency

## Technical Details

- **Real-time rendering**: Uses background queues for smooth UI
- **Memory visualization**: Direct display of atlas texture data
- **CoreText integration**: High-quality glyph rendering
- **Thread safety**: Safe concurrent access to atlas data
- **Efficient updates**: Only redraws when atlas changes

This visual demo shows the SwiftFontAtlas library working exactly as it would in a real application, with efficient rectangle packing and high-performance glyph caching.