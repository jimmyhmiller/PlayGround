# Swift Font Atlas Library

A high-performance font atlas library for Swift/Metal applications, inspired by Ghostty's robust implementation.

## Credits and Attribution

This library's architecture and implementation approach is heavily inspired by [Ghostty](https://github.com/ghostty-org/ghostty), a GPU-accelerated terminal emulator. The rectangle bin packing algorithm implementation is based on Ghostty's Zig implementation, which itself is based on:

- "A Thousand Ways to Pack the Bin - A Practical Approach to Two-Dimensional Rectangle Bin Packing" by Jukka Jylänki
- Nicolas P. Rougier's freetype-gl project
- Jukka's C++ implementation: https://github.com/juj/RectangleBinPack

### Ghostty License

Ghostty is licensed under the MIT License:

```
MIT License

Copyright (c) 2024 Mitchell Hashimoto, Ghostty contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Overview

This library implements a texture atlas system for efficient font rendering in Metal-based applications. It uses rectangle bin packing for optimal texture space utilization and provides thread-safe glyph caching.

## Architecture

### Core Components

#### 1. FontAtlas
The primary class implementing rectangle bin packing algorithm based on Jukka Jylänki's approach.

```swift
class FontAtlas {
    private(set) var data: Data
    private(set) var size: UInt32
    private var nodes: [Node]
    let format: PixelFormat
    
    // Thread-safe modification tracking
    private(set) var modificationCount: AtomicUInt64
    private(set) var resizeCount: AtomicUInt64
}
```

Key features:
- Efficient rectangle packing with automatic merging of adjacent free space
- 1-pixel border to prevent texture bleeding
- Automatic resizing when full
- Atomic counters for tracking modifications

#### 2. FontAtlasManager
Thread-safe coordinator managing font rendering state with caching.

```swift
class FontAtlasManager {
    private let grayscaleAtlas: FontAtlas
    private let colorAtlas: FontAtlas?  // Optional for initial version
    private var glyphCache: [GlyphKey: RenderedGlyph]
    private let lock = NSLock()  // or os_unfair_lock for performance
    
    // Font metrics
    let cellWidth: Float
    let cellHeight: Float
}
```

Responsibilities:
- Manages grayscale (and optionally color) atlases
- Caches rendered glyphs for performance
- Thread-safe access with read-heavy optimization
- Calculates font metrics for grid-based layouts

#### 3. Data Structures

```swift
struct AtlasRegion {
    let x, y, width, height: UInt32
}

struct RenderedGlyph {
    let width, height: UInt32
    let offsetX, offsetY: Int32  // Glyph bearings
    let atlasX, atlasY: UInt32   // Position in atlas
    let advanceX: Float          // Horizontal advance
}

struct GlyphKey: Hashable {
    let character: Character  // or UInt32 codepoint
    let fontSize: Float
    let fontName: String
}
```

#### 4. Metal Integration

```swift
protocol FontAtlasTextureProvider {
    func createTexture(device: MTLDevice) -> MTLTexture
    func updateTexture(_ texture: MTLTexture)
    var needsTextureUpdate: Bool { get }
}
```

## Implementation Details

### Rectangle Bin Packing Algorithm

Based on Ghostty's implementation of Jylänki's algorithm:

1. Maintain a list of free rectangles (nodes) in the atlas
2. For each glyph allocation request:
   - Find the best-fitting node using best-height heuristic
   - Split the node and update the free list
   - Merge adjacent nodes with same Y coordinate
3. Grow atlas by doubling size when full

### Thread Safety Strategy

- Use read-write lock for cache access (optimized for reads)
- Atomic counters for modification tracking
- Lock-free checks for texture update needs
- Safe concurrent glyph rendering with proper synchronization

### CoreText Integration

```swift
extension FontAtlasManager {
    func renderGlyph(_ character: Character, font: CTFont) -> RenderedGlyph? {
        // 1. Check cache
        // 2. Get glyph from CoreText
        // 3. Render to bitmap
        // 4. Pack into atlas
        // 5. Cache and return
    }
}
```

### Metal Texture Management

1. **Pixel Formats**:
   - `.r8Unorm` for grayscale text
   - `.bgra8Unorm_sRGB` for color (future)

2. **Storage Mode**:
   - `.storageModeShared` for unified memory architecture
   - `.storageModeManaged` for discrete GPU systems

3. **Update Strategy**:
   - Track modifications with atomic counter
   - Only upload to GPU when counter changes
   - Support partial updates for efficiency

## API Design

### Coordinate System Support

The library now supports both top-left (Metal/screen) and bottom-left (CoreGraphics) coordinate systems:

```swift
// For Metal rendering (top-left origin)
let fontManager = try FontAtlasManager(
    fontName: "SF Mono",
    fontSize: 14.0,
    coordinateOrigin: .topLeft
)

// For CoreGraphics (bottom-left origin) 
let fontManager = try FontAtlasManager(
    fontName: "SF Mono",
    fontSize: 14.0,
    coordinateOrigin: .bottomLeft  // default
)
```

### High-Level API (Recommended)

The new `TextRenderer` class provides a simple, high-level API that handles all Metal setup:

```swift
// Create text renderer
let textRenderer = try TextRenderer(
    device: metalDevice,
    fontName: "SF Mono",
    fontSize: 16.0,
    coordinateOrigin: .topLeft  // Metal coordinates
)

// In your render loop
textRenderer.setProjectionMatrix(using: renderEncoder, viewportSize: view.bounds.size)

// Draw text - one line!
textRenderer.drawText(
    "Hello, World!",
    at: CGPoint(x: 20, y: 50),
    color: simd_float4(1, 1, 1, 1),
    using: renderEncoder
)

// Draw multiline text with alignment
textRenderer.drawText(
    "Line 1\nLine 2\nLine 3",
    in: CGRect(x: 20, y: 100, width: 300, height: 100),
    alignment: .center,
    using: renderEncoder
)

// Draw wrapped text
textRenderer.drawWrappedText(
    "This is a long text that will be automatically wrapped to fit within the specified rectangle.",
    in: CGRect(x: 20, y: 200, width: 200, height: 100),
    wrapMode: .word,
    alignment: .left,
    using: renderEncoder
)
```

### Simple Metal Integration

Use the `MetalTextHelper` for even simpler setup:

```swift
// Create a text view
let textView = MetalTextHelper.createTextView(
    frame: view.bounds,
    device: metalDevice
)

// Use the simple delegate
let delegate = ExampleTextDelegate(metalKitView: textView)
textView.delegate = delegate

// Or create your own delegate
class MyTextDelegate: MetalTextHelper.SimpleTextDelegate {
    override var textToRender: String {
        return "Custom text here"
    }
    
    override var textPosition: CGPoint {
        return CGPoint(x: 50, y: 100)
    }
    
    override var textColor: simd_float4 {
        return simd_float4(1, 0, 0, 1)  // Red
    }
}
```

### Low-Level API (Advanced)

For fine-grained control, use the FontAtlasManager directly:

```swift
let fontManager = try FontAtlasManager(
    fontName: "SF Mono",
    fontSize: 14.0,
    atlasSize: 512,
    coordinateOrigin: .topLeft
)

// Render individual characters
for char in "Hello, World!" {
    if let glyph = fontManager.renderCharacter(char) {
        // Use glyph data for custom vertex generation
        let position = fontManager.glyphPosition(
            for: glyph,
            baselineX: currentX,
            baselineY: baselineY
        )
        // Generate vertices...
    }
}

// Manual texture management
let texture = fontManager.createManagedTexture(device: metalDevice)
if fontManager.isModified(since: lastModCount) {
    fontManager.updateTexture(texture)
}
```

### Text Layout

The `TextLayout` helper provides advanced text layout features:

```swift
let layout = TextLayout(fontManager: fontManager)

// Layout text with word wrapping
let lines = layout.layoutText(
    "Long text that needs wrapping",
    maxWidth: 300,
    wrapMode: .word
)

// Calculate text bounds
let bounds = layout.textBounds(
    "Measure this text",
    maxWidth: 200,
    wrapMode: .character
)

// Process laid out lines
for line in lines {
    print("Line: \(line.text), width: \(line.width)")
    for (char, glyph) in line.glyphs {
        // Process each glyph
    }
}
```

## Implementation Phases

### Phase 1: Core Atlas (MVP) ✅
- [x] Study Ghostty implementation
- [x] Implement rectangle bin packing algorithm
- [x] Create FontAtlas class with basic operations
- [x] Add CoreText glyph rendering
- [x] Support ASCII characters only
- [x] Basic Metal texture creation

### Phase 2: Enhanced Features ✅
- [x] Add Unicode support
- [x] Implement automatic atlas resizing
- [x] Add thread-safe caching system
- [x] Optimize Metal texture updates
- [x] Add font metrics calculation
- [x] Support monospaced fonts properly
- [x] Add coordinate system configuration (topLeft/bottomLeft)
- [x] Create high-level TextRenderer API
- [x] Add text layout and wrapping support
- [x] Create Metal helper utilities

### Phase 3: Advanced Features
- [ ] Color emoji support (separate BGRA atlas)
- [ ] Subpixel rendering options
- [ ] Font fallback system
- [ ] Performance optimizations
- [ ] Comprehensive test suite
- [ ] Signed distance field rendering
- [ ] Custom shader effects support

## Key Insights from Ghostty

1. **Atlas Management**:
   - Keep 1-pixel border to prevent sampling artifacts
   - Use power-of-2 sizes for compatibility
   - Start at 512x512, double when full

2. **Caching Strategy**:
   - Cache both codepoint→font lookups and rendered glyphs
   - Use efficient hash keys for fast lookups
   - Separate caches for different rendering modes

3. **Performance**:
   - Atomic counters avoid unnecessary GPU uploads
   - Read-heavy locking strategy for concurrent access
   - Batch glyph renders when possible

4. **Metal Specifics**:
   - Use `.writeCombined` CPU cache mode for upload-only data
   - Linear filtering for constrained glyphs
   - Nearest filtering for pixel-perfect rendering

## Testing Strategy

1. **Unit Tests**:
   - Rectangle packing algorithm correctness
   - Thread safety under concurrent access
   - Cache hit/miss scenarios

2. **Integration Tests**:
   - CoreText rendering accuracy
   - Metal texture updates
   - Memory usage under stress

3. **Performance Tests**:
   - Glyph rendering throughput
   - Cache efficiency metrics
   - Atlas space utilization

## Future Enhancements

- Signed distance field (SDF) rendering for scalable text
- Multi-channel SDF for sharper corners
- Persistent atlas caching to disk
- Custom shader effects support
- Integration with SwiftUI/AppKit