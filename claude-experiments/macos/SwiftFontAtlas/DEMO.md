# SwiftFontAtlas Demo

This directory contains a complete demonstration of the SwiftFontAtlas library showcasing all its key features.

## Running the Demo

### Command Line Demo (Recommended)

The easiest way to see SwiftFontAtlas in action:

```bash
cd FontAtlasDemo
swift run
```

This will run a comprehensive demonstration showing:

1. **Atlas Creation** - Creating a font atlas with SF Mono font
2. **ASCII Prerendering** - Rendering all printable ASCII characters
3. **Custom Text** - Rendering various text strings including Unicode
4. **Character Details** - Showing glyph metrics and atlas positions
5. **Cache Performance** - Demonstrating cache hit speedup
6. **Atlas Statistics** - Memory usage and utilization
7. **Growth Test** - Extended Unicode character rendering
8. **Thread Safety** - Concurrent rendering performance
9. **Atlas Analysis** - Detailed pixel utilization metrics

### Expected Output

The demo will show output like:

```
🚀 SwiftFontAtlas Library Demo
==============================

1️⃣ Creating Font Atlas
   Font: SF Mono, Size: 14pt, Atlas: 512x512
   ✅ Atlas created successfully!
   📏 Cell size: (8.0, 14.0)
   📊 Font metrics: ascent=10.8, descent=3.2

2️⃣ Prerendering ASCII Characters
   ✅ Rendered 95 ASCII characters
   ⏱️  Time: 3.856ms

3️⃣ Rendering Custom Text
   'Hello, World!': 13 characters in 0.004ms
   'SwiftFontAtlas 🚀': 15 characters in 0.008ms
   ...

🎉 All tests completed successfully!
```

## What the Demo Shows

### Core Features

- **Rectangle Bin Packing**: Efficient texture space utilization using Jukka Jylänki's algorithm
- **Thread-Safe Caching**: Multiple threads can safely access the atlas simultaneously
- **Auto-Resizing**: Atlas automatically grows when it runs out of space
- **Unicode Support**: Beyond ASCII characters including emoji and special symbols
- **Metal Integration**: Ready for GPU texture usage (texture creation methods included)
- **Performance**: Sub-millisecond character rendering with caching

### Performance Metrics

The demo measures and reports:
- Character rendering times (first render vs cached)
- Cache hit ratios and speedup factors
- Memory usage and pixel utilization
- Concurrent rendering throughput
- Atlas modification tracking

### Technical Details

- **Atlas Format**: Grayscale (1 byte per pixel) for text rendering
- **Default Size**: 512×512 pixels (expandable to 2048×2048 or larger)
- **Cell Metrics**: Font-specific cell dimensions for grid-based layouts
- **Modification Tracking**: Atomic counters for efficient GPU synchronization

## Interactive SwiftUI Demo (Optional)

A SwiftUI-based visual demo is also available in `FontAtlasDemoApp.swift`. This provides:

- Interactive font and size selection
- Real-time atlas visualization
- Zoom controls for detailed atlas inspection
- Live statistics and performance metrics
- Visual representation of glyph packing

To run the SwiftUI demo:

```bash
swift FontAtlasDemoApp.swift
```

Note: The SwiftUI demo requires more complex setup but provides visual feedback of how glyphs are packed into the atlas texture.

## Integration Example

Here's how you'd integrate SwiftFontAtlas into your own project:

```swift
import SwiftFontAtlas

// Create font atlas manager
let manager = try FontAtlasManager(
    fontName: "SF Mono",
    fontSize: 14.0,
    atlasSize: 512
)

// Pre-render common characters
manager.prerenderASCII()

// Render text
for character in "Hello, World!" {
    if let glyph = manager.renderCharacter(character) {
        // Use glyph.atlasX, glyph.atlasY for texture coordinates
        // Use glyph.width, glyph.height for quad dimensions
        // Use glyph.advanceX for text layout positioning
    }
}

// Create Metal texture
let texture = manager.createManagedTexture(device: metalDevice)
```

## Performance Characteristics

Based on the demo results:

- **ASCII Rendering**: ~95 characters in 3-4ms (first render)
- **Cache Performance**: 1000+ cache hits in ~0.2ms
- **Concurrent Access**: 1M+ operations per second
- **Memory Efficiency**: ~2-3% pixel utilization for typical usage
- **Unicode Support**: Extended characters render in ~1ms per character

This demonstrates that SwiftFontAtlas is ready for high-performance text rendering applications requiring efficient font atlas management.