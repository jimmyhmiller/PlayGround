# SwiftFontAtlas

A high-performance font atlas library for Swift/Metal applications, inspired by [Ghostty's](https://github.com/ghostty-org/ghostty) robust implementation.

## Features

- 🎯 **Rectangle bin packing** for optimal texture space utilization
- 🔒 **Thread-safe** glyph caching with read-heavy optimization  
- 📈 **Automatic atlas resizing** when full
- ⚡ **Metal integration** with modification tracking
- 🖥️ **CoreText rendering** for high-quality glyph output
- 🏃‍♂️ **Performance optimized** with atomic counters and efficient algorithms

## Requirements

- macOS 14.0+ or iOS 17.0+
- Swift 6.1+
- Xcode 16.0+

## Installation

### Swift Package Manager

Add this to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/SwiftFontAtlas.git", from: "1.0.0")
]
```

Or add it through Xcode:
1. File → Add Package Dependencies
2. Enter the repository URL
3. Choose your version requirements

## Quick Start

```swift
import SwiftFontAtlas

// Create a font atlas manager
let manager = try FontAtlasManager(
    fontName: "SF Mono",
    fontSize: 14.0,
    atlasSize: 512
)

// Pre-render common characters for better performance
manager.prerenderASCII()

// Render individual characters
if let glyph = manager.renderCharacter("A") {
    print("Glyph 'A' at atlas position (\(glyph.atlasX), \(glyph.atlasY))")
    print("Size: \(glyph.width) x \(glyph.height)")
    print("Advance: \(glyph.advanceX)")
}

// Create Metal texture (if using Metal)
let texture = manager.createManagedTexture(device: metalDevice)
```

## Architecture

### Core Components

1. **FontAtlas** - Implements rectangle bin packing for texture management
2. **FontAtlasManager** - Thread-safe coordinator with glyph caching
3. **GlyphRenderer** - CoreText-based glyph rendering
4. **FontAtlasTexture** - Metal integration with automatic updates

### Rectangle Bin Packing

The library uses Jukka Jylänki's rectangle bin packing algorithm to efficiently pack glyphs into the texture atlas. This ensures optimal space utilization and minimal texture memory usage.

### Thread Safety

- Read-write locks for cache access (optimized for read-heavy workloads)
- Atomic counters for modification tracking
- Safe concurrent glyph rendering

## Usage Examples

### Basic Text Rendering

```swift
// Initialize with your preferred font
let manager = try FontAtlasManager(
    fontName: "Helvetica",
    fontSize: 12.0
)

// Render a string
let text = "Hello, World!"
for character in text {
    if let glyph = manager.renderCharacter(character) {
        // Use glyph for your rendering pipeline
        renderGlyph(glyph, at: position)
        position.x += glyph.advanceX
    }
}
```

### Metal Integration

```swift
import Metal
import MetalKit

class MyRenderer {
    let manager: FontAtlasManager
    let texture: FontAtlasTexture
    
    init(device: MTLDevice) throws {
        manager = try FontAtlasManager(fontName: "SF Mono", fontSize: 14.0)
        texture = manager.createManagedTexture(device: device)
    }
    
    func render() {
        // Get current texture (auto-updates as needed)
        guard let metalTexture = texture.metalTexture else { return }
        
        // Use texture in your Metal shaders
        renderEncoder.setFragmentTexture(metalTexture, index: 0)
    }
}
```

### Performance Optimization

```swift
// Pre-render commonly used characters
manager.prerenderASCII()  // ASCII 32-126
manager.prerenderString("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

// Check modification status for efficient GPU updates
if manager.isModified(since: lastModificationCount) {
    updateGPUTexture()
    lastModificationCount = manager.modificationCount
}
```

## API Reference

### FontAtlasManager

The main interface for font atlas operations.

```swift
public class FontAtlasManager {
    public init(fontName: String, fontSize: Float, atlasSize: UInt32 = 512) throws
    public func renderCharacter(_ character: Character) -> RenderedGlyph?
    public func renderCodepoint(_ codepoint: UInt32) -> RenderedGlyph?
    public func prerenderASCII() -> Int
    public func prerenderString(_ string: String) -> Int
    public var metrics: FontMetrics { get }
    public var cellSize: CGSize { get }
    public var modificationCount: UInt64 { get }
}
```

### RenderedGlyph

Information about a rendered glyph in the atlas.

```swift
public struct RenderedGlyph {
    public let width: UInt32      // Glyph width in pixels
    public let height: UInt32     // Glyph height in pixels  
    public let offsetX: Int32     // Left bearing
    public let offsetY: Int32     // Top bearing
    public let atlasX: UInt32     // X position in atlas
    public let atlasY: UInt32     // Y position in atlas
    public let advanceX: Float    // Horizontal advance
}
```

### FontAtlas

Low-level atlas management with rectangle bin packing.

```swift
public class FontAtlas {
    public init(size: UInt32, format: PixelFormat) throws
    public func reserve(width: UInt32, height: UInt32) throws -> AtlasRegion
    public func set(region: AtlasRegion, data: Data)
    public func grow(to newSize: UInt32) throws
    public var size: UInt32 { get }
    public var data: Data { get }
}
```

## Performance

The library is optimized for performance with:

- **O(n) rectangle packing** with efficient merging
- **Thread-safe caching** to avoid redundant renders
- **Atomic modification tracking** to minimize GPU uploads
- **Read-heavy locking** for concurrent access
- **Automatic atlas growth** to handle large character sets

Typical performance characteristics:
- Cache hit: ~0.001ms per character
- Cache miss: ~0.1-1ms per character (depending on glyph complexity)
- Atlas space utilization: 85-95% depending on glyph sizes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

This library's architecture is heavily inspired by [Ghostty](https://github.com/ghostty-org/ghostty), a GPU-accelerated terminal emulator. The rectangle bin packing algorithm is based on:

- "A Thousand Ways to Pack the Bin - A Practical Approach to Two-Dimensional Rectangle Bin Packing" by Jukka Jylänki
- Nicolas P. Rougier's freetype-gl project  
- Jukka's C++ implementation: https://github.com/juj/RectangleBinPack

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request