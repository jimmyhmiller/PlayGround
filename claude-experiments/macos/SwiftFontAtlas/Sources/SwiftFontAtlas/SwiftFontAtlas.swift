/// SwiftFontAtlas - A high-performance font atlas library for Swift/Metal applications
///
/// Inspired by Ghostty's robust implementation, this library provides efficient
/// texture atlas management for font rendering in Metal-based applications.
///
/// # Usage
///
/// ```swift
/// // Create a font atlas manager
/// let manager = try FontAtlasManager(
///     fontName: "SF Mono",
///     fontSize: 14.0,
///     atlasSize: 512
/// )
///
/// // Pre-render common characters
/// manager.prerenderASCII()
///
/// // Render individual characters
/// if let glyph = manager.renderCharacter("A") {
///     // Use glyph for rendering
/// }
///
/// // Create Metal texture
/// let texture = manager.createManagedTexture(device: metalDevice)
/// ```
///
/// The library handles:
/// - Rectangle bin packing for efficient texture space usage
/// - Thread-safe glyph caching
/// - Automatic atlas resizing when full
/// - Metal texture integration with modification tracking
