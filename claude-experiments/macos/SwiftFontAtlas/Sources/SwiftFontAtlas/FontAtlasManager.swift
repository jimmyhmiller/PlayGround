import Foundation
import CoreText
import CoreGraphics
import os.lock

/// Thread-safe manager for font atlas operations
public class FontAtlasManager {
    /// Grayscale atlas for regular text
    private let grayscaleAtlas: FontAtlas
    
    /// Cache of rendered glyphs
    private var glyphCache: [GlyphKey: RenderedGlyph] = [:]
    
    /// Lock for thread-safe access
    private let lock = NSLock()
    
    /// The font being used
    private let font: CTFont
    
    /// Font name for cache keys
    private let fontName: String
    
    /// Font size for cache keys
    private let fontSize: Float
    
    /// Calculated font metrics
    public let metrics: FontMetrics
    
    /// Cell size for grid-based layouts
    public var cellSize: CGSize {
        CGSize(width: metrics.cellWidth, height: metrics.cellHeight)
    }
    
    /// Initialize a new font atlas manager
    /// - Parameters:
    ///   - fontName: Name of the font to use
    ///   - fontSize: Size of the font in points
    ///   - atlasSize: Initial size of the atlas (default: 512)
    public init(fontName: String, fontSize: Float, atlasSize: UInt32 = 512) throws {
        // Validate font size
        guard fontSize > 0 else {
            throw FontAtlasError.fontCreationFailed
        }
        
        self.fontName = fontName
        self.fontSize = fontSize
        
        // Create CTFont with fallback
        let font: CTFont
        let createdFont = CTFont(fontName as CFString, size: CGFloat(fontSize))
        let actualName = CTFontCopyPostScriptName(createdFont) as String
        
        // Check if we got the requested font (not a fallback)
        if actualName.lowercased().contains(fontName.lowercased().replacingOccurrences(of: " ", with: "")) ||
           fontName.lowercased().contains("system") {
            font = createdFont
        } else {
            // Fallback to system monospace font if the requested font doesn't exist
            print("Warning: Font '\(fontName)' not found, falling back to Menlo")
            font = CTFont("Menlo" as CFString, size: CGFloat(fontSize))
        }
        self.font = font
        
        // Calculate metrics
        self.metrics = GlyphRenderer.calculateMetrics(for: font)
        
        // Create atlas
        self.grayscaleAtlas = try FontAtlas(size: atlasSize, format: .grayscale)
        
        // Pre-allocate cache capacity
        glyphCache.reserveCapacity(128)
    }
    
    /// Render a character and cache the result
    /// - Parameter character: The character to render
    /// - Returns: The rendered glyph, or nil if the character cannot be rendered
    public func renderCharacter(_ character: Character) -> RenderedGlyph? {
        let key = GlyphKey(character: character, fontSize: fontSize, fontName: fontName)
        
        // Fast path: check cache with read lock
        lock.lock()
        if let cached = glyphCache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        
        // Slow path: render the glyph
        return renderAndCache(character: character, key: key)
    }
    
    /// Render a codepoint and cache the result
    /// - Parameter codepoint: The Unicode codepoint to render
    /// - Returns: The rendered glyph, or nil if the codepoint cannot be rendered
    public func renderCodepoint(_ codepoint: UInt32) -> RenderedGlyph? {
        // Convert codepoint to Character if possible
        guard let scalar = UnicodeScalar(codepoint) else { return nil }
        let character = Character(scalar)
        return renderCharacter(character)
    }
    
    /// Get the current atlas for reading
    /// - Parameter block: Block to execute with atlas access
    public func withAtlas<T>(_ block: (FontAtlas) throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try block(grayscaleAtlas)
    }
    
    /// Check if the atlas has been modified since a given count
    /// - Parameter since: The modification count to compare against
    /// - Returns: True if the atlas has been modified
    public func isModified(since: UInt64) -> Bool {
        return grayscaleAtlas.modificationCount.withLock { $0 } > since
    }
    
    /// Get the current modification count
    public var modificationCount: UInt64 {
        grayscaleAtlas.modificationCount.withLock { $0 }
    }
    
    /// Check if the atlas has been resized since a given count
    /// - Parameter since: The resize count to compare against
    /// - Returns: True if the atlas has been resized
    public func isResized(since: UInt64) -> Bool {
        return grayscaleAtlas.resizeCount.withLock { $0 } > since
    }
    
    /// Get the current resize count
    public var resizeCount: UInt64 {
        grayscaleAtlas.resizeCount.withLock { $0 }
    }
    
    // MARK: - Private Methods
    
    private func renderAndCache(character: Character, key: GlyphKey) -> RenderedGlyph? {
        // Try to render the character
        guard let (bitmapData, metrics) = try? GlyphRenderer.renderCharacter(
            font: font,
            character: character
        ) else {
            return nil
        }
        
        // Handle empty glyphs (like space)
        if metrics.width == 0 || metrics.height == 0 {
            let glyph = RenderedGlyph(
                width: 0,
                height: 0,
                offsetX: 0,
                offsetY: 0,
                atlasX: 0,
                atlasY: 0,
                advanceX: metrics.advanceX
            )
            
            // Cache even empty glyphs
            lock.lock()
            glyphCache[key] = glyph
            lock.unlock()
            
            return glyph
        }
        
        // Acquire write lock for atlas modification
        lock.lock()
        defer { lock.unlock() }
        
        // Check cache again in case another thread rendered it
        if let cached = glyphCache[key] {
            return cached
        }
        
        // Try to allocate space in atlas
        let region: AtlasRegion
        do {
            region = try grayscaleAtlas.reserve(
                width: metrics.width,
                height: metrics.height
            )
        } catch FontAtlasError.atlasFull {
            // Atlas is full, try to grow it
            let newSize = grayscaleAtlas.size * 2
            do {
                try grayscaleAtlas.grow(to: newSize)
                region = try grayscaleAtlas.reserve(
                    width: metrics.width,
                    height: metrics.height
                )
            } catch {
                // Failed to grow or still doesn't fit
                return nil
            }
        } catch {
            return nil
        }
        
        // Copy bitmap data to atlas
        grayscaleAtlas.set(region: region, data: bitmapData)
        
        // Create rendered glyph
        let glyph = RenderedGlyph(
            width: metrics.width,
            height: metrics.height,
            offsetX: metrics.offsetX,
            offsetY: metrics.offsetY,
            atlasX: region.x,
            atlasY: region.y,
            advanceX: metrics.advanceX
        )
        
        // Cache the result
        glyphCache[key] = glyph
        
        return glyph
    }
}

// MARK: - Batch Operations

extension FontAtlasManager {
    /// Pre-render a string of characters
    /// - Parameter string: The string to pre-render
    /// - Returns: Number of glyphs successfully rendered
    @discardableResult
    public func prerenderString(_ string: String) -> Int {
        var count = 0
        for character in string {
            if renderCharacter(character) != nil {
                count += 1
            }
        }
        return count
    }
    
    /// Pre-render common ASCII characters
    @discardableResult
    public func prerenderASCII() -> Int {
        var count = 0
        // Printable ASCII range (32-126)
        for codepoint in 32...126 {
            if let scalar = UnicodeScalar(codepoint) {
                let character = Character(scalar)
                if renderCharacter(character) != nil {
                    count += 1
                }
            }
        }
        return count
    }
}

// MARK: - Text Layout and Positioning

extension FontAtlasManager {
    /// Calculate the baseline position for text rendering in a CGContext
    /// - Parameters:
    ///   - contextHeight: Height of the CGContext
    ///   - topMargin: Margin from the top of the context
    /// - Returns: Y coordinate for the baseline position
    public func baselinePosition(contextHeight: CGFloat, topMargin: CGFloat = 20) -> CGFloat {
        // CoreGraphics uses bottom-left origin
        // Baseline should be positioned from the bottom, accounting for descent
        return contextHeight - topMargin - metrics.descent
    }
    
    /// Calculate the proper position for drawing a glyph relative to baseline
    /// - Parameters:
    ///   - glyph: The rendered glyph to position
    ///   - baselineX: X coordinate of the baseline
    ///   - baselineY: Y coordinate of the baseline
    /// - Returns: (x, y) coordinates where the glyph should be drawn
    public func glyphPosition(for glyph: RenderedGlyph, baselineX: CGFloat, baselineY: CGFloat) -> (x: CGFloat, y: CGFloat) {
        let x = baselineX + CGFloat(glyph.offsetX)
        let y = baselineY + CGFloat(glyph.offsetY)
        return (x: x, y: y)
    }
    
    /// Calculate positions for all characters in a line of text
    /// - Parameters:
    ///   - text: The text to layout
    ///   - baselineX: Starting X coordinate
    ///   - baselineY: Baseline Y coordinate
    /// - Returns: Array of (character, glyph, x, y) tuples for positioned glyphs
    public func layoutLine(_ text: String, baselineX: CGFloat, baselineY: CGFloat) -> [(character: Character, glyph: RenderedGlyph, x: CGFloat, y: CGFloat)] {
        var result: [(Character, RenderedGlyph, CGFloat, CGFloat)] = []
        var currentX = baselineX
        
        for character in text {
            if let glyph = renderCharacter(character) {
                let position = glyphPosition(for: glyph, baselineX: currentX, baselineY: baselineY)
                result.append((character, glyph, position.x, position.y))
                currentX += CGFloat(glyph.advanceX)
            }
        }
        
        return result
    }
    
    /// Calculate line height for multi-line text
    public var lineHeight: CGFloat {
        return metrics.cellHeight
    }
    
    /// Calculate the width of a line of text
    /// - Parameter text: The text to measure
    /// - Returns: Width of the text line in points
    public func lineWidth(_ text: String) -> CGFloat {
        var width: CGFloat = 0
        for character in text {
            if let glyph = renderCharacter(character) {
                width += CGFloat(glyph.advanceX)
            }
        }
        return width
    }
}