import Foundation
import CoreGraphics

/// Pixel formats supported by the atlas
public enum PixelFormat: UInt8 {
    /// 1 byte per pixel grayscale
    case grayscale = 0
    /// 3 bytes per pixel BGR
    case bgr = 1
    /// 4 bytes per pixel BGRA
    case bgra = 2
    
    /// Bytes per pixel for this format
    public var bytesPerPixel: Int {
        switch self {
        case .grayscale: return 1
        case .bgr: return 3
        case .bgra: return 4
        }
    }
}

/// A region within the texture atlas
public struct AtlasRegion: Equatable {
    public let x: UInt32
    public let y: UInt32
    public let width: UInt32
    public let height: UInt32
    
    public init(x: UInt32, y: UInt32, width: UInt32, height: UInt32) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

/// A rendered glyph with its atlas location and metrics
public struct RenderedGlyph: Equatable {
    /// Width of glyph in pixels
    public let width: UInt32
    
    /// Height of glyph in pixels
    public let height: UInt32
    
    /// Left bearing (horizontal offset from origin)
    public let offsetX: Int32
    
    /// Top bearing (vertical offset from baseline to top)
    public let offsetY: Int32
    
    /// Position in atlas
    public let atlasX: UInt32
    public let atlasY: UInt32
    
    /// Horizontal advance for text layout
    public let advanceX: Float
    
    public init(
        width: UInt32,
        height: UInt32,
        offsetX: Int32,
        offsetY: Int32,
        atlasX: UInt32,
        atlasY: UInt32,
        advanceX: Float
    ) {
        self.width = width
        self.height = height
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.atlasX = atlasX
        self.atlasY = atlasY
        self.advanceX = advanceX
    }
}

/// Key for caching glyphs
public struct GlyphKey: Hashable {
    public let codepoint: UInt32
    public let fontSize: Float
    public let fontName: String
    
    public init(codepoint: UInt32, fontSize: Float, fontName: String) {
        self.codepoint = codepoint
        self.fontSize = fontSize
        self.fontName = fontName
    }
    
    public init(character: Character, fontSize: Float, fontName: String) {
        // Get the first unicode scalar value
        self.codepoint = character.unicodeScalars.first?.value ?? 0
        self.fontSize = fontSize
        self.fontName = fontName
    }
}

/// Font metrics for grid-based layouts
public struct FontMetrics {
    public let cellWidth: CGFloat
    public let cellHeight: CGFloat
    public let ascent: CGFloat
    public let descent: CGFloat
    public let lineGap: CGFloat
    
    public init(
        cellWidth: CGFloat,
        cellHeight: CGFloat,
        ascent: CGFloat,
        descent: CGFloat,
        lineGap: CGFloat
    ) {
        self.cellWidth = cellWidth
        self.cellHeight = cellHeight
        self.ascent = ascent
        self.descent = descent
        self.lineGap = lineGap
    }
}

/// Errors that can occur in the font atlas system
public enum FontAtlasError: Error {
    /// Atlas cannot fit the desired region
    case atlasFull
    
    /// Failed to create Core Graphics context
    case contextCreationFailed
    
    /// Failed to create CTFont
    case fontCreationFailed
    
    /// Failed to get glyph for character
    case glyphNotFound
    
    /// Invalid atlas size (must be power of 2)
    case invalidAtlasSize
}