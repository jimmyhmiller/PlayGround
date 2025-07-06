import Foundation
import CoreText
import CoreGraphics

/// Handles rendering of glyphs using CoreText
public struct GlyphRenderer {
    
    /// Render a glyph into a bitmap
    /// - Parameters:
    ///   - font: The CTFont to use
    ///   - glyph: The glyph ID to render
    /// - Returns: Bitmap data and glyph metrics
    static func renderGlyph(font: CTFont, glyph: CGGlyph) throws -> (data: Data, metrics: GlyphMetrics) {
        // Get glyph bounding box
        var mutableGlyph = glyph
        var glyphRect = CGRect.zero
        CTFontGetBoundingRectsForGlyphs(font, .default, &mutableGlyph, &glyphRect, 1)
        
        // Get advance width
        var advance = CGSize.zero
        CTFontGetAdvancesForGlyphs(font, .default, &mutableGlyph, &advance, 1)
        
        // Get font metrics for proper padding calculation
        let ascent = CTFontGetAscent(font)
        let descent = CTFontGetDescent(font)
        
        // Calculate padding based on font metrics (ensure we capture descenders and antialiasing)
        let padding: CGFloat = max(3, ceil(max(ascent, descent) * 0.15))
        let width = Int(ceil(max(glyphRect.width + padding * 2, 1)))
        let height = Int(ceil(max(glyphRect.height + padding * 2, 1)))
        
        // Handle empty glyphs (like space)
        if width <= 0 || height <= 0 {
            return (
                data: Data(),
                metrics: GlyphMetrics(
                    width: 0,
                    height: 0,
                    offsetX: 0,
                    offsetY: 0,
                    advanceX: Float(advance.width)
                )
            )
        }
        
        // Create bitmap context
        let bytesPerRow = width
        var pixelData = Data(count: width * height)
        
        guard let context = CGContext(
            data: pixelData.withUnsafeMutableBytes { $0.baseAddress },
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            throw FontAtlasError.contextCreationFailed
        }
        
        // Configure context for high-quality rendering
        context.setAllowsAntialiasing(true)
        context.setShouldAntialias(true)
        context.setAllowsFontSmoothing(true)
        context.setShouldSmoothFonts(true)
        
        // Clear background
        context.setFillColor(CGColor(gray: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Set text color to white
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        
        // Calculate drawing position for CoreGraphics bottom-up coordinate system
        let drawX = padding - glyphRect.minX
        let drawY = padding - glyphRect.minY
        
        // Draw the glyph
        var position = CGPoint(x: drawX, y: drawY)
        CTFontDrawGlyphs(font, &mutableGlyph, &position, 1, context)
        
        // Create glyph metrics
        // offsetX is the left bearing (distance from origin to left edge of glyph)
        // offsetY is the bearing from baseline to glyph bottom (negative for descenders)
        let metrics = GlyphMetrics(
            width: UInt32(width),
            height: UInt32(height),
            offsetX: Int32(round(glyphRect.minX - padding)),
            offsetY: Int32(round(glyphRect.minY - padding)), // Bottom of glyph relative to baseline
            advanceX: Float(advance.width)
        )
        
        return (data: pixelData, metrics: metrics)
    }
    
    /// Render a character into a bitmap
    /// - Parameters:
    ///   - font: The CTFont to use
    ///   - character: The character to render
    /// - Returns: Bitmap data and glyph metrics, or nil if glyph not found
    static func renderCharacter(font: CTFont, character: Character) throws -> (data: Data, metrics: GlyphMetrics)? {
        // Convert character to UTF-16
        let utf16 = Array(character.utf16)
        guard !utf16.isEmpty else { return nil }
        
        // Get glyph for character
        var glyphs = [CGGlyph](repeating: 0, count: utf16.count)
        let success = CTFontGetGlyphsForCharacters(font, utf16, &glyphs, utf16.count)
        
        guard success, let glyph = glyphs.first, glyph != 0 else {
            return nil
        }
        
        return try renderGlyph(font: font, glyph: glyph)
    }
    
    /// Calculate font metrics for a given font
    public static func calculateMetrics(for font: CTFont) -> FontMetrics {
        let ascent = CTFontGetAscent(font)
        let descent = CTFontGetDescent(font)
        let leading = CTFontGetLeading(font)
        
        // For monospace fonts, we can use the advance of any character
        // Let's use "0" as it should exist in all fonts
        var glyph: CGGlyph = 0
        let chars: [UniChar] = [48] // "0"
        CTFontGetGlyphsForCharacters(font, chars, &glyph, 1)
        
        var advance = CGSize.zero
        CTFontGetAdvancesForGlyphs(font, .default, &glyph, &advance, 1)
        
        let cellWidth = ceil(advance.width)
        let cellHeight = ceil(ascent + descent + leading)
        
        return FontMetrics(
            cellWidth: cellWidth,
            cellHeight: cellHeight,
            ascent: ascent,
            descent: descent,
            lineGap: leading
        )
    }
}

/// Internal structure for glyph metrics
struct GlyphMetrics {
    let width: UInt32
    let height: UInt32
    let offsetX: Int32
    let offsetY: Int32
    let advanceX: Float
}