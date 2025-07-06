#!/usr/bin/env swift

import Foundation
import CoreText
import CoreGraphics

// Copy the types and classes here to test directly
import os.lock

enum PixelFormat: UInt8 {
    case grayscale = 0
    case bgr = 1
    case bgra = 2
    
    var bytesPerPixel: Int {
        switch self {
        case .grayscale: return 1
        case .bgr: return 3
        case .bgra: return 4
        }
    }
}

struct AtlasRegion: Equatable {
    let x: UInt32
    let y: UInt32
    let width: UInt32
    let height: UInt32
    
    init(x: UInt32, y: UInt32, width: UInt32, height: UInt32) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

struct RenderedGlyph: Equatable {
    let width: UInt32
    let height: UInt32
    let offsetX: Int32
    let offsetY: Int32
    let atlasX: UInt32
    let atlasY: UInt32
    let advanceX: Float
    
    init(
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

struct GlyphKey: Hashable {
    let codepoint: UInt32
    let fontSize: Float
    let fontName: String
    
    init(codepoint: UInt32, fontSize: Float, fontName: String) {
        self.codepoint = codepoint
        self.fontSize = fontSize
        self.fontName = fontName
    }
    
    init(character: Character, fontSize: Float, fontName: String) {
        self.codepoint = character.unicodeScalars.first?.value ?? 0
        self.fontSize = fontSize
        self.fontName = fontName
    }
}

struct FontMetrics {
    let cellWidth: CGFloat
    let cellHeight: CGFloat
    let ascent: CGFloat
    let descent: CGFloat
    let lineGap: CGFloat
    
    init(
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

enum FontAtlasError: Error {
    case atlasFull
    case contextCreationFailed
    case fontCreationFailed
    case glyphNotFound
    case invalidAtlasSize
}

print("Testing font atlas functionality...")

// Test basic font creation
do {
    let fontName = "SF Mono"
    let fontSize: CGFloat = 14.0
    
    let font = CTFont(fontName as CFString, size: fontSize)
    let actualFontName = CTFontCopyPostScriptName(font) as String? ?? "Unknown"
    print("Created font: \(actualFontName) at \(fontSize)pt")
    
    // Test glyph rendering
    let testChar: Character = "A"
    let utf16 = Array(testChar.utf16)
    var glyphs = [CGGlyph](repeating: 0, count: utf16.count)
    let success = CTFontGetGlyphsForCharacters(font, utf16, &glyphs, utf16.count)
    
    if success, let glyph = glyphs.first, glyph != 0 {
        print("✅ Successfully got glyph for '\(testChar)': \(glyph)")
        
        // Test glyph metrics
        var mutableGlyph = glyph
        var glyphRect = CGRect.zero
        CTFontGetBoundingRectsForGlyphs(font, .default, &mutableGlyph, &glyphRect, 1)
        
        var advance = CGSize.zero
        CTFontGetAdvancesForGlyphs(font, .default, &mutableGlyph, &advance, 1)
        
        print("   Glyph rect: \(glyphRect)")
        print("   Advance: \(advance)")
        
        // Test bitmap rendering
        let padding: CGFloat = 2
        let width = Int(ceil(glyphRect.width + padding * 2))
        let height = Int(ceil(glyphRect.height + padding * 2))
        
        if width > 0 && height > 0 {
            let bytesPerRow = width
            var pixelData = Data(count: width * height)
            
            if let context = CGContext(
                data: pixelData.withUnsafeMutableBytes { $0.baseAddress },
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) {
                context.setAllowsAntialiasing(true)
                context.setShouldAntialias(true)
                context.setFillColor(CGColor(gray: 0, alpha: 1))
                context.fill(CGRect(x: 0, y: 0, width: width, height: height))
                context.setFillColor(CGColor(gray: 1, alpha: 1))
                
                let drawX = padding - glyphRect.minX
                let drawY = padding - glyphRect.minY
                var position = CGPoint(x: drawX, y: drawY)
                CTFontDrawGlyphs(font, &mutableGlyph, &position, 1, context)
                
                // Check if we got any non-zero pixels
                var nonZeroPixels = 0
                for byte in pixelData {
                    if byte != 0 { nonZeroPixels += 1 }
                }
                
                print("   ✅ Rendered glyph to \(width)x\(height) bitmap with \(nonZeroPixels) non-zero pixels")
            } else {
                print("   ❌ Failed to create bitmap context")
            }
        } else {
            print("   ❌ Invalid glyph dimensions: \(width)x\(height)")
        }
    } else {
        print("❌ Failed to get glyph for '\(testChar)'")
    }
    
} catch {
    print("❌ Error: \(error)")
}