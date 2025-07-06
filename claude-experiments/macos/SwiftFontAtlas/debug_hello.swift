#!/usr/bin/env swift

import Foundation
import CoreText
import CoreGraphics

// Debug "Hello" to see why e and o are misaligned
let fontName = "Menlo"
let fontSize: CGFloat = 24

let font = CTFont(fontName as CFString, size: fontSize)
print("Debugging 'Hello' alignment with \(fontName) \(fontSize)pt")

let testWord = "Hello"
print("\nAnalyzing '\(testWord)':")

for (i, char) in testWord.enumerated() {
    let utf16 = Array(char.utf16)
    var glyphs = [CGGlyph](repeating: 0, count: utf16.count)
    let success = CTFontGetGlyphsForCharacters(font, utf16, &glyphs, utf16.count)
    
    if success, let glyph = glyphs.first, glyph != 0 {
        var mutableGlyph = glyph
        var glyphRect = CGRect.zero
        CTFontGetBoundingRectsForGlyphs(font, .default, &mutableGlyph, &glyphRect, 1)
        
        let padding: CGFloat = 3
        let offsetY = Int32(round(glyphRect.minY - padding))
        
        print("  [\(i)] '\(char)': minY=\(glyphRect.minY), maxY=\(glyphRect.maxY), offsetY=\(offsetY)")
        
        // Calculate position with currentY - offsetY approach
        let currentY: Float = 50.0
        let finalY = currentY - Float(offsetY)
        print("      With currentY=\(currentY): finalY=\(finalY)")
    }
}

print("\nPattern analysis:")
print("Characters that appear too high likely have different offsetY values")
print("Need to normalize positioning to ensure consistent baseline")