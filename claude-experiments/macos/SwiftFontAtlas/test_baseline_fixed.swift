#!/usr/bin/env swift

import Foundation
import CoreText
import CoreGraphics

// Test baseline alignment with different character types
let fontName = "Menlo"
let fontSize: CGFloat = 24

let font = CTFont(fontName as CFString, size: fontSize)
print("Testing baseline alignment with \(fontName) \(fontSize)pt")

// Test different character types that should sit on the same baseline
let testChars: [Character] = ["H", "g", "j", "p", "y", "A", "a", "e"]

for char in testChars {
    let utf16 = Array(char.utf16)
    var glyphs = [CGGlyph](repeating: 0, count: utf16.count)
    let success = CTFontGetGlyphsForCharacters(font, utf16, &glyphs, utf16.count)
    
    if success, let glyph = glyphs.first, glyph != 0 {
        var mutableGlyph = glyph
        var glyphRect = CGRect.zero
        CTFontGetBoundingRectsForGlyphs(font, .default, &mutableGlyph, &glyphRect, 1)
        
        let padding: CGFloat = 3
        let offsetY_old = Int32(round(glyphRect.minY - padding))
        let offsetY_new = Int32(round(glyphRect.maxY + padding))
        
        print("'\(char)': rect=\(glyphRect) -> old_offsetY=\(offsetY_old), new_offsetY=\(offsetY_new)")
    }
}

print("\nWith new calculation:")
print("- All characters should have similar offsetY values for consistent baseline alignment")
print("- offsetY represents distance from baseline to TOP of glyph bitmap")