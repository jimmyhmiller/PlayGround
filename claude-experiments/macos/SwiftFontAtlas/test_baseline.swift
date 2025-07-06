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
        let offsetY = Int32(round(glyphRect.minY - padding))
        
        print("'\(char)': rect=\(glyphRect) -> offsetY=\(offsetY)")
    }
}

print("\nBaseline should be at Y=0. Characters with descenders (g,j,p,y) should have negative offsetY.")
print("Characters without descenders (H,A,a,e) should have offsetY near 0 or slightly negative.")