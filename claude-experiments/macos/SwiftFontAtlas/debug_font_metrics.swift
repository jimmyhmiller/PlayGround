#!/usr/bin/env swift

import Foundation
import CoreText
import CoreGraphics

// Debug font metrics to understand baseline issues
let fontName = "Menlo"
let fontSize: CGFloat = 24

let font = CTFont(fontName as CFString, size: fontSize)
let ascent = CTFontGetAscent(font)
let descent = CTFontGetDescent(font)
let leading = CTFontGetLeading(font)

print("Font: \(fontName) \(fontSize)pt")
print("Ascent: \(ascent)")
print("Descent: \(descent)")
print("Leading: \(leading)")
print("Line Height: \(ascent + descent + leading)")

// Test the word "through" specifically
let testWord = "through"
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

print("\nProblem analysis:")
print("- Characters with descenders (g, p) have more negative minY")
print("- This gives them more negative offsetY")
print("- Using currentY - offsetY makes them positioned HIGHER")
print("- This is the opposite of what we want for baseline alignment")