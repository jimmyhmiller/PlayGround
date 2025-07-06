#!/usr/bin/env swift

import Foundation
import CoreText
import CoreGraphics
import AppKit

// Quick test to verify the font atlas rendering works correctly

let fontName = "Menlo"
let fontSize: CGFloat = 24

// Test font creation
let font = CTFont(fontName as CFString, size: fontSize)
let actualFontName = CTFontCopyPostScriptName(font) as String? ?? "Unknown"
print("Testing font: \(actualFontName) at \(fontSize)pt")

// Test character rendering
let testChar: Character = "A"
let utf16 = Array(testChar.utf16)
var glyphs = [CGGlyph](repeating: 0, count: utf16.count)
let success = CTFontGetGlyphsForCharacters(font, utf16, &glyphs, utf16.count)

if success, let glyph = glyphs.first, glyph != 0 {
    print("✅ Glyph found for '\(testChar)': \(glyph)")
    
    // Get glyph metrics
    var mutableGlyph = glyph
    var glyphRect = CGRect.zero
    CTFontGetBoundingRectsForGlyphs(font, .default, &mutableGlyph, &glyphRect, 1)
    
    var advance = CGSize.zero
    CTFontGetAdvancesForGlyphs(font, .default, &mutableGlyph, &advance, 1)
    
    print("Glyph rect: \(glyphRect)")
    print("Advance: \(advance)")
    
    // Test rendering to bitmap
    let padding: CGFloat = 3
    let width = Int(ceil(max(glyphRect.width + padding * 2, 1)))
    let height = Int(ceil(max(glyphRect.height + padding * 2, 1)))
    
    print("Bitmap size: \(width)x\(height)")
    
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
            // Configure context
            context.setAllowsAntialiasing(true)
            context.setShouldAntialias(true)
            context.setAllowsFontSmoothing(true)
            context.setShouldSmoothFonts(true)
            
            // Clear to black
            context.setFillColor(CGColor(gray: 0, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            
            // Set text to white
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            
            // Draw glyph
            let drawX = padding - glyphRect.minX
            let drawY = padding - glyphRect.minY
            var position = CGPoint(x: drawX, y: drawY)
            CTFontDrawGlyphs(font, &mutableGlyph, &position, 1, context)
            
            // Count non-zero pixels
            var nonZeroPixels = 0
            for byte in pixelData {
                if byte != 0 { nonZeroPixels += 1 }
            }
            
            print("✅ Rendered with \(nonZeroPixels) non-zero pixels")
            
            // Create a test image to verify rendering
            if let cgImage = context.makeImage() {
                let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
                
                // Try to save to desktop for verification
                if let tiffData = nsImage.tiffRepresentation,
                   let bitmapRep = NSBitmapImageRep(data: tiffData),
                   let pngData = bitmapRep.representation(using: .png, properties: [:]) {
                    
                    let desktopURL = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
                    let fileURL = desktopURL.appendingPathComponent("test_glyph_\(testChar).png")
                    
                    do {
                        try pngData.write(to: fileURL)
                        print("✅ Saved test glyph image to: \(fileURL.path)")
                    } catch {
                        print("❌ Failed to save image: \(error)")
                    }
                }
            }
            
        } else {
            print("❌ Failed to create bitmap context")
        }
    }
} else {
    print("❌ Failed to get glyph for '\(testChar)'")
}