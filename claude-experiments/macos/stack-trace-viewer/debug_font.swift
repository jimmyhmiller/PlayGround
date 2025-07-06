#!/usr/bin/env swift

import Cocoa
import Foundation

func createTestFontAtlas() {
    let fontSize: CGFloat = 24
    let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .medium)
    
    // Test the actual layout - make sure we have all characters
    let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:()[]<>"
    print("Creating test font atlas with \(characters.count) characters")
    print("Character order: \(characters)")
    
    // Fixed atlas size - same as main app
    let atlasWidth = 512
    let atlasHeight = 384
    let charWidth = 32
    let charHeight = 48
    let charsPerRow = 16
    
    // Create CGContext for proper text rendering
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    guard let context = CGContext(
        data: nil,
        width: atlasWidth,
        height: atlasHeight,
        bitsPerComponent: 8,
        bytesPerRow: atlasWidth * 4,
        space: colorSpace,
        bitmapInfo: bitmapInfo
    ) else { 
        print("Failed to create CGContext")
        return 
    }
    
    // Clear to gray background
    context.setFillColor(CGColor(red: 0.3, green: 0.3, blue: 0.3, alpha: 1.0))
    context.fill(CGRect(x: 0, y: 0, width: atlasWidth, height: atlasHeight))
    
    // Draw all characters to see the layout
    for (index, char) in characters.enumerated() {
        let col = index % charsPerRow
        let row = index / charsPerRow
        
        let startX = col * charWidth
        let startY = row * charHeight
        
        let flippedStartY = (atlasHeight - (row + 1) * charHeight)
        print("Character '\(char)' (index \(index)) -> col: \(col), row: \(row), pos: (\(startX), \(startY)) -> flipped Y: \(flippedStartY)")
        print("  UV coords: (\(Float(startX)/Float(atlasWidth)), \(Float(flippedStartY)/Float(atlasHeight)))")
        
        // Draw border around each character slot
        context.setStrokeColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0))
        context.setLineWidth(1.0)
        let borderRect = CGRect(x: CGFloat(startX), y: CGFloat(startY), width: CGFloat(charWidth), height: CGFloat(charHeight))
        context.stroke(borderRect)
        
        // Highlight first character (A) with red border
        if index == 0 {
            context.setStrokeColor(CGColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0))
            context.setLineWidth(3.0)
            context.stroke(borderRect)
        }
        
        // Draw the character
        let string = String(char)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: NSColor.white
        ]
        let attributedString = NSAttributedString(string: string, attributes: attributes)
        
        NSGraphicsContext.saveGraphicsState()
        let nsContext = NSGraphicsContext(cgContext: context, flipped: false)
        NSGraphicsContext.current = nsContext
        
        let textRect = CGRect(x: CGFloat(startX + 4), y: CGFloat(startY + 4), width: CGFloat(charWidth - 8), height: CGFloat(charHeight - 8))
        attributedString.draw(in: textRect)
        
        NSGraphicsContext.restoreGraphicsState()
    }
    
    // Save atlas as PNG file for debugging
    if let cgImage = context.makeImage() {
        let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
        if let pngData = bitmapRep.representation(using: .png, properties: [:]) {
            let fileURL = URL(fileURLWithPath: "/tmp/test_font_atlas.png")
            try? pngData.write(to: fileURL)
            print("Test font atlas saved to: \(fileURL.path)")
            print("Check if 'A' appears in top-left corner")
        }
    }
}

createTestFontAtlas()