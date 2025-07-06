import Foundation
import SwiftFontAtlas
import CoreGraphics
import AppKit

print("🎨 Rendering 'through' test image...")

do {
    // Create font atlas
    let manager = try FontAtlasManager(
        fontName: "Menlo",
        fontSize: 24,
        atlasSize: 512
    )
    
    // Pre-render the text
    let testText = "through"
    _ = manager.prerenderString(testText)
    
    // Create image context
    let imageWidth = 400
    let imageHeight = 100
    
    guard let context = CGContext(
        data: nil,
        width: imageWidth,
        height: imageHeight,
        bitsPerComponent: 8,
        bytesPerRow: imageWidth * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        print("❌ Failed to create context")
        exit(1)
    }
    
    // Clear to white background
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight))
    
    // Set text color to black
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    
    // Position text in center
    let startX: CGFloat = 50
    let baselineY: CGFloat = 50  // From bottom of image
    
    var currentX: CGFloat = startX
    
    print("📝 Rendering characters:")
    for character in testText {
        if let glyph = manager.renderCharacter(character) {
            print("  '\(character)': offset(\(glyph.offsetX), \(glyph.offsetY)) size(\(glyph.width)x\(glyph.height))")
            
            if glyph.width > 0 && glyph.height > 0 {
                // Get glyph bitmap from atlas
                manager.withAtlas { atlas in
                    // Create a small context for this glyph
                    guard let glyphContext = CGContext(
                        data: nil,
                        width: Int(glyph.width),
                        height: Int(glyph.height),
                        bitsPerComponent: 8,
                        bytesPerRow: Int(glyph.width),
                        space: CGColorSpaceCreateDeviceGray(),
                        bitmapInfo: CGImageAlphaInfo.none.rawValue
                    ) else { return }
                    
                    // Copy glyph data from atlas
                    let glyphData = glyphContext.data!.assumingMemoryBound(to: UInt8.self)
                    for y in 0..<Int(glyph.height) {
                        for x in 0..<Int(glyph.width) {
                            let atlasOffset = Int((glyph.atlasY + UInt32(y)) * atlas.size + (glyph.atlasX + UInt32(x)))
                            let glyphOffset = y * Int(glyph.width) + x
                            if atlasOffset < atlas.data.count {
                                glyphData[glyphOffset] = atlas.data[atlasOffset]
                            }
                        }
                    }
                    
                    // Create CGImage from glyph data
                    if let glyphImage = glyphContext.makeImage() {
                        // Calculate position using the corrected baseline formula
                        let glyphX = currentX + CGFloat(glyph.offsetX)
                        let glyphY = baselineY + CGFloat(glyph.offsetY)  // Fixed: use + instead of -
                        
                        print("    Drawing at (\(glyphX), \(glyphY))")
                        
                        // Draw glyph (note: CoreGraphics has bottom-left origin)
                        context.draw(
                            glyphImage,
                            in: CGRect(
                                x: glyphX,
                                y: glyphY,
                                width: CGFloat(glyph.width),
                                height: CGFloat(glyph.height)
                            )
                        )
                    }
                }
            }
            
            currentX += CGFloat(glyph.advanceX)
        }
    }
    
    // Create final image and save
    if let finalImage = context.makeImage() {
        let nsImage = NSImage(cgImage: finalImage, size: NSSize(width: imageWidth, height: imageHeight))
        
        // Save to desktop
        let desktopURL = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
        let imageURL = desktopURL.appendingPathComponent("through_test.png")
        
        if let tiffData = nsImage.tiffRepresentation,
           let imageRep = NSBitmapImageRep(data: tiffData),
           let pngData = imageRep.representation(using: .png, properties: [:]) {
            try pngData.write(to: imageURL)
            print("✅ Image saved to: \(imageURL.path)")
            print("📄 Check your desktop for 'through_test.png'")
        } else {
            print("❌ Failed to save image")
        }
    } else {
        print("❌ Failed to create final image")
    }
    
} catch {
    print("❌ Error: \(error)")
    exit(1)
}