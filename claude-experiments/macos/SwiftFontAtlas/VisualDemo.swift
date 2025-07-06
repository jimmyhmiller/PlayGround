#!/usr/bin/env swift

import Foundation
import CoreGraphics
import AppKit

// Import the local SwiftFontAtlas module types
typealias FontAtlasManager = SwiftFontAtlas.FontAtlasManager
typealias FontAtlas = SwiftFontAtlas.FontAtlas

print("🎨 Visual Font Atlas Demo")
print("=========================")
print("This demo creates font atlases and saves them as images you can view.")

func createAtlasImage(from atlas: FontAtlas, filename: String) -> Bool {
    let size = Int(atlas.size)
    
    // Create CGContext
    guard let context = CGContext(
        data: nil,
        width: size,
        height: size,
        bitsPerComponent: 8,
        bytesPerRow: size,
        space: CGColorSpaceCreateDeviceGray(),
        bitmapInfo: CGImageAlphaInfo.none.rawValue
    ) else {
        print("❌ Failed to create graphics context")
        return false
    }
    
    // Copy atlas data
    atlas.data.withUnsafeBytes { bytes in
        if let data = context.data {
            data.copyMemory(from: bytes.baseAddress!, byteCount: atlas.data.count)
        }
    }
    
    guard let cgImage = context.makeImage() else {
        print("❌ Failed to create CGImage")
        return false
    }
    
    // Create NSImage and save as PNG
    let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: size, height: size))
    
    guard let tiffData = nsImage.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiffData),
          let pngData = bitmap.representation(using: NSBitmapImageRep.FileType.png, properties: [:]) else {
        print("❌ Failed to create PNG data")
        return false
    }
    
    let fileURL = URL(fileURLWithPath: filename)
    
    do {
        try pngData.write(to: fileURL)
        print("✅ Saved atlas image: \(filename)")
        return true
    } catch {
        print("❌ Failed to save image: \(error)")
        return false
    }
}

func runDemo() {
    do {
        print("\n1️⃣ Creating Font Atlas (SF Mono, 16pt, 512×512)")
        let manager = try FontAtlasManager(
            fontName: "SF Mono",
            fontSize: 16.0,
            atlasSize: 512
        )
        
        // Save empty atlas
        manager.withAtlas { atlas in
            _ = createAtlasImage(from: atlas, filename: "atlas_01_empty.png")
        }
        
        print("\n2️⃣ Rendering ASCII Characters")
        let asciiCount = manager.prerenderASCII()
        print("   Rendered \(asciiCount) ASCII characters")
        
        // Save after ASCII
        manager.withAtlas { atlas in
            _ = createAtlasImage(from: atlas, filename: "atlas_02_ascii.png")
        }
        
        print("\n3️⃣ Adding Custom Text")
        let customTexts = [
            "Hello, World!",
            "SwiftFontAtlas 🚀",
            "The quick brown fox jumps over the lazy dog",
            "áéíóúàèìòùâêîôûäëïöüç",
            "αβγδεζηθικλμνξοπρστυφχψω",
            "→←↑↓⇒⇐⇑⇓⇔⇕",
            "©®™℠°±×÷"
        ]
        
        for text in customTexts {
            let count = manager.prerenderString(text)
            print("   '\(text)': \(count) new characters")
        }
        
        // Save after custom text
        manager.withAtlas { atlas in
            _ = createAtlasImage(from: atlas, filename: "atlas_03_unicode.png")
        }
        
        print("\n4️⃣ Adding Emoji and Symbols")
        let emojiText = "🚀🎯📱💻⚡🔥🎨🌟💯🚀"
        let emojiCount = manager.prerenderString(emojiText)
        print("   Added \(emojiCount) emoji/symbols")
        
        // Save final atlas
        manager.withAtlas { atlas in
            _ = createAtlasImage(from: atlas, filename: "atlas_04_final.png")
            
            // Print final statistics
            print("\n📊 Final Atlas Statistics:")
            print("   Size: \(atlas.size)×\(atlas.size) pixels")
            print("   Memory: \(atlas.data.count) bytes (\(Double(atlas.data.count) / 1024)KB)")
            print("   Modifications: \(atlas.modificationCount.withLock { $0 })")
            
            // Calculate utilization
            var nonZeroPixels = 0
            for byte in atlas.data {
                if byte != 0 { nonZeroPixels += 1 }
            }
            let utilization = Double(nonZeroPixels) / Double(atlas.data.count) * 100
            print("   Pixel utilization: \(String(format: "%.2f", utilization))%")
        }
        
        print("\n🎉 Demo Complete!")
        print("\nGenerated Images:")
        print("   📄 atlas_01_empty.png - Empty atlas")
        print("   📄 atlas_02_ascii.png - After ASCII rendering")
        print("   📄 atlas_03_unicode.png - After Unicode text")
        print("   📄 atlas_04_final.png - Final atlas with emoji")
        print("\nOpen these PNG files to see the font atlas in action!")
        print("White pixels = rendered glyphs, Black = empty space")
        
        print("\n💡 What to Look For:")
        print("   • Rectangle packing efficiency")
        print("   • Glyph shapes and spacing")
        print("   • How different characters use atlas space")
        print("   • Progression from empty to full atlas")
        
        // Create a zoomed version for better viewing
        print("\n🔍 Creating zoomed version for easier viewing...")
        manager.withAtlas { atlas in
            createZoomedAtlasImage(from: atlas, filename: "atlas_zoomed_8x.png", zoomFactor: 8)
        }
        
    } catch {
        print("❌ Error: \(error)")
    }
}

func createZoomedAtlasImage(from atlas: FontAtlas, filename: String, zoomFactor: Int) -> Bool {
    let originalSize = Int(atlas.size)
    let zoomedSize = originalSize * zoomFactor
    
    // Create larger context
    guard let context = CGContext(
        data: nil,
        width: zoomedSize,
        height: zoomedSize,
        bitsPerComponent: 8,
        bytesPerRow: zoomedSize,
        space: CGColorSpaceCreateDeviceGray(),
        bitmapInfo: CGImageAlphaInfo.none.rawValue
    ) else {
        return false
    }
    
    // Scale up each pixel
    for y in 0..<originalSize {
        for x in 0..<originalSize {
            let originalIndex = y * originalSize + x
            let pixelValue = atlas.data[originalIndex]
            
            // Draw zoomFactor×zoomFactor square for each pixel
            for zy in 0..<zoomFactor {
                for zx in 0..<zoomFactor {
                    let zoomedIndex = (y * zoomFactor + zy) * zoomedSize + (x * zoomFactor + zx)
                    if let data = context.data {
                        data.storeBytes(of: pixelValue, toByteOffset: zoomedIndex, as: UInt8.self)
                    }
                }
            }
        }
    }
    
    guard let cgImage = context.makeImage() else { return false }
    
    let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: zoomedSize, height: zoomedSize))
    
    guard let tiffData = nsImage.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiffData),
          let pngData = bitmap.representation(using: NSBitmapImageRep.FileType.png, properties: [:]) else {
        return false
    }
    
    let fileURL = URL(fileURLWithPath: filename)
    
    do {
        try pngData.write(to: fileURL)
        print("✅ Saved zoomed atlas: \(filename)")
        return true
    } catch {
        return false
    }
}

runDemo()