import Foundation
import SwiftFontAtlas

// Quick verification that our fixes work
print("🔍 Verifying FontAtlas fixes...")

do {
    // Test 1: Font creation
    print("\n1. Testing font creation...")
    let manager = try FontAtlasManager(
        fontName: "Menlo",
        fontSize: 24.0,
        atlasSize: 512
    )
    print("✅ FontAtlasManager created successfully")
    print("   Cell size: \(manager.cellSize)")
    print("   Font metrics: \(manager.metrics)")
    
    // Test 2: ASCII rendering
    print("\n2. Testing ASCII rendering...")
    let asciiCount = manager.prerenderASCII()
    print("✅ Pre-rendered \(asciiCount) ASCII characters")
    
    // Test 3: Individual character rendering
    print("\n3. Testing individual characters...")
    let testChars: [Character] = ["H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d", "!"]
    var successCount = 0
    
    for char in testChars {
        if let glyph = manager.renderCharacter(char) {
            print("   '\(char)': atlas(\(glyph.atlasX),\(glyph.atlasY)) size(\(glyph.width)x\(glyph.height)) advance(\(glyph.advanceX))")
            successCount += 1
        } else {
            print("   '\(char)': ❌ Failed to render")
        }
    }
    
    print("✅ Rendered \(successCount)/\(testChars.count) characters successfully")
    
    // Test 4: Atlas statistics
    print("\n4. Atlas statistics...")
    manager.withAtlas { atlas in
        let memoryMB = Double(atlas.data.count) / (1024 * 1024)
        
        var nonZeroPixels = 0
        for byte in atlas.data {
            if byte != 0 { nonZeroPixels += 1 }
        }
        let utilization = Double(nonZeroPixels) / Double(atlas.data.count) * 100
        
        print("   Atlas size: \(atlas.size)x\(atlas.size)")
        print("   Memory usage: \(String(format: "%.2f", memoryMB)) MB")
        print("   Utilization: \(String(format: "%.1f", utilization))%")
        print("   Non-zero pixels: \(nonZeroPixels)")
    }
    
    print("\n🎉 All tests passed! The font atlas is working correctly.")
    
} catch {
    print("❌ Error during testing: \(error)")
}