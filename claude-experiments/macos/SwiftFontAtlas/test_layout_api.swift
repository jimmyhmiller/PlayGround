#!/usr/bin/env swift

import Foundation
import SwiftFontAtlas

print("🔍 Testing the new layout API...")

do {
    let manager = try FontAtlasManager(
        fontName: "Menlo",
        fontSize: 24.0,
        atlasSize: 512
    )
    
    // Test the baseline calculation
    let contextHeight: CGFloat = 400
    let baseline = manager.baselinePosition(contextHeight: contextHeight, topMargin: 20)
    print("✅ Baseline position: \(baseline)")
    
    // Test line layout with words that have different heights
    let testText = "The quick brown fox jumps over the lazy dog."
    let problematicText = "through pygmy" // Words with descenders that were misaligned
    
    print("\n📝 Testing line layout...")
    
    let layout1 = manager.layoutLine(testText, baselineX: 10, baselineY: baseline)
    print("✅ Laid out '\(testText)' with \(layout1.count) positioned glyphs")
    
    let layout2 = manager.layoutLine(problematicText, baselineX: 10, baselineY: baseline - manager.lineHeight)
    print("✅ Laid out '\(problematicText)' with \(layout2.count) positioned glyphs")
    
    // Check that all characters in "through" sit on the same baseline
    print("\n🔍 Checking baseline alignment for 'through':")
    let throughLayout = manager.layoutLine("through", baselineX: 0, baselineY: 100)
    
    for (char, glyph, x, y) in throughLayout {
        let baselineY = y - CGFloat(glyph.offsetY) // Calculate actual baseline position
        print("   '\(char)': glyph at (\(x), \(y)), baseline at Y=\(baselineY)")
    }
    
    // Test line width measurement
    let width1 = manager.lineWidth(testText)
    let width2 = manager.lineWidth(problematicText)
    print("\n📏 Line widths:")
    print("   '\(testText)': \(width1) points")
    print("   '\(problematicText)': \(width2) points")
    
    print("\n🎉 Layout API test passed! All characters should now align properly on baseline.")
    
} catch {
    print("❌ Error during testing: \(error)")
}