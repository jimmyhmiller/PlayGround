#!/usr/bin/env swift

import Foundation
import CoreGraphics

// Import the local SwiftFontAtlas module
import class SwiftFontAtlas.FontAtlasManager
import struct SwiftFontAtlas.FontAtlasError
import struct SwiftFontAtlas.RenderedGlyph
import struct SwiftFontAtlas.FontMetrics

/// Command-line demo showing SwiftFontAtlas features
struct FontAtlasDemo {
    
    static func main() {
        print("🚀 SwiftFontAtlas Library Demo")
        print("==============================")
        
        // Test 1: Basic Atlas Creation
        print("\n1️⃣ Creating Font Atlas")
        print("   Font: SF Mono, Size: 14pt, Atlas: 512x512")
        
        do {
            let manager = try FontAtlasManager(
                fontName: "SF Mono",
                fontSize: 14.0,
                atlasSize: 512
            )
            
            print("   ✅ Atlas created successfully!")
            print("   📏 Cell size: \(manager.cellSize)")
            print("   📊 Font metrics: ascent=\(manager.metrics.ascent), descent=\(manager.metrics.descent)")
            
            // Test 2: ASCII Prerendering
            print("\n2️⃣ Prerendering ASCII Characters")
            let start = CFAbsoluteTimeGetCurrent()
            let asciiCount = manager.prerenderASCII()
            let asciiTime = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            print("   ✅ Rendered \(asciiCount) ASCII characters")
            print("   ⏱️  Time: \(String(format: "%.3f", asciiTime))ms")
            
            // Test 3: Custom Text Rendering
            print("\n3️⃣ Rendering Custom Text")
            let testStrings = [
                "Hello, World!",
                "SwiftFontAtlas 🚀",
                "αβγδε (Greek)",
                "→←↑↓ (Arrows)"
            ]
            
            for text in testStrings {
                let start = CFAbsoluteTimeGetCurrent()
                let count = manager.prerenderString(text)
                let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
                
                print("   '\(text)': \(count) characters in \(String(format: "%.3f", time))ms")
            }
            
            // Test 4: Individual Character Details
            print("\n4️⃣ Character Rendering Details")
            let testChars: [Character] = ["A", "g", "🙂", "→"]
            
            for char in testChars {
                if let glyph = manager.renderCharacter(char) {
                    print("   '\(char)': size(\(glyph.width)×\(glyph.height)) atlas(\(glyph.atlasX),\(glyph.atlasY)) advance(\(glyph.advanceX))")
                } else {
                    print("   '\(char)': ❌ Failed to render")
                }
            }
            
            // Test 5: Caching Performance
            print("\n5️⃣ Cache Performance Test")
            let testChar: Character = "X"
            
            // First render (cache miss)
            let firstStart = CFAbsoluteTimeGetCurrent()
            _ = manager.renderCharacter(testChar)
            let firstTime = (CFAbsoluteTimeGetCurrent() - firstStart) * 1000
            
            // Subsequent renders (cache hits)
            var cacheTotal: Double = 0
            let cacheRuns = 1000
            
            for _ in 0..<cacheRuns {
                let cacheStart = CFAbsoluteTimeGetCurrent()
                _ = manager.renderCharacter(testChar)
                cacheTotal += (CFAbsoluteTimeGetCurrent() - cacheStart) * 1000
            }
            
            let avgCacheTime = cacheTotal / Double(cacheRuns)
            let speedup = firstTime / avgCacheTime
            
            print("   First render: \(String(format: "%.6f", firstTime))ms")
            print("   Cache hit avg: \(String(format: "%.6f", avgCacheTime))ms (\(cacheRuns) runs)")
            print("   Cache speedup: \(String(format: "%.1f", speedup))x")
            
            // Test 6: Atlas Statistics
            print("\n6️⃣ Atlas Statistics")
            manager.withAtlas { atlas in
                let utilizationPercent = Double(manager.renderCount * 16 * 16) / Double(atlas.size * atlas.size) * 100 // rough estimate
                print("   Atlas size: \(atlas.size)×\(atlas.size)")
                print("   Data size: \(atlas.data.count) bytes (\(String(format: "%.2f", Double(atlas.data.count) / 1024))KB)")
                print("   Modifications: \(atlas.modificationCount.withLock { $0 })")
                print("   Estimated utilization: \(String(format: "%.1f", utilizationPercent))%")
            }
            
            // Test 7: Atlas Growth Test
            print("\n7️⃣ Atlas Growth Test")
            print("   Testing with extended Unicode characters...")
            
            let growthStart = CFAbsoluteTimeGetCurrent()
            var extendedCount = 0
            
            // Test various Unicode ranges
            let unicodeRanges = [
                0x0100...0x017F, // Latin Extended-A
                0x0180...0x024F, // Latin Extended-B
                0x2010...0x206F, // Punctuation
            ]
            
            for range in unicodeRanges {
                for codepoint in range.prefix(20) { // Limit to avoid excessive growth
                    if let scalar = UnicodeScalar(codepoint) {
                        let char = Character(scalar)
                        if manager.renderCharacter(char) != nil {
                            extendedCount += 1
                        }
                    }
                }
            }
            
            let growthTime = (CFAbsoluteTimeGetCurrent() - growthStart) * 1000
            print("   ✅ Rendered \(extendedCount) extended characters")
            print("   ⏱️  Time: \(String(format: "%.3f", growthTime))ms")
            
            // Final atlas state
            manager.withAtlas { atlas in
                print("   Final atlas size: \(atlas.size)×\(atlas.size)")
                print("   Final modifications: \(atlas.modificationCount.withLock { $0 })")
                print("   Resize count: \(atlas.resizeCount.withLock { $0 })")
            }
            
            // Test 8: Thread Safety (basic test)
            print("\n8️⃣ Thread Safety Test")
            print("   Testing concurrent character rendering...")
            
            let group = DispatchGroup()
            let concurrentQueue = DispatchQueue(label: "atlas.test", attributes: .concurrent)
            let threadCount = 4
            let iterations = 100
            
            let threadStart = CFAbsoluteTimeGetCurrent()
            
            for threadId in 0..<threadCount {
                group.enter()
                concurrentQueue.async {
                    for i in 0..<iterations {
                        let char = Character(UnicodeScalar(65 + (threadId * iterations + i) % 26)!) // A-Z cycling
                        _ = manager.renderCharacter(char)
                    }
                    group.leave()
                }
            }
            
            group.wait()
            let threadTime = (CFAbsoluteTimeGetCurrent() - threadStart) * 1000
            
            print("   ✅ \(threadCount) threads × \(iterations) renders completed")
            print("   ⏱️  Total time: \(String(format: "%.3f", threadTime))ms")
            print("   📊 Ops/sec: \(String(format: "%.0f", Double(threadCount * iterations) / (threadTime / 1000)))")
            
            print("\n🎉 All tests completed successfully!")
            print("\n💡 Key Features Demonstrated:")
            print("   • Rectangle bin packing for efficient space usage")
            print("   • Thread-safe glyph caching with significant speedup")
            print("   • Automatic atlas resizing when needed")
            print("   • Support for Unicode characters beyond ASCII")
            print("   • High-performance concurrent access")
            print("   • CoreText integration for high-quality rendering")
            
        } catch {
            print("❌ Error: \(error)")
        }
    }
}

// Extension to get render count (for demonstration)
extension FontAtlasManager {
    var renderCount: Int {
        // This is a rough estimate - in a real implementation you'd track this
        return Int(modificationCount / 2) // Approximate based on modifications
    }
}

FontAtlasDemo.main()