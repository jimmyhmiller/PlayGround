import SwiftFontAtlas
import Foundation

/// Example demonstrating basic usage of SwiftFontAtlas
struct FontAtlasExample {
    
    static func run() {
        print("SwiftFontAtlas Example")
        print("======================")
        
        do {
            // Create a font atlas manager
            print("\n1. Creating font atlas manager...")
            let manager = try FontAtlasManager(
                fontName: "SF Mono",
                fontSize: 14.0,
                atlasSize: 512
            )
            
            print("   Cell size: \(manager.cellSize)")
            print("   Font metrics: \(manager.metrics)")
            
            // Pre-render ASCII characters
            print("\n2. Pre-rendering ASCII characters...")
            let asciiCount = manager.prerenderASCII()
            print("   Rendered \(asciiCount) ASCII characters")
            
            // Render some specific characters
            print("\n3. Rendering individual characters...")
            let testChars: [Character] = ["H", "e", "l", "o", "!", "🙂"]
            
            for char in testChars {
                if let glyph = manager.renderCharacter(char) {
                    print("   '\(char)': atlas(\(glyph.atlasX), \(glyph.atlasY)) size(\(glyph.width)x\(glyph.height)) advance(\(glyph.advanceX))")
                } else {
                    print("   '\(char)': Failed to render")
                }
            }
            
            // Show atlas statistics
            print("\n4. Atlas statistics...")
            manager.withAtlas { atlas in
                print("   Atlas size: \(atlas.size)x\(atlas.size)")
                print("   Atlas format: \(atlas.format)")
                print("   Modification count: \(atlas.modificationCount.withLock { $0 })")
                print("   Data size: \(atlas.data.count) bytes")
            }
            
            // Test string rendering
            print("\n5. Rendering string...")
            let testString = "Hello, Swift!"
            let stringCount = manager.prerenderString(testString)
            print("   Pre-rendered \(stringCount) characters from '\(testString)'")
            
            // Demonstrate caching
            print("\n6. Testing cache performance...")
            let start = CFAbsoluteTimeGetCurrent()
            
            // First render (cache miss)
            _ = manager.renderCharacter("X")
            let firstTime = CFAbsoluteTimeGetCurrent() - start
            
            let cacheStart = CFAbsoluteTimeGetCurrent()
            // Second render (cache hit)
            _ = manager.renderCharacter("X")
            let cacheTime = CFAbsoluteTimeGetCurrent() - cacheStart
            
            print("   First render: \(firstTime * 1000) ms")
            print("   Cached render: \(cacheTime * 1000) ms")
            print("   Cache speedup: \(firstTime / cacheTime)x")
            
            print("\n✅ Example completed successfully!")
            
        } catch {
            print("❌ Error: \(error)")
        }
    }
}

// Run the example if this file is executed directly
if CommandLine.arguments.count > 0 && CommandLine.arguments[0].contains("FontAtlasExample") {
    FontAtlasExample.run()
}