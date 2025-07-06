import XCTest
@testable import SwiftFontAtlas

final class FontAtlasManagerTests: XCTestCase {
    
    func testManagerInitialization() throws {
        let manager = try FontAtlasManager(
            fontName: "Helvetica",
            fontSize: 12.0,
            atlasSize: 256
        )
        
        XCTAssertGreaterThan(manager.metrics.cellWidth, 0)
        XCTAssertGreaterThan(manager.metrics.cellHeight, 0)
        XCTAssertGreaterThan(manager.cellSize.width, 0)
        XCTAssertGreaterThan(manager.cellSize.height, 0)
    }
    
    func testInvalidFontThrows() {
        // Note: CTFont will create a fallback font even if the name doesn't exist
        // So this test might not actually throw. Let's test with an invalid font size instead
        XCTAssertThrowsError(
            try FontAtlasManager(
                fontName: "Helvetica",
                fontSize: -1.0 // Invalid font size
            )
        ) { error in
            // We expect some kind of error, not necessarily fontCreationFailed
        }
    }
    
    func testRenderBasicCharacters() throws {
        let manager = try FontAtlasManager(
            fontName: "Helvetica",
            fontSize: 12.0,
            atlasSize: 256
        )
        
        // Test rendering some basic characters
        let characters = ["A", "B", "1", "!", " "]
        
        for char in characters {
            let glyph = manager.renderCharacter(Character(char))
            
            if char == " " {
                // Space should have advance but might have empty dimensions
                if let g = glyph {
                    XCTAssertGreaterThan(g.advanceX, 0)
                    // Space might or might not have dimensions, that's OK
                }
            } else {
                // Other characters should render
                XCTAssertNotNil(glyph, "Failed to render character: \(char)")
                if let g = glyph {
                    XCTAssertGreaterThan(g.advanceX, 0)
                }
            }
        }
    }
    
    func testCaching() throws {
        let manager = try FontAtlasManager(
            fontName: "Helvetica",
            fontSize: 12.0,
            atlasSize: 256
        )
        
        let character = Character("X")
        
        // Render the same character twice
        let glyph1 = manager.renderCharacter(character)
        let glyph2 = manager.renderCharacter(character)
        
        XCTAssertNotNil(glyph1)
        XCTAssertNotNil(glyph2)
        XCTAssertEqual(glyph1, glyph2)
    }
    
    func testCodepointRendering() throws {
        let manager = try FontAtlasManager(
            fontName: "Helvetica",
            fontSize: 12.0,
            atlasSize: 256
        )
        
        // Test ASCII 'A' (65)
        let glyph = manager.renderCodepoint(65)
        XCTAssertNotNil(glyph)
        
        // Test invalid codepoint
        let invalidGlyph = manager.renderCodepoint(0xFFFFFFFF)
        XCTAssertNil(invalidGlyph)
    }
    
    func testPrerenderASCII() throws {
        let manager = try FontAtlasManager(
            fontName: "Helvetica",
            fontSize: 12.0,
            atlasSize: 512
        )
        
        let count = manager.prerenderASCII()
        
        // Should have rendered most ASCII characters
        XCTAssertGreaterThan(count, 90) // Printable ASCII is 32-126 = 95 chars
        
        // Verify some characters are cached
        let cachedGlyph = manager.renderCharacter("A")
        XCTAssertNotNil(cachedGlyph)
    }
    
    func testPrerenderString() throws {
        let manager = try FontAtlasManager(
            fontName: "Helvetica",
            fontSize: 12.0,
            atlasSize: 256
        )
        
        let testString = "Hello, World!"
        let count = manager.prerenderString(testString)
        
        XCTAssertGreaterThan(count, 0)
        
        // Verify characters from string are cached
        for char in testString {
            let glyph = manager.renderCharacter(char)
            // Most characters should be renderable
            if char != "," && char != "!" {
                XCTAssertNotNil(glyph, "Character '\(char)' should be cached")
            }
        }
    }
    
    func testModificationTracking() throws {
        let manager = try FontAtlasManager(
            fontName: "Helvetica",
            fontSize: 12.0,
            atlasSize: 256
        )
        
        let initialModCount = manager.modificationCount
        
        // Render a character (should modify atlas)
        _ = manager.renderCharacter("A")
        
        let newModCount = manager.modificationCount
        XCTAssertGreaterThan(newModCount, initialModCount)
        
        // Check modification status
        XCTAssertTrue(manager.isModified(since: initialModCount))
        XCTAssertFalse(manager.isModified(since: newModCount))
    }
    
    func testAtlasAccess() throws {
        let manager = try FontAtlasManager(
            fontName: "Helvetica",
            fontSize: 12.0,
            atlasSize: 256
        )
        
        var atlasSize: UInt32 = 0
        manager.withAtlas { atlas in
            atlasSize = atlas.size
        }
        
        XCTAssertEqual(atlasSize, 256)
    }
}