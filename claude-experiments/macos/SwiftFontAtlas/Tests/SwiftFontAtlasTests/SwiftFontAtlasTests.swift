import Testing
@testable import SwiftFontAtlas

@Test func basicAtlasCreation() async throws {
    let atlas = try FontAtlas(size: 256, format: .grayscale)
    #expect(atlas.size == 256)
    #expect(atlas.format == .grayscale)
}

@Test func fontAtlasManagerCreation() async throws {
    let manager = try FontAtlasManager(
        fontName: "Helvetica",
        fontSize: 12.0,
        atlasSize: 256
    )
    
    #expect(manager.metrics.cellWidth > 0)
    #expect(manager.metrics.cellHeight > 0)
}

@Test func characterRendering() async throws {
    let manager = try FontAtlasManager(
        fontName: "Helvetica",
        fontSize: 12.0,
        atlasSize: 256
    )
    
    let glyph = manager.renderCharacter("A")
    #expect(glyph != nil)
    
    if let g = glyph {
        #expect(g.advanceX > 0)
    }
}
