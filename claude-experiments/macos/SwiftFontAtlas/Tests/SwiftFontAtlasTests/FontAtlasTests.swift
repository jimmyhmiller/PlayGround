import XCTest
@testable import SwiftFontAtlas

final class FontAtlasTests: XCTestCase {
    
    func testAtlasInitialization() throws {
        let atlas = try FontAtlas(size: 256, format: .grayscale)
        XCTAssertEqual(atlas.size, 256)
        XCTAssertEqual(atlas.format, .grayscale)
        XCTAssertEqual(atlas.data.count, 256 * 256 * 1)
    }
    
    func testInvalidSizeThrows() {
        // Size must be power of 2
        XCTAssertThrowsError(try FontAtlas(size: 100, format: .grayscale))
        XCTAssertThrowsError(try FontAtlas(size: 0, format: .grayscale))
    }
    
    func testReserveRegion() throws {
        let atlas = try FontAtlas(size: 64, format: .grayscale)
        
        // Reserve a small region
        let region = try atlas.reserve(width: 10, height: 10)
        XCTAssertGreaterThan(region.width, 0)
        XCTAssertGreaterThan(region.height, 0)
        
        // Check that it's within bounds (accounting for 1px border)
        XCTAssertGreaterThan(region.x, 0)
        XCTAssertGreaterThan(region.y, 0)
        XCTAssertLessThan(region.x + region.width, atlas.size - 1)
        XCTAssertLessThan(region.y + region.height, atlas.size - 1)
    }
    
    func testAtlasFullError() throws {
        let atlas = try FontAtlas(size: 4, format: .grayscale) // Very small atlas
        
        // Try to reserve too large a region
        XCTAssertThrowsError(try atlas.reserve(width: 10, height: 10)) { error in
            XCTAssertEqual(error as? FontAtlasError, .atlasFull)
        }
    }
    
    func testSetData() throws {
        let atlas = try FontAtlas(size: 64, format: .grayscale)
        let region = try atlas.reserve(width: 2, height: 2)
        
        let testData = Data([255, 128, 64, 32])
        atlas.set(region: region, data: testData)
        
        // Verify data was written
        let depth = atlas.format.bytesPerPixel
        let offset = Int((region.y * atlas.size + region.x)) * depth
        
        XCTAssertEqual(atlas.data[offset], 255)
        XCTAssertEqual(atlas.data[offset + 1], 128)
        XCTAssertEqual(atlas.data[offset + Int(atlas.size)], 64)
        XCTAssertEqual(atlas.data[offset + Int(atlas.size) + 1], 32)
    }
    
    func testAtlasGrow() throws {
        var atlas = try FontAtlas(size: 32, format: .grayscale)
        
        // Fill with some data
        let region = try atlas.reserve(width: 4, height: 4)
        let testData = Data(repeating: 255, count: 16)
        atlas.set(region: region, data: testData)
        
        let oldModCount = atlas.modificationCount.withLock { $0 }
        let oldResizeCount = atlas.resizeCount.withLock { $0 }
        
        // Grow the atlas
        try atlas.grow(to: 64)
        
        XCTAssertEqual(atlas.size, 64)
        XCTAssertGreaterThan(atlas.modificationCount.withLock { $0 }, oldModCount)
        XCTAssertGreaterThan(atlas.resizeCount.withLock { $0 }, oldResizeCount)
    }
    
    func testNormalizedCoordinates() throws {
        let atlas = try FontAtlas(size: 128, format: .grayscale) // Use power of 2
        let region = AtlasRegion(x: 25, y: 50, width: 10, height: 20)
        
        let coords = atlas.normalizedCoordinates(for: region)
        
        XCTAssertEqual(coords.0, 25.0/128.0, accuracy: 0.001) // x1
        XCTAssertEqual(coords.1, 50.0/128.0, accuracy: 0.001)  // y1
        XCTAssertEqual(coords.2, 35.0/128.0, accuracy: 0.001) // x2
        XCTAssertEqual(coords.3, 70.0/128.0, accuracy: 0.001)  // y2
    }
    
    func testMultipleReservations() throws {
        let atlas = try FontAtlas(size: 128, format: .grayscale)
        
        // Reserve multiple regions
        let region1 = try atlas.reserve(width: 10, height: 10)
        let region2 = try atlas.reserve(width: 15, height: 8)
        let region3 = try atlas.reserve(width: 5, height: 12)
        
        // Ensure they don't overlap
        let regions = [region1, region2, region3]
        for i in 0..<regions.count {
            for j in (i+1)..<regions.count {
                let r1 = regions[i]
                let r2 = regions[j]
                
                let noOverlap = (r1.x + r1.width <= r2.x) ||
                               (r2.x + r2.width <= r1.x) ||
                               (r1.y + r1.height <= r2.y) ||
                               (r2.y + r2.height <= r1.y)
                
                XCTAssertTrue(noOverlap, "Regions \(i) and \(j) overlap")
            }
        }
    }
    
    func testPixelFormats() throws {
        XCTAssertEqual(PixelFormat.grayscale.bytesPerPixel, 1)
        XCTAssertEqual(PixelFormat.bgr.bytesPerPixel, 3)
        XCTAssertEqual(PixelFormat.bgra.bytesPerPixel, 4)
    }
}