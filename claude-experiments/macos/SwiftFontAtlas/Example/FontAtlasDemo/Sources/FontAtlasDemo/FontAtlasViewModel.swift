import SwiftUI
import SwiftFontAtlas
import CoreGraphics
import AppKit

@MainActor
class FontAtlasViewModel: ObservableObject {
    @Published var selectedFont = "SF Mono"
    @Published var fontSize: Double = 14
    @Published var atlasSize: UInt32 = 512
    @Published var customText = "Hello, World! 🚀"
    @Published var atlasImage: NSImage?
    @Published var statistics: AtlasStatistics?
    @Published var zoomLevel: Double = 1.0
    @Published var hasAtlas = false
    
    private var fontManager: FontAtlasManager?
    private var renderCount = 0
    
    let availableFonts = [
        "SF Mono",
        "Helvetica",
        "Courier",
        "Monaco",
        "Menlo",
        "Times",
        "Arial"
    ]
    
    struct AtlasStatistics {
        let atlasSize: UInt32
        let glyphCount: Int
        let memoryUsage: String
        let modificationCount: UInt64
        let cellWidth: Double
        let cellHeight: Double
        let lastOperationTime: Double
    }
    
    func createAtlas() {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            fontManager = try FontAtlasManager(
                fontName: selectedFont,
                fontSize: Float(fontSize),
                atlasSize: atlasSize
            )
            hasAtlas = true
            updateVisualization()
            updateStatistics(operationTime: (CFAbsoluteTimeGetCurrent() - startTime) * 1000)
            print("✅ Atlas created successfully")
        } catch {
            print("❌ Failed to create atlas: \(error)")
            hasAtlas = false
        }
    }
    
    func prerenderASCII() {
        guard let manager = fontManager else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let count = manager.prerenderASCII()
        let operationTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        renderCount += count
        updateVisualization()
        updateStatistics(operationTime: operationTime)
        print("✅ Prerendered \(count) ASCII characters")
    }
    
    func renderCustomText() {
        guard let manager = fontManager else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let count = manager.prerenderString(customText)
        let operationTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        renderCount += count
        updateVisualization()
        updateStatistics(operationTime: operationTime)
        print("✅ Rendered \(count) characters from custom text")
    }
    
    func clearAtlas() {
        guard let manager = fontManager else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        manager.withAtlas { atlas in
            atlas.clear()
        }
        let operationTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        renderCount = 0
        updateVisualization()
        updateStatistics(operationTime: operationTime)
        print("✅ Atlas cleared")
    }
    
    func stressTest() {
        guard let manager = fontManager else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Test with various Unicode ranges
        let testStrings = [
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "!@#$%^&*()_+-=[]{}|;:,.<>?",
            "áéíóúàèìòùâêîôûäëïöüç",
            "αβγδεζηθικλμνξοπρστυφχψω",
            "→←↑↓⇒⇐⇑⇓⇔⇕"
        ]
        
        var totalCount = 0
        for testString in testStrings {
            totalCount += manager.prerenderString(testString)
        }
        
        let operationTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        renderCount += totalCount
        updateVisualization()
        updateStatistics(operationTime: operationTime)
        print("✅ Stress test completed: \(totalCount) characters rendered")
    }
    
    private func updateVisualization() {
        guard let manager = fontManager else {
            atlasImage = nil
            return
        }
        
        manager.withAtlas { atlas in
            atlasImage = createAtlasImage(from: atlas)
        }
    }
    
    private func createAtlasImage(from atlas: FontAtlas) -> NSImage? {
        let size = Int(atlas.size)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        
        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return nil
        }
        
        // Copy atlas data to context
        atlas.data.withUnsafeBytes { bytes in
            if let data = context.data {
                data.copyMemory(from: bytes.baseAddress!, byteCount: atlas.data.count)
            }
        }
        
        guard let cgImage = context.makeImage() else { return nil }
        
        // Convert to NSImage and flip vertically (Core Graphics is bottom-up)
        let image = NSImage(cgImage: cgImage, size: NSSize(width: size, height: size))
        image.isFlipped = true
        
        return image
    }
    
    private func updateStatistics(operationTime: Double) {
        guard let manager = fontManager else {
            statistics = nil
            return
        }
        
        manager.withAtlas { atlas in
            let memoryMB = Double(atlas.data.count) / (1024 * 1024)
            
            statistics = AtlasStatistics(
                atlasSize: atlas.size,
                glyphCount: renderCount,
                memoryUsage: String(format: "%.2f MB", memoryMB),
                modificationCount: atlas.modificationCount.withLock { $0 },
                cellWidth: manager.cellSize.width,
                cellHeight: manager.cellSize.height,
                lastOperationTime: operationTime
            )
        }
    }
    
    func zoomIn() {
        zoomLevel = min(zoomLevel * 1.5, 8.0)
    }
    
    func zoomOut() {
        zoomLevel = max(zoomLevel / 1.5, 0.25)
    }
    
    func resetZoom() {
        zoomLevel = 1.0
    }
}