#!/usr/bin/env swift

import SwiftUI
import SwiftFontAtlas
import CoreGraphics
import AppKit

@main
struct FontAtlasDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
    }
}

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
    @Published var logMessages: [String] = []
    
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
            addLog("✅ Atlas created successfully with \(selectedFont) \(Int(fontSize))pt")
        } catch {
            addLog("❌ Failed to create atlas: \(error)")
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
        addLog("✅ Prerendered \(count) ASCII characters in \(String(format: "%.2f", operationTime))ms")
    }
    
    func renderCustomText() {
        guard let manager = fontManager else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let count = manager.prerenderString(customText)
        let operationTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        renderCount += count
        updateVisualization()
        updateStatistics(operationTime: operationTime)
        addLog("✅ Rendered \(count) characters from '\(customText)' in \(String(format: "%.2f", operationTime))ms")
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
        addLog("✅ Atlas cleared in \(String(format: "%.2f", operationTime))ms")
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
        addLog("✅ Stress test completed: \(totalCount) characters in \(String(format: "%.2f", operationTime))ms")
    }
    
    func demonstrateAtlasGrowth() {
        guard let manager = fontManager else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Force atlas to grow by rendering many unique characters
        let unicodeRanges = [
            0x0100...0x017F, // Latin Extended-A
            0x0180...0x024F, // Latin Extended-B
            0x1E00...0x1EFF, // Latin Extended Additional
        ]
        
        var count = 0
        for range in unicodeRanges {
            for codepoint in range.prefix(50) { // Limit to avoid too much growth
                if let scalar = UnicodeScalar(codepoint) {
                    let char = Character(scalar)
                    if manager.renderCharacter(char) != nil {
                        count += 1
                    }
                }
            }
        }
        
        let operationTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        renderCount += count
        updateVisualization()
        updateStatistics(operationTime: operationTime)
        addLog("✅ Atlas growth test: \(count) extended characters in \(String(format: "%.2f", operationTime))ms")
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
    
    private func addLog(_ message: String) {
        let timestamp = DateFormatter().apply {
            $0.dateFormat = "HH:mm:ss.SSS"
        }.string(from: Date())
        
        logMessages.append("[\(timestamp)] \(message)")
        
        // Keep only last 50 messages
        if logMessages.count > 50 {
            logMessages.removeFirst(logMessages.count - 50)
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

struct ContentView: View {
    @StateObject private var viewModel = FontAtlasViewModel()
    
    var body: some View {
        HSplitView {
            // Left panel - Controls
            VStack(alignment: .leading, spacing: 20) {
                Text("SwiftFontAtlas Demo")
                    .font(.title)
                    .fontWeight(.bold)
                
                // Font Settings
                VStack(alignment: .leading, spacing: 10) {
                    Text("Font Settings")
                        .font(.headline)
                    
                    HStack {
                        Text("Font:")
                        Picker("Font", selection: $viewModel.selectedFont) {
                            ForEach(viewModel.availableFonts, id: \.self) { font in
                                Text(font).tag(font)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    
                    HStack {
                        Text("Size:")
                        Slider(value: $viewModel.fontSize, in: 8...48, step: 1)
                        Text("\(Int(viewModel.fontSize))pt")
                            .frame(width: 40)
                    }
                    
                    VStack(alignment: .leading) {
                        Text("Atlas Size:")
                        Picker("Atlas Size", selection: $viewModel.atlasSize) {
                            ForEach([256, 512, 1024, 2048], id: \.self) { size in
                                Text("\(size)").tag(UInt32(size))
                            }
                        }
                        .pickerStyle(.segmented)
                    }
                    
                    Button("Create Atlas") {
                        viewModel.createAtlas()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
                
                // Actions
                VStack(alignment: .leading, spacing: 10) {
                    Text("Actions")
                        .font(.headline)
                    
                    Button("Prerender ASCII") {
                        viewModel.prerenderASCII()
                    }
                    .disabled(!viewModel.hasAtlas)
                    
                    HStack {
                        TextField("Custom text", text: $viewModel.customText)
                        Button("Render") {
                            viewModel.renderCustomText()
                        }
                        .disabled(!viewModel.hasAtlas)
                    }
                    
                    Button("Stress Test") {
                        viewModel.stressTest()
                    }
                    .disabled(!viewModel.hasAtlas)
                    
                    Button("Atlas Growth Test") {
                        viewModel.demonstrateAtlasGrowth()
                    }
                    .disabled(!viewModel.hasAtlas)
                    
                    Button("Clear Atlas") {
                        viewModel.clearAtlas()
                    }
                    .disabled(!viewModel.hasAtlas)
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
                
                // Statistics
                VStack(alignment: .leading, spacing: 10) {
                    Text("Statistics")
                        .font(.headline)
                    
                    if let stats = viewModel.statistics {
                        VStack(alignment: .leading, spacing: 5) {
                            Text("Atlas: \(stats.atlasSize)×\(stats.atlasSize)")
                            Text("Glyphs: \(stats.glyphCount)")
                            Text("Memory: \(stats.memoryUsage)")
                            Text("Mods: \(stats.modificationCount)")
                            Text("Cell: \(String(format: "%.1f×%.1f", stats.cellWidth, stats.cellHeight))")
                            Text("Last: \(String(format: "%.2f", stats.lastOperationTime))ms")
                        }
                        .font(.system(.caption, design: .monospaced))
                    } else {
                        Text("No atlas created")
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
                
                // Log
                VStack(alignment: .leading, spacing: 5) {
                    Text("Log")
                        .font(.headline)
                    
                    ScrollView {
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(viewModel.logMessages, id: \.self) { message in
                                Text(message)
                                    .font(.system(.caption, design: .monospaced))
                                    .textSelection(.enabled)
                            }
                        }
                    }
                    .frame(height: 100)
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
                
                Spacer()
            }
            .frame(width: 320)
            .padding()
            
            // Right panel - Visualization
            VStack {
                Text("Atlas Visualization")
                    .font(.title2)
                    .padding(.top)
                
                if let atlasImage = viewModel.atlasImage {
                    ScrollView([.horizontal, .vertical]) {
                        Image(nsImage: atlasImage)
                            .interpolation(.none)
                            .scaleEffect(viewModel.zoomLevel)
                            .onTapGesture(count: 2) {
                                viewModel.resetZoom()
                            }
                    }
                    .overlay(alignment: .bottomTrailing) {
                        VStack(spacing: 4) {
                            Button("+") { viewModel.zoomIn() }
                            Button("−") { viewModel.zoomOut() }
                            Button("1:1") { viewModel.resetZoom() }
                            Text("\(Int(viewModel.zoomLevel * 100))%")
                                .font(.caption)
                        }
                        .padding()
                        .background(.regularMaterial)
                        .cornerRadius(8)
                        .padding()
                    }
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "photo")
                            .font(.system(size: 64))
                            .foregroundColor(.secondary)
                        
                        VStack(spacing: 10) {
                            Text("Create an atlas to see visualization")
                                .font(.title3)
                                .foregroundColor(.secondary)
                            
                            Text("The atlas shows rendered glyphs as white pixels on a black background. Each glyph is packed efficiently using rectangle bin packing.")
                                .multilineTextAlignment(.center)
                                .foregroundColor(.secondary)
                                .frame(maxWidth: 300)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .frame(minWidth: 900, minHeight: 700)
        .onAppear {
            viewModel.addLog("SwiftFontAtlas Demo started")
            viewModel.addLog("Select font settings and click 'Create Atlas' to begin")
        }
    }
}

extension DateFormatter {
    func apply(_ closure: (DateFormatter) -> Void) -> DateFormatter {
        closure(self)
        return self
    }
}