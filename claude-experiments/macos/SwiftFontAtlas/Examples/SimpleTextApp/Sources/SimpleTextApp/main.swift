import SwiftUI
import SwiftFontAtlas
import AppKit

@main
struct SimpleTextApp: App {
    init() {
        print("🚀 Starting Simple Text Rendering App...")
        
        // Force app to front
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }
    }
    
    var body: some Scene {
        WindowGroup("SwiftFontAtlas Text Demo") {
            TextDemoView()
        }
        .defaultSize(width: 900, height: 600)
    }
}

struct TextDemoView: View {
    @StateObject private var viewModel = TextDemoViewModel()
    
    var body: some View {
        HSplitView {
            // Controls
            VStack(alignment: .leading, spacing: 16) {
                Text("Text Demo")
                    .font(.title)
                    .fontWeight(.bold)
                
                GroupBox("Font Settings") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Font:")
                            Picker("", selection: $viewModel.selectedFont) {
                                Text("SF Mono").tag("SF Mono")
                                Text("Menlo").tag("Menlo")
                                Text("Monaco").tag("Monaco")
                                Text("Helvetica").tag("Helvetica")
                            }
                            .pickerStyle(.menu)
                        }
                        
                        HStack {
                            Text("Size:")
                            Slider(value: $viewModel.fontSize, in: 8...48, step: 1)
                            Text("\(Int(viewModel.fontSize))pt")
                                .frame(width: 35)
                        }
                        
                        Button("Create Atlas") {
                            viewModel.createAtlas()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isCreating)
                    }
                }
                
                GroupBox("Text Input") {
                    VStack(alignment: .leading, spacing: 8) {
                        TextEditor(text: $viewModel.inputText)
                            .font(.system(.body, design: .monospaced))
                            .frame(height: 120)
                            .border(Color.gray.opacity(0.3))
                        
                        Button("Render Text") {
                            viewModel.renderText()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!viewModel.hasAtlas)
                    }
                }
                
                GroupBox("Statistics") {
                    VStack(alignment: .leading, spacing: 4) {
                        if let stats = viewModel.stats {
                            Text("Atlas: \(stats.atlasSize)×\(stats.atlasSize)")
                            Text("Characters: \(stats.charCount)")
                            Text("Memory: \(stats.memoryUsage)")
                            Text("Utilization: \(stats.utilization)%")
                        } else {
                            Text("No atlas created")
                                .foregroundColor(.secondary)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                }
                
                Spacer()
            }
            .frame(width: 300)
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            // Rendering area
            VStack {
                if viewModel.hasAtlas {
                    ScrollView([.horizontal, .vertical]) {
                        VStack(alignment: .leading, spacing: 4) {
                            if !viewModel.renderedLines.isEmpty {
                                ForEach(Array(viewModel.renderedLines.enumerated()), id: \.offset) { index, line in
                                    HStack(spacing: 0) {
                                        ForEach(Array(line.glyphs.enumerated()), id: \.offset) { charIndex, glyph in
                                            GlyphView(glyph: glyph, cellSize: viewModel.cellSize)
                                        }
                                        Spacer()
                                    }
                                }
                            } else {
                                Text("Enter text and click 'Render Text' to see characters rendered through the font atlas.")
                                    .foregroundColor(.secondary)
                                    .padding()
                            }
                        }
                        .padding()
                    }
                    .background(Color.black)
                    .cornerRadius(8)
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "textformat")
                            .font(.system(size: 64))
                            .foregroundColor(.secondary)
                        
                        Text("Create a font atlas to start rendering text")
                            .font(.title2)
                            .foregroundColor(.secondary)
                        
                        Text("This demo shows text being rendered through SwiftFontAtlas with proper glyph positioning and atlas texture coordinates.")
                            .multilineTextAlignment(.center)
                            .foregroundColor(.secondary)
                            .frame(maxWidth: 400)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding()
        }
        .onAppear {
            print("✅ Text demo appeared!")
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}

struct GlyphView: View {
    let glyph: RenderedGlyph
    let cellSize: CGSize
    
    var body: some View {
        Rectangle()
            .fill(Color.white)
            .frame(width: CGFloat(glyph.width), height: CGFloat(glyph.height))
            .overlay(
                Text("•")
                    .font(.system(size: 8))
                    .foregroundColor(.red)
                    .opacity(0.7),
                alignment: .topLeading
            )
            .background(
                Rectangle()
                    .stroke(Color.blue.opacity(0.3), lineWidth: 0.5)
                    .frame(width: cellSize.width, height: cellSize.height)
            )
    }
}

@MainActor
class TextDemoViewModel: ObservableObject {
    @Published var selectedFont = "SF Mono"
    @Published var fontSize: Double = 18
    @Published var inputText = "Hello, SwiftFontAtlas!\n\nThis text is rendered using our\ncustom font atlas system.\n\n• Each character is a glyph\n• White rectangles show glyph bounds\n• Blue outlines show cell boundaries\n• Red dots mark glyph origins"
    @Published var hasAtlas = false
    @Published var isCreating = false
    @Published var stats: Stats?
    @Published var renderedLines: [RenderedLine] = []
    
    private var fontManager: FontAtlasManager?
    
    var cellSize: CGSize {
        fontManager?.cellSize ?? CGSize(width: 8, height: 16)
    }
    
    struct Stats {
        let atlasSize: UInt32
        let charCount: Int
        let memoryUsage: String
        let utilization: String
    }
    
    struct RenderedLine {
        let glyphs: [RenderedGlyph]
    }
    
    func createAtlas() {
        isCreating = true
        
        Task {
            do {
                let manager = try FontAtlasManager(
                    fontName: selectedFont,
                    fontSize: Float(fontSize),
                    atlasSize: 512
                )
                
                // Pre-render common characters
                _ = manager.prerenderASCII()
                
                await MainActor.run {
                    self.fontManager = manager
                    self.hasAtlas = true
                    self.isCreating = false
                    self.updateStats()
                    print("✅ Font atlas created with \(selectedFont) \(Int(fontSize))pt")
                }
            } catch {
                await MainActor.run {
                    self.isCreating = false
                    print("❌ Failed to create atlas: \(error)")
                }
            }
        }
    }
    
    func renderText() {
        guard let manager = fontManager else { return }
        
        renderedLines.removeAll()
        
        let lines = inputText.components(separatedBy: .newlines)
        
        for lineText in lines {
            var lineGlyphs: [RenderedGlyph] = []
            
            for character in lineText {
                if let glyph = manager.renderCharacter(character) {
                    lineGlyphs.append(glyph)
                } else if character == " " {
                    // Create a space glyph
                    let spaceGlyph = RenderedGlyph(
                        width: 0,
                        height: 0,
                        offsetX: 0,
                        offsetY: 0,
                        atlasX: 0,
                        atlasY: 0,
                        advanceX: Float(cellSize.width * 0.5)
                    )
                    lineGlyphs.append(spaceGlyph)
                }
            }
            
            renderedLines.append(RenderedLine(glyphs: lineGlyphs))
        }
        
        updateStats()
        
        let totalChars = renderedLines.reduce(0) { $0 + $1.glyphs.count }
        print("✅ Rendered \(totalChars) characters in \(renderedLines.count) lines")
    }
    
    private func updateStats() {
        guard let manager = fontManager else {
            stats = nil
            return
        }
        
        let totalChars = renderedLines.reduce(0) { $0 + $1.glyphs.count }
        
        manager.withAtlas { atlas in
            let memoryMB = Double(atlas.data.count) / (1024 * 1024)
            
            // Calculate utilization
            var nonZeroPixels = 0
            for byte in atlas.data {
                if byte != 0 { nonZeroPixels += 1 }
            }
            let utilization = Double(nonZeroPixels) / Double(atlas.data.count) * 100
            
            stats = Stats(
                atlasSize: atlas.size,
                charCount: totalChars,
                memoryUsage: String(format: "%.1f MB", memoryMB),
                utilization: String(format: "%.1f", utilization)
            )
        }
    }
}