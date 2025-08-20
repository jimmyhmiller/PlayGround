import SwiftUI
import SwiftFontAtlas
import CoreGraphics

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
                    
                    HStack {
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
                    
                    Button("Clear Atlas") {
                        viewModel.clearAtlas()
                    }
                    .disabled(!viewModel.hasAtlas)
                    
                    Button("Stress Test") {
                        viewModel.stressTest()
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
                            Text("Glyphs cached: \(stats.glyphCount)")
                            Text("Memory: \(stats.memoryUsage)")
                            Text("Modifications: \(stats.modificationCount)")
                            Text("Cell size: \(String(format: "%.1f×%.1f", stats.cellWidth, stats.cellHeight))")
                            Text("Last operation: \(String(format: "%.3f", stats.lastOperationTime))ms")
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
                
                Spacer()
            }
            .frame(width: 300)
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
                        VStack {
                            Button("+") { viewModel.zoomIn() }
                            Button("−") { viewModel.zoomOut() }
                            Button("1:1") { viewModel.resetZoom() }
                        }
                        .padding()
                    }
                } else {
                    VStack {
                        Image(systemName: "photo")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("Create an atlas to see visualization")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .frame(minWidth: 800, minHeight: 600)
    }
}