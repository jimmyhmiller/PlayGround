#!/usr/bin/env swift

import SwiftUI
import AppKit

@main
struct SimpleViewerApp: App {
    init() {
        print("🚀 Simple Font Atlas Viewer starting...")
        
        // Force app to front
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }
    }
    
    var body: some Scene {
        WindowGroup("Simple Font Atlas Viewer") {
            SimpleContentView()
        }
        .defaultSize(width: 800, height: 600)
    }
}

struct SimpleContentView: View {
    @State private var atlasImage: NSImage?
    @State private var status = "Ready to create atlas"
    @State private var isCreating = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Font Atlas Viewer")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text(status)
                .foregroundColor(.secondary)
            
            Button(action: createAndShowAtlas) {
                Text(isCreating ? "Creating..." : "Create Font Atlas")
            }
            .disabled(isCreating)
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            
            if let image = atlasImage {
                ScrollView([.horizontal, .vertical]) {
                    Image(nsImage: image)
                        .interpolation(.none)
                        .scaleEffect(3.0) // 3x zoom to see pixels
                        .background(Color.black)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
                
                Text("White pixels = rendered glyphs, Black = empty space")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                Rectangle()
                    .fill(Color.gray.opacity(0.1))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .overlay(
                        VStack {
                            Image(systemName: "textformat")
                                .font(.system(size: 48))
                                .foregroundColor(.secondary)
                            Text("Atlas will appear here")
                                .foregroundColor(.secondary)
                        }
                    )
                    .cornerRadius(8)
            }
        }
        .padding(20)
        .frame(minWidth: 600, minHeight: 500)
        .onAppear {
            print("✅ Simple viewer appeared!")
            NSApp.activate(ignoringOtherApps: true)
        }
    }
    
    func createAndShowAtlas() {
        print("📝 Creating atlas...")
        isCreating = true
        status = "Creating font atlas..."
        
        // Import SwiftFontAtlas here to avoid module issues
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // We'll create this inline to avoid import issues
                let atlasData = createSimpleAtlas()
                
                DispatchQueue.main.async {
                    self.atlasImage = atlasData.image
                    self.status = atlasData.message
                    self.isCreating = false
                    print("✅ Atlas created and displayed!")
                }
            } catch {
                DispatchQueue.main.async {
                    self.status = "Error: \(error)"
                    self.isCreating = false
                    print("❌ Error creating atlas: \(error)")
                }
            }
        }
    }
}

// Simple atlas creation without external dependencies
func createSimpleAtlas() -> (image: NSImage, message: String) {
    let size = 256
    
    // Create a simple test pattern
    var data = Data(count: size * size)
    
    // Draw some test patterns
    for y in 0..<size {
        for x in 0..<size {
            let index = y * size + x
            
            // Create a simple pattern
            if (x / 16 + y / 16) % 2 == 0 {
                data[index] = 128 // Gray
            } else if x < 50 && y < 50 {
                data[index] = 255 // White square in corner
            } else {
                data[index] = 0 // Black
            }
        }
    }
    
    // Convert to NSImage
    guard let context = CGContext(
        data: data.withUnsafeMutableBytes { $0.baseAddress },
        width: size,
        height: size,
        bitsPerComponent: 8,
        bytesPerRow: size,
        space: CGColorSpaceCreateDeviceGray(),
        bitmapInfo: CGImageAlphaInfo.none.rawValue
    ) else {
        return (NSImage(), "Failed to create context")
    }
    
    guard let cgImage = context.makeImage() else {
        return (NSImage(), "Failed to create image")
    }
    
    let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: size, height: size))
    
    return (nsImage, "Test atlas created - checkerboard pattern with white square")
}