import SwiftUI
import SwiftFontAtlas
import MetalKit

struct MetalTextView: NSViewRepresentable {
    let viewModel: RealTextViewModel
    
    func makeNSView(context: Context) -> MTKView {
        let metalView = MTKView()
        metalView.device = viewModel.device
        metalView.delegate = viewModel
        metalView.preferredFramesPerSecond = 60
        metalView.enableSetNeedsDisplay = false
        metalView.isPaused = false
        metalView.clearColor = MTLClearColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        
        print("✅ Metal view created")
        return metalView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Updates handled by MTKViewDelegate
    }
}

struct TextVertex {
    let position: simd_float2
    let texCoord: simd_float2
    let color: simd_float4
}

struct TextUniforms {
    let projectionMatrix: simd_float4x4
    let screenSize: simd_float2
}