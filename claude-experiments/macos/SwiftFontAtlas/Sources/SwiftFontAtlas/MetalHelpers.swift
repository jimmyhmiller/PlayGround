import Metal
import MetalKit
import simd

/// Helper class for common Metal text rendering setup
public class MetalTextHelper {
    /// Create a simple MTKView configured for text rendering
    /// - Parameters:
    ///   - frame: View frame
    ///   - device: Metal device
    /// - Returns: Configured MTKView
    @MainActor
    public static func createTextView(frame: CGRect, device: MTLDevice) -> MTKView {
        let view = MTKView(frame: frame, device: device)
        
        // Configure for text rendering
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.isPaused = false
        view.enableSetNeedsDisplay = true
        
        return view
    }
    
    /// Create a basic command queue
    /// - Parameter device: Metal device
    /// - Returns: Command queue
    public static func createCommandQueue(device: MTLDevice) -> MTLCommandQueue? {
        return device.makeCommandQueue()
    }
    
    /// Simple text rendering delegate
    open class SimpleTextDelegate: NSObject, MTKViewDelegate {
        public let device: MTLDevice
        public let commandQueue: MTLCommandQueue
        public var textRenderer: TextRenderer?
        public var clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        /// Text to render (override in subclass or set directly)
        open var textToRender: String {
            return "Hello, World!"
        }
        
        /// Text color
        open var textColor: simd_float4 {
            return simd_float4(1, 1, 1, 1)
        }
        
        /// Text position
        open var textPosition: CGPoint {
            return CGPoint(x: 20, y: 50)
        }
        
        @MainActor
        public init?(metalKitView: MTKView) {
            guard let device = metalKitView.device,
                  let queue = device.makeCommandQueue() else {
                return nil
            }
            
            self.device = device
            self.commandQueue = queue
            
            super.init()
            
            // Setup text renderer
            do {
                self.textRenderer = try TextRenderer(
                    device: device,
                    fontName: "SF Mono",
                    fontSize: 16,
                    coordinateOrigin: .topLeft
                )
            } catch {
                print("Failed to create text renderer: \(error)")
                return nil
            }
        }
        
        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            // Handle size changes if needed
        }
        
        public func draw(in view: MTKView) {
            guard let textRenderer = textRenderer,
                  let drawable = view.currentDrawable,
                  let descriptor = view.currentRenderPassDescriptor else {
                return
            }
            
            descriptor.colorAttachments[0].clearColor = clearColor
            
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
                return
            }
            
            // Set projection matrix
            textRenderer.setProjectionMatrix(using: renderEncoder, viewportSize: view.drawableSize)
            
            // Draw text
            renderText(using: renderEncoder, textRenderer: textRenderer)
            
            renderEncoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
        
        /// Override this method to customize text rendering
        open func renderText(using renderEncoder: MTLRenderCommandEncoder, textRenderer: TextRenderer) {
            textRenderer.drawText(
                textToRender,
                at: textPosition,
                color: textColor,
                using: renderEncoder
            )
        }
    }
}

/// Example usage delegate that demonstrates the simplified API
public class ExampleTextDelegate: MetalTextHelper.SimpleTextDelegate {
    private var frameCount = 0
    
    public override var textToRender: String {
        return """
        Frame: \(frameCount)
        SwiftFontAtlas Demo
        
        This text is rendered using the high-level API.
        No manual vertex generation required!
        
        Features:
        • Automatic vertex generation
        • Built-in shader compilation
        • Simple one-line text rendering
        • Configurable coordinate systems
        """
    }
    
    public override var textColor: simd_float4 {
        // Animate color
        let hue = Float(frameCount % 360) / 360.0
        return simd_float4(
            0.5 + 0.5 * cos(hue * 2 * .pi),
            0.5 + 0.5 * cos((hue + 0.33) * 2 * .pi),
            0.5 + 0.5 * cos((hue + 0.67) * 2 * .pi),
            1.0
        )
    }
    
    public override func renderText(using renderEncoder: MTLRenderCommandEncoder, textRenderer: TextRenderer) {
        frameCount += 1
        
        // Draw main text
        super.renderText(using: renderEncoder, textRenderer: textRenderer)
        
        // Draw wrapped text in a box
        let rect = CGRect(x: 400, y: 100, width: 300, height: 200)
        textRenderer.drawWrappedText(
            "This text is automatically wrapped to fit within the specified rectangle. The TextLayout system handles word wrapping, line breaking, and text alignment.",
            in: rect,
            wrapMode: TextLayout.WrapMode.word,
            alignment: TextRenderer.TextAlignment.left,
            color: simd_float4(0.8, 0.8, 1.0, 1.0),
            using: renderEncoder
        )
        
        // Draw centered text
        textRenderer.drawText(
            "Centered Text",
            in: CGRect(x: 400, y: 350, width: 300, height: 50),
            alignment: TextRenderer.TextAlignment.center,
            color: simd_float4(1.0, 1.0, 0.8, 1.0),
            using: renderEncoder
        )
        
        // Draw right-aligned text
        textRenderer.drawText(
            "Right Aligned",
            in: CGRect(x: 400, y: 400, width: 300, height: 50),
            alignment: TextRenderer.TextAlignment.right,
            color: simd_float4(1.0, 0.8, 0.8, 1.0),
            using: renderEncoder
        )
    }
}