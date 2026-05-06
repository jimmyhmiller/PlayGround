import AppKit
import CoreText
import Metal
import MetalKit
import simd

private let maxCutouts = 8

private struct SurfaceUniform {
    var rect: SIMD4<Float>
    var color: SIMD4<Float>
    var params: SIMD4<Float>
    var cutoutCount: UInt32
    var shadowCount: UInt32
    var isOuterRect: UInt32
    var isPressed: UInt32
    var cutoutRect0: SIMD4<Float> = .zero
    var cutoutRect1: SIMD4<Float> = .zero
    var cutoutRect2: SIMD4<Float> = .zero
    var cutoutRect3: SIMD4<Float> = .zero
    var cutoutRect4: SIMD4<Float> = .zero
    var cutoutRect5: SIMD4<Float> = .zero
    var cutoutRect6: SIMD4<Float> = .zero
    var cutoutRect7: SIMD4<Float> = .zero
    var cutoutRadiiA: SIMD4<Float> = .zero
    var cutoutRadiiB: SIMD4<Float> = .zero
    var shadowRect0: SIMD4<Float> = .zero
    var shadowRect1: SIMD4<Float> = .zero
    var shadowRect2: SIMD4<Float> = .zero
    var shadowRect3: SIMD4<Float> = .zero
    var shadowRect4: SIMD4<Float> = .zero
    var shadowRect5: SIMD4<Float> = .zero
    var shadowRect6: SIMD4<Float> = .zero
    var shadowRect7: SIMD4<Float> = .zero
    var shadowRadiiA: SIMD4<Float> = .zero
    var shadowRadiiB: SIMD4<Float> = .zero
}

private struct TextVertex {
    var position: SIMD2<Float>
    var uv: SIMD2<Float>
}

private struct ViewUniforms {
    var viewport: SIMD2<Float>
}

struct RenderTuning {
    var grainScale: Float = 2.0
    var grainAmount: Float = 0.120
    var edgeDarkening: Float = 0.044
    var upperHighlight: Float = 0.084
    var lowLayerBrightness: Float = 1.100
    var highLayerBrightness: Float = 1.221
    var upperEdgeMark: Float = 0.035
    var ambientOcclusion: Float = 1.123
    var aoRadius: Float = 12.271
    var shadowStrength: Float = 0.493
    var shadowRadius: Float = 21.732
    var shadowX: Float = 0.045
    var shadowY: Float = 0.036
    var recessBoost: Float = 0.466
}

private struct RenderTuningUniform {
    var shadingA: SIMD4<Float>
    var shadingB: SIMD4<Float>
    var shadingC: SIMD4<Float>
    var shadingD: SIMD4<Float>

    init(_ tuning: RenderTuning) {
        shadingA = SIMD4(tuning.grainScale, tuning.grainAmount, tuning.edgeDarkening, tuning.upperHighlight)
        shadingB = SIMD4(tuning.lowLayerBrightness, tuning.highLayerBrightness, tuning.upperEdgeMark, tuning.ambientOcclusion)
        shadingC = SIMD4(tuning.aoRadius, tuning.shadowStrength, tuning.shadowRadius, tuning.shadowX)
        shadingD = SIMD4(tuning.shadowY, tuning.recessBoost, 0, 0)
    }
}

@MainActor
protocol PaperInteractionController: AnyObject {
    func handleClick(at point: CGPoint)
    func handleHover(at point: CGPoint)
    func handleCharacters(_ characters: String)
    func handleSpecialKey(_ keyCode: UInt16)
}

@MainActor
final class PaperCutView: MTKView {
    weak var controller: PaperInteractionController?

    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        controller?.handleClick(at: convert(event.locationInWindow, from: nil))
    }

    override func mouseMoved(with event: NSEvent) {
        controller?.handleHover(at: convert(event.locationInWindow, from: nil))
    }

    override func mouseDragged(with event: NSEvent) {
        controller?.handleHover(at: convert(event.locationInWindow, from: nil))
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(
            NSTrackingArea(
                rect: bounds,
                options: [.activeInKeyWindow, .mouseMoved, .inVisibleRect],
                owner: self,
                userInfo: nil
            )
        )
    }

    override func keyDown(with event: NSEvent) {
        if let characters = event.charactersIgnoringModifiers, !characters.isEmpty, event.keyCode != 51 && event.keyCode != 36 && event.keyCode != 48 {
            controller?.handleCharacters(characters)
            return
        }
        controller?.handleSpecialKey(event.keyCode)
    }
}

@MainActor
final class Renderer: NSObject, MTKViewDelegate, PaperInteractionController {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let surfacePipeline: MTLRenderPipelineState
    private let proceduralPipeline: MTLRenderPipelineState
    private let textPipeline: MTLRenderPipelineState
    private let samplerState: MTLSamplerState
    private weak var view: MTKView?

    private var state = SceneState()
    private var layout = PaperScene.makeLayout(viewport: .init(width: 1180, height: 820))
    private var surfaceBuffer: MTLBuffer?
    private var textVerticesBuffer: MTLBuffer?
    private var textTextureCache: [String: MTLTexture] = [:]
    private var lastTime: CFTimeInterval = CACurrentMediaTime()
    var tuning = RenderTuning()

    init(view: MTKView, device: MTLDevice) {
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Unable to create command queue.")
        }
        self.commandQueue = commandQueue

        let shaderURL = Bundle.module.url(forResource: "ShaderSource", withExtension: "metal")!
        let shaderSource = try! String(contentsOf: shaderURL)
        let library = try! device.makeLibrary(source: shaderSource, options: nil)

        let surfaceDescriptor = MTLRenderPipelineDescriptor()
        surfaceDescriptor.vertexFunction = library.makeFunction(name: "surfaceVertex")
        surfaceDescriptor.fragmentFunction = library.makeFunction(name: "surfaceFragment")
        surfaceDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat

        let proceduralDescriptor = MTLRenderPipelineDescriptor()
        proceduralDescriptor.vertexFunction = library.makeFunction(name: "proceduralVertex")
        proceduralDescriptor.fragmentFunction = library.makeFunction(name: "proceduralFragment")
        proceduralDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat

        let textDescriptor = MTLRenderPipelineDescriptor()
        textDescriptor.vertexFunction = library.makeFunction(name: "textVertex")
        textDescriptor.fragmentFunction = library.makeFunction(name: "textFragment")
        textDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        textDescriptor.colorAttachments[0].isBlendingEnabled = true
        textDescriptor.colorAttachments[0].rgbBlendOperation = .add
        textDescriptor.colorAttachments[0].alphaBlendOperation = .add
        textDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        textDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        textDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        textDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        self.surfacePipeline = try! device.makeRenderPipelineState(descriptor: surfaceDescriptor)
        self.proceduralPipeline = try! device.makeRenderPipelineState(descriptor: proceduralDescriptor)
        self.textPipeline = try! device.makeRenderPipelineState(descriptor: textDescriptor)

        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.sAddressMode = .clampToEdge
        samplerDescriptor.tAddressMode = .clampToEdge
        self.samplerState = device.makeSamplerState(descriptor: samplerDescriptor)!
        self.view = view
        super.init()

        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.13, green: 0.13, blue: 0.15, alpha: 1)
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = false
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        layout = PaperScene.makeLayout(viewport: view.bounds.size)
    }

    func draw(in view: MTKView) {
        guard let descriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            return
        }

        let now = CACurrentMediaTime()
        let delta = Float(min(now - lastTime, 1.0 / 20.0))
        lastTime = now
        state.submitPulse = max(0, state.submitPulse - delta * 1.6)

        let viewportSize = view.bounds.size
        layout = PaperScene.makeLayout(viewport: viewportSize)
        var viewUniforms = ViewUniforms(viewport: SIMD2(Float(viewportSize.width), Float(viewportSize.height)))
        var tuningUniform = RenderTuningUniform(tuning)

        encoder.setRenderPipelineState(proceduralPipeline)
        encoder.setVertexBytes(&viewUniforms, length: MemoryLayout<ViewUniforms>.stride, index: 0)
        encoder.setFragmentBytes(&viewUniforms, length: MemoryLayout<ViewUniforms>.stride, index: 0)
        encoder.setFragmentBytes(&tuningUniform, length: MemoryLayout<RenderTuningUniform>.stride, index: 2)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        let ink = PaperScene.buildInk(layout: layout, state: state)
        encoder.setRenderPipelineState(textPipeline)
        encoder.setVertexBytes(&viewUniforms, length: MemoryLayout<ViewUniforms>.stride, index: 0)
        encoder.setFragmentSamplerState(samplerState, index: 0)

        for stamp in ink {
            drawEngravedText(stamp, encoder: encoder)
        }

        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    func handleClick(at point: CGPoint) {
        let scenePoint = convert(point)
        switch PaperScene.hitTest(scenePoint, layout: layout) {
        case .field(let field):
            state.focus = field
            state.buttonPressed = false
        case .checkbox:
            state.rememberMe.toggle()
        case .button:
            state.buttonPressed = true
            state.submitPulse = 1
        case .none:
            state.buttonPressed = false
        }
    }

    func handleHover(at point: CGPoint) {
        state.hoverTarget = PaperScene.hitTest(convert(point), layout: layout)
    }

    func handleCharacters(_ characters: String) {
        state.buttonPressed = false
        let filteredScalars = characters.unicodeScalars.filter { !CharacterSet.controlCharacters.contains($0) }
        let filtered = String(String.UnicodeScalarView(filteredScalars))
        guard !filtered.isEmpty else { return }
        state.editValue(for: state.focus) { current in
            String((current + filtered).prefix(32))
        }
    }

    func handleSpecialKey(_ keyCode: UInt16) {
        switch keyCode {
        case 48:
            if let nextIndex = FieldFocus.allCases.firstIndex(of: state.focus).map({ ($0 + 1) % FieldFocus.allCases.count }) {
                state.focus = FieldFocus.allCases[nextIndex]
            }
        case 51:
            state.editValue(for: state.focus) { current in
                String(current.dropLast())
            }
        case 36:
            state.buttonPressed = true
            state.submitPulse = 1
        default:
            break
        }
    }

    private func convert(_ point: CGPoint) -> CGPoint {
        CGPoint(x: point.x, y: layout.viewport.height - point.y)
    }

    private func makeSurfaceUniform(from surface: SurfaceSpec) -> SurfaceUniform {
        var uniform = SurfaceUniform(
            rect: SIMD4(Float(surface.rect.minX), Float(surface.rect.minY), Float(surface.rect.width), Float(surface.rect.height)),
            color: surface.color,
            params: SIMD4(surface.radius, surface.z, surface.shadowDepth + state.submitPulse * (surface.rect == layout.buttonRect.offsetBy(dx: 0, dy: state.buttonPressed ? 5 : 0) ? 0.5 : 0), surface.inkMask),
            cutoutCount: UInt32(surface.cutouts.count),
            shadowCount: UInt32(surface.shadowSources.count),
            isOuterRect: surface.isOuterRect ? 1 : 0,
            isPressed: surface.isPressed ? 1 : 0
        )

        for (index, cutout) in surface.cutouts.prefix(maxCutouts).enumerated() {
            let packedRect = SIMD4(Float(cutout.rect.minX), Float(cutout.rect.minY), Float(cutout.rect.width), Float(cutout.rect.height))
            switch index {
            case 0: uniform.cutoutRect0 = packedRect; uniform.cutoutRadiiA.x = cutout.radius
            case 1: uniform.cutoutRect1 = packedRect; uniform.cutoutRadiiA.y = cutout.radius
            case 2: uniform.cutoutRect2 = packedRect; uniform.cutoutRadiiA.z = cutout.radius
            case 3: uniform.cutoutRect3 = packedRect; uniform.cutoutRadiiA.w = cutout.radius
            case 4: uniform.cutoutRect4 = packedRect; uniform.cutoutRadiiB.x = cutout.radius
            case 5: uniform.cutoutRect5 = packedRect; uniform.cutoutRadiiB.y = cutout.radius
            case 6: uniform.cutoutRect6 = packedRect; uniform.cutoutRadiiB.z = cutout.radius
            case 7: uniform.cutoutRect7 = packedRect; uniform.cutoutRadiiB.w = cutout.radius
            default: break
            }
        }

        for (index, source) in surface.shadowSources.prefix(maxCutouts).enumerated() {
            let packedRect = SIMD4(Float(source.rect.minX), Float(source.rect.minY), Float(source.rect.width), Float(source.rect.height))
            switch index {
            case 0: uniform.shadowRect0 = packedRect; uniform.shadowRadiiA.x = source.radius
            case 1: uniform.shadowRect1 = packedRect; uniform.shadowRadiiA.y = source.radius
            case 2: uniform.shadowRect2 = packedRect; uniform.shadowRadiiA.z = source.radius
            case 3: uniform.shadowRect3 = packedRect; uniform.shadowRadiiA.w = source.radius
            case 4: uniform.shadowRect4 = packedRect; uniform.shadowRadiiB.x = source.radius
            case 5: uniform.shadowRect5 = packedRect; uniform.shadowRadiiB.y = source.radius
            case 6: uniform.shadowRect6 = packedRect; uniform.shadowRadiiB.z = source.radius
            case 7: uniform.shadowRect7 = packedRect; uniform.shadowRadiiB.w = source.radius
            default: break
            }
        }

        return uniform
    }

    private func makeTextQuad(_ rect: CGRect) -> [TextVertex] {
        let x0 = Float(rect.minX)
        let y0 = Float(rect.minY)
        let x1 = Float(rect.maxX)
        let y1 = Float(rect.maxY)
        return [
            TextVertex(position: SIMD2(x0, y0), uv: SIMD2(0, 1)),
            TextVertex(position: SIMD2(x1, y0), uv: SIMD2(1, 1)),
            TextVertex(position: SIMD2(x0, y1), uv: SIMD2(0, 0)),
            TextVertex(position: SIMD2(x1, y1), uv: SIMD2(1, 0)),
        ]
    }

    private func drawEngravedText(_ stamp: InkStamp, encoder: MTLRenderCommandEncoder) {
        let highlight = stamp.with(
            id: "\(stamp.id)_engrave_highlight",
            rect: stamp.rect.offsetBy(dx: -0.9, dy: 0.9),
            color: NSColor(calibratedRed: 0.86, green: 0.72, blue: 0.98, alpha: 0.28)
        )
        let compression = stamp.with(
            id: "\(stamp.id)_engrave_compression",
            rect: stamp.rect.offsetBy(dx: 1.1, dy: -1.1),
            color: NSColor(calibratedRed: 0.06, green: 0.04, blue: 0.08, alpha: 0.34)
        )
        let pressed = stamp.with(
            id: "\(stamp.id)_engrave_pressed",
            color: NSColor(calibratedRed: 0.10, green: 0.08, blue: 0.13, alpha: 0.66)
        )

        drawTextStamp(highlight, encoder: encoder)
        drawTextStamp(compression, encoder: encoder)
        drawTextStamp(pressed, encoder: encoder)
    }

    private func drawTextStamp(_ stamp: InkStamp, encoder: MTLRenderCommandEncoder) {
        guard let texture = makeTextTexture(for: stamp) else { return }
        var vertices = makeTextQuad(stamp.rect)
        textVerticesBuffer = device.makeBuffer(bytes: &vertices, length: MemoryLayout<TextVertex>.stride * vertices.count)
        encoder.setVertexBuffer(textVerticesBuffer, offset: 0, index: 1)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: vertices.count)
    }

    private func makeTextTexture(for stamp: InkStamp) -> MTLTexture? {
        let colorKey = stamp.color.usingColorSpace(.deviceRGB).map {
            String(format: "%.3f,%.3f,%.3f,%.3f", $0.redComponent, $0.greenComponent, $0.blueComponent, $0.alphaComponent)
        } ?? "unknown"
        let key = "\(stamp.id)|\(stamp.text)|\(stamp.font.fontName)|\(stamp.font.pointSize)|\(colorKey)|\(Int(stamp.rect.width))x\(Int(stamp.rect.height))"
        if let cached = textTextureCache[key] {
            return cached
        }

        let width = max(2, Int(stamp.rect.width.rounded(.up)))
        let height = max(2, Int(stamp.rect.height.rounded(.up)))
        let bytesPerRow = width * 4
        var pixels = Array(repeating: UInt8(0), count: width * height * 4)

        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.setFillColor(NSColor.clear.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setShouldAntialias(true)

        let paragraph = NSMutableParagraphStyle()
        switch stamp.alignment {
        case .center: paragraph.alignment = .center
        case .right: paragraph.alignment = .right
        default: paragraph.alignment = .left
        }

        let attributes: [NSAttributedString.Key: Any] = [
            .font: stamp.font,
            .foregroundColor: stamp.color,
            .paragraphStyle: paragraph,
        ]
        let attributedString = NSAttributedString(string: stamp.text, attributes: attributes)
        let framesetter = CTFramesetterCreateWithAttributedString(attributedString)
        let path = CGPath(rect: CGRect(x: 0, y: 0, width: width, height: height), transform: nil)
        let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: attributedString.length), path, nil)
        CTFrameDraw(frame, context)

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: pixels, bytesPerRow: bytesPerRow)
        textTextureCache[key] = texture
        return texture
    }
}
