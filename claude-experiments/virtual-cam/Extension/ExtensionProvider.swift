import Foundation
import CoreMediaIO
import AVFoundation
import Metal
import IOKit.audio
import os.log

// MARK: - Configuration

/// Fallback resolution used ONLY if the source camera's native format can't be
/// read at startup. Normally the virtual camera adopts the source's EXACT native
/// format — no preset, no crop, no scale — and we only flip the pixels 180°.
private let kFallbackWidth: Int32 = 1920
private let kFallbackHeight: Int32 = 1080
private let kFrameRate: Int = 30

/// Name the virtual camera shows under in FaceTime / Zoom / browsers, and the
/// stable identifiers CMIO uses to track the device across launches.
private let kDeviceName = "YoloCam Flipped"
private let kDeviceUID = "com.jimmyhmiller.YoloCamFlip.device"
private let kStreamUID = "com.jimmyhmiller.YoloCamFlip.stream"

/// Substrings we prefer when auto-picking the physical source camera.
private let kPreferredSourceHints = ["y-cam", "ycam", "yolo", "s7"]

private let log = Logger(subsystem: "com.jimmyhmiller.YoloCamFlip", category: "extension")

/// File logging into the shared app-group container so the extension's capture
/// path is observable from outside (unified logging is unreliable here).
func extLog(_ s: String) {
    log.log("\(s, privacy: .public)")
    let line = "\(Date()): \(s)\n"
    guard let data = line.data(using: .utf8),
          let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "7J8U597P7P.com.jimmyhmiller.YoloCamFlip") else { return }
    let url = container.appendingPathComponent("extension.log")
    if let fh = try? FileHandle(forWritingTo: url) { fh.seekToEndOfFile(); fh.write(data); try? fh.close() }
    else { try? data.write(to: url) }
}

// MARK: - Provider

class ExtensionProviderSource: NSObject, CMIOExtensionProviderSource {

    private(set) var provider: CMIOExtensionProvider!
    private var deviceSource: ExtensionDeviceSource!

    init(clientQueue: DispatchQueue?) {
        super.init()
        provider = CMIOExtensionProvider(source: self, clientQueue: clientQueue)
        deviceSource = ExtensionDeviceSource(localizedName: kDeviceName)
        do {
            try provider.addDevice(deviceSource.device)
        } catch {
            log.error("Failed to add device: \(error.localizedDescription, privacy: .public)")
            fatalError("Failed to add virtual camera device: \(error.localizedDescription)")
        }
    }

    func connect(to client: CMIOExtensionClient) throws {}

    func disconnect(from client: CMIOExtensionClient) {}

    var availableProperties: Set<CMIOExtensionProperty> {
        return [.providerManufacturer]
    }

    func providerProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionProviderProperties {
        let p = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) {
            p.manufacturer = "Jimmy Miller"
        }
        return p
    }

    func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {}
}

// MARK: - Device

class ExtensionDeviceSource: NSObject, CMIOExtensionDeviceSource, AVCaptureVideoDataOutputSampleBufferDelegate {

    private(set) var device: CMIOExtensionDevice!
    private var streamSource: ExtensionStreamSource!

    // Capture pipeline.
    private let captureQueue = DispatchQueue(label: "com.jimmyhmiller.YoloCamFlip.capture", qos: .userInitiated)
    private var captureSession: AVCaptureSession?
    private let videoOutput = AVCaptureVideoDataOutput()
    private var pixelBufferPool: CVPixelBufferPool?
    private var destFormatDescription: CMFormatDescription?
    private var bufWidth: Int32 = 0
    private var bufHeight: Int32 = 0

    private let metalFlipper = MetalFlipper()

    private var streaming = false

    init(localizedName: String) {
        super.init()
        let deviceID = UUID()
        device = CMIOExtensionDevice(localizedName: localizedName,
                                     deviceID: deviceID,
                                     legacyDeviceID: kDeviceUID,
                                     source: self)

        // Adopt the source camera's native resolution so the virtual camera is a
        // pure passthrough (flipped). Fall back only if no source is present.
        let (w, h) = Self.nativeSourceDimensions() ?? (kFallbackWidth, kFallbackHeight)
        buildBuffers(width: w, height: h)

        let streamFormat = CMIOExtensionStreamFormat(
            formatDescription: destFormatDescription!,
            maxFrameDuration: CMTime(value: 1, timescale: Int32(kFrameRate)),
            minFrameDuration: CMTime(value: 1, timescale: Int32(kFrameRate)),
            validFrameDurations: nil)

        streamSource = ExtensionStreamSource(localizedName: "\(localizedName).stream",
                                             streamID: UUID(),
                                             streamFormat: streamFormat,
                                             device: self)
        do {
            try device.addStream(streamSource.stream)
        } catch {
            fatalError("Failed to add stream: \(error.localizedDescription)")
        }
    }

    /// (Re)build the pixel-buffer pool and format description for a given frame
    /// size. Called at init and whenever an incoming frame's size differs.
    private func buildBuffers(width: Int32, height: Int32) {
        guard width > 0, height > 0 else { return }
        bufWidth = width; bufHeight = height

        var fd: CMFormatDescription?
        CMVideoFormatDescriptionCreate(allocator: kCFAllocatorDefault,
                                       codecType: kCVPixelFormatType_32BGRA,
                                       width: width, height: height,
                                       extensions: nil, formatDescriptionOut: &fd)
        destFormatDescription = fd

        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: Int(width),
            kCVPixelBufferHeightKey as String: Int(height),
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool)
        pixelBufferPool = pool
    }

    /// Read the preferred source camera's native active-format dimensions.
    private static func nativeSourceDimensions() -> (Int32, Int32)? {
        guard let src = pickSourceDevice() else { return nil }
        let dims = CMVideoFormatDescriptionGetDimensions(src.activeFormat.formatDescription)
        return (dims.width, dims.height)
    }

    var availableProperties: Set<CMIOExtensionProperty> {
        return [.deviceTransportType, .deviceModel]
    }

    func deviceProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionDeviceProperties {
        let p = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) {
            p.transportType = kIOAudioDeviceTransportTypeVirtual
        }
        if properties.contains(.deviceModel) {
            p.model = "YoloCam Flip"
        }
        return p
    }

    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}

    // MARK: Streaming lifecycle (called by the stream source)

    func startStreaming() {
        guard !streaming else { return }
        streaming = true
        extLog("startStreaming: a client began consuming the virtual camera")
        captureQueue.async { [weak self] in
            self?.configureAndStartCapture()
        }
    }

    func stopStreaming() {
        guard streaming else { return }
        streaming = false
        captureQueue.async { [weak self] in
            self?.captureSession?.stopRunning()
            self?.captureSession = nil
        }
    }

    // MARK: Physical camera capture

    private func configureAndStartCapture() {
        let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
        if authStatus == .notDetermined {
            let sem = DispatchSemaphore(value: 0)
            AVCaptureDevice.requestAccess(for: .video) { _ in sem.signal() }
            sem.wait()
        }
        guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
            extLog("Camera access NOT authorized for the extension (status \(AVCaptureDevice.authorizationStatus(for: .video).rawValue)). No frames will be produced.")
            return
        }

        let allCams = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .builtInWideAngleCamera, .continuityCamera],
            mediaType: .video, position: .unspecified).devices.map { $0.localizedName }
        extLog("Visible cameras: \(allCams.joined(separator: " | "))")

        guard let source = Self.pickSourceDevice() else {
            extLog("No physical source camera found to flip. Plug in the YoloCam and restart the consuming app.")
            return
        }
        let nativeDims = CMVideoFormatDescriptionGetDimensions(source.activeFormat.formatDescription)
        extLog("Using source camera: \(source.localizedName) native \(nativeDims.width)x\(nativeDims.height)")

        let session = AVCaptureSession()
        session.beginConfiguration()

        // Do NOT set a preset. macOS defaults to .high, i.e. the device's native
        // full-FOV format — we take exactly what the camera delivers and only
        // flip it. (Forcing .hd1280x720 made the Y-CAM deliver a cropped/zoomed
        // image, because its 720p mode is a center crop, not a downscale.)

        do {
            let input = try AVCaptureDeviceInput(device: source)
            guard session.canAddInput(input) else {
                log.error("Cannot add camera input to capture session.")
                return
            }
            session.addInput(input)
        } catch {
            log.error("Failed to open camera input: \(error.localizedDescription, privacy: .public)")
            return
        }

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: captureQueue)
        guard session.canAddOutput(videoOutput) else {
            log.error("Cannot add video output to capture session.")
            return
        }
        session.addOutput(videoOutput)

        session.commitConfiguration()
        session.startRunning()
        captureSession = session
        extLog("Capture session started (preset \(session.sessionPreset.rawValue)). Waiting for frames…")
    }

    /// Choose a real camera to flip: never our own virtual device, prefer one
    /// whose name matches the YoloCam hints, then any external (USB) camera,
    /// then the built-in camera.
    private static func pickSourceDevice() -> AVCaptureDevice? {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .builtInWideAngleCamera, .continuityCamera],
            mediaType: .video,
            position: .unspecified)
        let candidates = discovery.devices.filter { dev in
            dev.localizedName != kDeviceName          // never capture ourselves
        }
        if let preferred = candidates.first(where: { dev in
            let name = dev.localizedName.lowercased()
            return kPreferredSourceHints.contains { name.contains($0) }
        }) {
            return preferred
        }
        if let external = candidates.first(where: { $0.deviceType == .external }) {
            return external
        }
        return candidates.first
    }

    // MARK: Per-frame rotation + delivery

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard streaming,
              let srcPixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Pure passthrough: match the virtual camera's buffers to the source
        // frame's EXACT size. No scaling, no cropping — only a 180° flip.
        let inW = Int32(CVPixelBufferGetWidth(srcPixel))
        let inH = Int32(CVPixelBufferGetHeight(srcPixel))
        if inW != bufWidth || inH != bufHeight {
            buildBuffers(width: inW, height: inH)
        }
        guard let pool = pixelBufferPool, let formatDescription = destFormatDescription else { return }

        var destPixelOpt: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &destPixelOpt) == kCVReturnSuccess,
              let destPixel = destPixelOpt else {
            log.error("Failed to allocate destination pixel buffer from pool.")
            return
        }

        guard let flipper = metalFlipper else {
            log.error("Metal flipper unavailable; cannot process frame.")
            return
        }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        // GPU-only 180° flip. The CPU touches no pixels — we just publish the
        // result once the GPU signals completion. `sampleBuffer` is captured to
        // keep the source pixels alive until the GPU has read them.
        flipper.flip(src: srcPixel, dst: destPixel) { [weak self] ok in
            guard let self, ok else { return }
            withExtendedLifetime(sampleBuffer) {}

            var timing = CMSampleTimingInfo(
                duration: CMTime(value: 1, timescale: Int32(kFrameRate)),
                presentationTimeStamp: pts,
                decodeTimeStamp: .invalid)
            var outSample: CMSampleBuffer?
            let status = CMSampleBufferCreateReadyWithImageBuffer(
                allocator: kCFAllocatorDefault,
                imageBuffer: destPixel,
                formatDescription: formatDescription,
                sampleTiming: &timing,
                sampleBufferOut: &outSample)
            guard status == noErr, let out = outSample else {
                log.error("Failed to wrap flipped frame in a sample buffer (status \(status)).")
                return
            }
            let hostNs = UInt64(max(0, pts.seconds) * Double(NSEC_PER_SEC))
            self.streamSource.stream.send(out, discontinuity: [], hostTimeInNanoseconds: hostNs)

            self.framesSent += 1
            if self.framesSent == 1 {
                extLog("First GPU-flipped frame sent (\(CVPixelBufferGetWidth(destPixel))x\(CVPixelBufferGetHeight(destPixel))).")
            }
        }
    }
    private var framesSent = 0
}

// MARK: - Stream

class ExtensionStreamSource: NSObject, CMIOExtensionStreamSource {

    private(set) var stream: CMIOExtensionStream!
    private let streamFormat: CMIOExtensionStreamFormat
    private weak var deviceSource: ExtensionDeviceSource?

    init(localizedName: String,
         streamID: UUID,
         streamFormat: CMIOExtensionStreamFormat,
         device: ExtensionDeviceSource) {
        self.streamFormat = streamFormat
        self.deviceSource = device
        super.init()
        stream = CMIOExtensionStream(localizedName: localizedName,
                                     streamID: streamID,
                                     direction: .source,
                                     clockType: .hostTime,
                                     source: self)
    }

    var formats: [CMIOExtensionStreamFormat] { [streamFormat] }

    var activeFormatIndex: Int = 0 {
        didSet {
            if activeFormatIndex != 0 {
                log.error("Invalid active format index: \(self.activeFormatIndex)")
            }
        }
    }

    var availableProperties: Set<CMIOExtensionProperty> {
        return [.streamActiveFormatIndex, .streamFrameDuration]
    }

    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionStreamProperties {
        let p = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) {
            p.activeFormatIndex = 0
        }
        if properties.contains(.streamFrameDuration) {
            p.frameDuration = CMTime(value: 1, timescale: Int32(kFrameRate))
        }
        return p
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {
        if let idx = streamProperties.activeFormatIndex {
            activeFormatIndex = idx
        }
    }

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool {
        return true
    }

    func startStream() throws {
        deviceSource?.startStreaming()
    }

    func stopStream() throws {
        deviceSource?.stopStreaming()
    }
}
