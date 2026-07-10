import AVFoundation
import CoreImage
import AppKit

/// Grabs a single warmed-up frame from a named capture device and hands back a
/// CGImage. Used to verify the virtual camera's output against the raw camera.
final class FrameGrabber: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private var session: AVCaptureSession?
    private var completion: ((CGImage?) -> Void)?
    private let queue = DispatchQueue(label: "com.jimmyhmiller.YoloCamFlip.grab")
    private var frameCount = 0
    private var finished = false
    private let ciContext = CIContext()

    func grab(deviceLocalizedName name: String, completion: @escaping (CGImage?) -> Void) {
        self.completion = completion
        self.frameCount = 0
        self.finished = false

        let devices = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .builtInWideAngleCamera, .continuityCamera, .deskViewCamera],
            mediaType: .video, position: .unspecified).devices
        guard let device = devices.first(where: { $0.localizedName == name }) else {
            completion(nil); return
        }
        let s = AVCaptureSession()
        s.beginConfiguration()
        if s.canSetSessionPreset(.hd1280x720) { s.sessionPreset = .hd1280x720 }
        guard let input = try? AVCaptureDeviceInput(device: device), s.canAddInput(input) else {
            completion(nil); return
        }
        s.addInput(input)
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: queue)
        guard s.canAddOutput(out) else { completion(nil); return }
        s.addOutput(out)
        s.commitConfiguration()
        s.startRunning()
        self.session = s
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !finished else { return }
        frameCount += 1
        // Skip warmup frames (auto-exposure/first black frames).
        guard frameCount >= 8, let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        finished = true
        let ci = CIImage(cvPixelBuffer: pixel)
        let cg = ciContext.createCGImage(ci, from: ci.extent)
        session?.stopRunning(); session = nil
        DispatchQueue.main.async { self.completion?(cg) }
    }

    static func savePNG(_ cg: CGImage, to url: URL) {
        let rep = NSBitmapImageRep(cgImage: cg)
        if let data = rep.representation(using: .png, properties: [:]) {
            try? data.write(to: url)
        }
    }
}
