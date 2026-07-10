import SwiftUI
import SystemExtensions
import AVFoundation
import os.log

// Bundle identifier of the embedded camera extension. Must match the
// extension target's PRODUCT_BUNDLE_IDENTIFIER in project.yml.
private let kExtensionBundleID = "com.jimmyhmiller.YoloCamFlip.Extension"

private let log = Logger(subsystem: "com.jimmyhmiller.YoloCamFlip", category: "app")

@main
struct YoloCamFlipApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("YoloCam Flip", id: "main") {
            ContentView(manager: appDelegate.manager)
                .frame(minWidth: 460, minHeight: 300)
        }
        .windowResizability(.contentSize)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    let manager = ExtensionManager()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        manager.activate()
    }
}

struct ContentView: View {
    @ObservedObject var manager: ExtensionManager

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("YoloCam Flip")
                .font(.largeTitle.bold())
            Text("Installs a virtual camera that mirrors your physical camera rotated 180°, so an upside-down mount looks upright in every app.")
                .foregroundStyle(.secondary)

            Divider()

            HStack(spacing: 12) {
                Button("Install / Activate") { manager.activate() }
                    .buttonStyle(.borderedProminent)
                Button("Uninstall") { manager.deactivate() }
            }

            GroupBox("Status") {
                ScrollView {
                    Text(manager.statusLog.isEmpty ? "Idle." : manager.statusLog)
                        .font(.system(.callout, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 120)
            }

            Text("After activating, approve the extension in System Settings → General → Login Items & Extensions → Camera Extensions if macOS prompts you.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(20)
    }
}

final class ExtensionManager: NSObject, ObservableObject, OSSystemExtensionRequestDelegate {
    @Published var statusLog: String = ""

    private func append(_ line: String) {
        log.log("\(line, privacy: .public)")
        NSLog("YoloCamFlip: %@", line)
        // Also append to a plain file in the shared app-group container so the
        // launch/activation flow is observable from outside the sandbox.
        let entry = "\(Date()): \(line)\n"
        if let data = entry.data(using: .utf8),
           let container = FileManager.default.containerURL(
                forSecurityApplicationGroupIdentifier: "7J8U597P7P.com.jimmyhmiller.YoloCamFlip") {
            let url = container.appendingPathComponent("yolocamflip.log")
            if let fh = try? FileHandle(forWritingTo: url) {
                fh.seekToEndOfFile(); fh.write(data); try? fh.close()
            } else {
                try? data.write(to: url)
            }
        }
        DispatchQueue.main.async {
            self.statusLog += (self.statusLog.isEmpty ? "" : "\n") + line
        }
    }

    private let grabber = FrameGrabber()

    private func containerURL(_ name: String) -> URL? {
        FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "7J8U597P7P.com.jimmyhmiller.YoloCamFlip")?
            .appendingPathComponent(name)
    }

    /// Capture one frame from the virtual (flipped) camera and one from the raw
    /// Y-CAM so the 180° relationship can be verified objectively.
    func runCaptureTest() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { self.append("❌ Camera access denied to the app; cannot run capture test."); return }
            DispatchQueue.main.async {
                self.append("Capturing frame from 'YoloCam Flipped'…")
                self.grabber.grab(deviceLocalizedName: "YoloCam Flipped") { cg in
                    if let cg, let url = self.containerURL("test_flipped.png") {
                        FrameGrabber.savePNG(cg, to: url)
                        self.append("Saved flipped frame (\(cg.width)x\(cg.height)).")
                    } else {
                        self.append("❌ No frame from 'YoloCam Flipped'.")
                    }
                    // Then grab the raw camera for comparison.
                    self.append("Capturing frame from raw 'Y-CAM-26200068'…")
                    let raw = FrameGrabber()
                    raw.grab(deviceLocalizedName: "Y-CAM-26200068") { rcg in
                        if let rcg, let url = self.containerURL("test_raw.png") {
                            FrameGrabber.savePNG(rcg, to: url)
                            self.append("Saved raw frame (\(rcg.width)x\(rcg.height)). Capture test complete.")
                        } else {
                            self.append("❌ No frame from raw camera.")
                        }
                        _ = raw // retain through callback
                    }
                }
            }
        }
    }

    func activate() {
        append("Bundle.main path: \(Bundle.main.bundlePath)")
        let sysextDir = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Library/SystemExtensions")
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: sysextDir.path)) ?? ["<none>"]
        append("SystemExtensions dir contents: \(contents.joined(separator: ", "))")
        append("Requesting activation of \(kExtensionBundleID)…")
        let req = OSSystemExtensionRequest.activationRequest(
            forExtensionWithIdentifier: kExtensionBundleID,
            queue: .main)
        req.delegate = self
        OSSystemExtensionManager.shared.submitRequest(req)
    }

    func deactivate() {
        append("Requesting deactivation of \(kExtensionBundleID)…")
        let req = OSSystemExtensionRequest.deactivationRequest(
            forExtensionWithIdentifier: kExtensionBundleID,
            queue: .main)
        req.delegate = self
        OSSystemExtensionManager.shared.submitRequest(req)
    }

    // MARK: OSSystemExtensionRequestDelegate

    func request(_ request: OSSystemExtensionRequest,
                 actionForReplacingExtension existing: OSSystemExtensionProperties,
                 withExtension ext: OSSystemExtensionProperties) -> OSSystemExtensionRequest.ReplacementAction {
        append("Replacing existing v\(existing.bundleShortVersion) with v\(ext.bundleShortVersion).")
        return .replace
    }

    func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        append("⚠️ Needs your approval — open System Settings → General → Login Items & Extensions → Camera Extensions and enable YoloCamFlip.")
    }

    func request(_ request: OSSystemExtensionRequest,
                 didFinishWithResult result: OSSystemExtensionRequest.Result) {
        switch result {
        case .completed:
            append("✅ Done. The 'YoloCam Flipped' camera should now be available in your apps.")
            runCaptureTest()
        case .willCompleteAfterReboot:
            append("Will complete after reboot.")
        @unknown default:
            append("Finished with result: \(result.rawValue)")
        }
    }

    func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
        append("❌ Failed: \(error.localizedDescription)  [\((error as NSError).domain) \((error as NSError).code)]")
    }
}
