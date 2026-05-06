import AppKit
import MetalKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    private var renderer: Renderer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is unavailable on this Mac.")
        }

        let frame = NSRect(x: 0, y: 0, width: 1480, height: 820)
        let view = PaperCutView(frame: NSRect(x: 0, y: 0, width: 1180, height: 820), device: device)
        let renderer = Renderer(view: view, device: device)
        view.controller = renderer
        view.delegate = renderer
        let tuningPanel = TuningPanel(renderer: renderer)

        let splitView = NSSplitView(frame: frame)
        splitView.isVertical = true
        splitView.dividerStyle = .thin
        splitView.addArrangedSubview(view)
        splitView.addArrangedSubview(tuningPanel)
        view.widthAnchor.constraint(greaterThanOrEqualToConstant: 900).isActive = true
        tuningPanel.widthAnchor.constraint(equalToConstant: 280).isActive = true

        let window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Paper-Cut Create Account"
        window.setFrame(frame, display: true)
        window.minSize = NSSize(width: 1180, height: 700)
        window.contentView = splitView
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)

        self.window = window
        self.renderer = renderer
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
