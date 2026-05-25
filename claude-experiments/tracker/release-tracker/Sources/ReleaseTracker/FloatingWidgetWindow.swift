import SwiftUI
import AppKit
import Shared

/// A frameless, always-on-top window that visually mimics a WidgetKit
/// widget but runs inside the host process. Used because a real WidgetKit
/// extension requires Apple Developer Portal action (provisioning profile
/// covering the widget's bundle id) which we can't automate.
@MainActor
final class FloatingWidgetController {
    private var window: NSWindow?
    private let store = ChecklistStore()
    private let positionKey = "FloatingWidgetFrameOrigin"

    func show() {
        if let window {
            window.makeKeyAndOrderFront(nil)
            return
        }

        let size = NSSize(width: 280, height: 180)
        let origin = restoredOrigin(forSize: size)

        let newWindow = DraggableWindow(
            contentRect: NSRect(origin: origin, size: size),
            styleMask: [.borderless, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        newWindow.isOpaque = false
        newWindow.backgroundColor = .clear
        newWindow.hasShadow = true
        // Live on the desktop — same window level as desktop icons. This
        // sits above the wallpaper but below every normal app window, so
        // it only becomes visible when you Show Desktop (F11 / four-finger
        // spread) or close the apps in front of it. That's how real
        // WidgetKit desktop widgets behave.
        newWindow.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.desktopIconWindow)))
        newWindow.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]
        newWindow.isMovableByWindowBackground = true
        newWindow.isMovable = true
        newWindow.isReleasedWhenClosed = false
        newWindow.titleVisibility = .hidden
        newWindow.titlebarAppearsTransparent = true
        newWindow.standardWindowButton(.closeButton)?.isHidden = true
        newWindow.standardWindowButton(.miniaturizeButton)?.isHidden = true
        newWindow.standardWindowButton(.zoomButton)?.isHidden = true

        let hosting = NSHostingController(rootView: FloatingWidgetView())
        hosting.view.frame = NSRect(origin: .zero, size: size)
        newWindow.contentViewController = hosting

        newWindow.delegate = WindowSaver.shared
        WindowSaver.shared.positionKey = positionKey

        newWindow.orderFront(nil)
        self.window = newWindow
    }

    private func restoredOrigin(forSize size: NSSize) -> NSPoint {
        if let data = UserDefaults.standard.array(forKey: positionKey) as? [Double], data.count == 2 {
            return NSPoint(x: data[0], y: data[1])
        }
        // Default: top-right of the main screen, 24 px from edges
        let screen = NSScreen.main?.visibleFrame ?? .zero
        return NSPoint(
            x: screen.maxX - size.width - 24,
            y: screen.maxY - size.height - 24
        )
    }
}

/// Borderless windows are not normally key-eligible; this subclass opts in
/// so SwiftUI controls inside can receive events.
private final class DraggableWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

/// Saves the window's origin to UserDefaults so position survives quits.
@MainActor
private final class WindowSaver: NSObject, NSWindowDelegate {
    static let shared = WindowSaver()
    var positionKey: String = ""

    func windowDidMove(_ notification: Notification) {
        guard let window = notification.object as? NSWindow else { return }
        let origin = window.frame.origin
        UserDefaults.standard.set([origin.x, origin.y], forKey: positionKey)
    }
}
