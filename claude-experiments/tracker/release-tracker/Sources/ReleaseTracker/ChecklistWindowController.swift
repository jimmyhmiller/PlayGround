import SwiftUI
import AppKit

/// Holds the checklist NSWindow so we can open/close it without losing
/// state. Used by the floating widget's expand button and the menu.
@MainActor
final class ChecklistWindowController {
    static let shared = ChecklistWindowController()
    private var window: NSWindow?

    func show() {
        if let window {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let hosting = NSHostingController(rootView: ChecklistView())
        let w = NSWindow(contentViewController: hosting)
        w.title = "Ease Release"
        w.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        w.setContentSize(NSSize(width: 560, height: 720))
        w.minSize = NSSize(width: 480, height: 600)
        w.isReleasedWhenClosed = false
        w.center()
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = w
    }
}
