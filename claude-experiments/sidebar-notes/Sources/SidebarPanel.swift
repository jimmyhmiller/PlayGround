import AppKit

final class SidebarPanel: NSPanel {
    var slideDuration: TimeInterval = 0.24

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    // NSWindow.animator().setFrame uses this — NOT NSAnimationContext.duration.
    // Returning a constant keeps the slide consistent regardless of distance / source position.
    override func animationResizeTime(_ newFrame: NSRect) -> TimeInterval {
        slideDuration
    }
}
