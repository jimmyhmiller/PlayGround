import AppKit
import SwiftUI

final class SidebarController {
    // Visual / behavior knobs.
    private let panelWidth: CGFloat = 340
    private let heightRatio: CGFloat = 0.6            // fraction of screen visibleFrame height
    private let edgeTriggerThickness: CGFloat = 2     // px from edge to trigger
    private let hideHysteresis: CGFloat = 12          // px outside panel before hiding
    private let slideDuration: TimeInterval = 0.24    // owned by SidebarPanel.animationResizeTime
    private let cornerRadius: CGFloat = 16

    private var panel: SidebarPanel!
    private var pollTimer: Timer?
    private var isShown = false
    private var isAnimating = false

    func start() {
        buildPanel()
        startPolling()
    }

    // MARK: - Panel construction

    private func buildPanel() {
        let screenFrame = currentScreenFrame()
        let shownFrame = shownFrame(in: screenFrame)
        let offscreen = offscreenFrame(for: shownFrame, in: screenFrame)

        let p = SidebarPanel(
            contentRect: offscreen,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        p.level = .floating
        p.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        p.isFloatingPanel = true
        p.becomesKeyOnlyIfNeeded = true
        p.hidesOnDeactivate = false
        p.isMovableByWindowBackground = false
        p.hasShadow = true
        p.isOpaque = false
        p.backgroundColor = .clear
        p.titleVisibility = .hidden
        p.titlebarAppearsTransparent = true
        p.slideDuration = slideDuration

        // Rounded container clipped to corners on the left only (right is flush with screen edge).
        let container = NSView(frame: NSRect(origin: .zero, size: offscreen.size))
        container.wantsLayer = true
        container.layer = CALayer()
        container.layer?.cornerRadius = cornerRadius
        container.layer?.maskedCorners = [.layerMinXMinYCorner, .layerMinXMaxYCorner]
        container.layer?.masksToBounds = true
        container.autoresizingMask = [.width, .height]

        let host = NSHostingView(rootView: NotesView())
        host.frame = container.bounds
        host.autoresizingMask = [.width, .height]
        container.addSubview(host)

        p.contentView = container
        p.orderFrontRegardless()
        panel = p
    }

    // MARK: - Polling

    private func startPolling() {
        let t = Timer(timeInterval: 0.04, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(t, forMode: .common)
        pollTimer = t
    }

    private func tick() {
        guard !isAnimating else { return }
        let mouse = NSEvent.mouseLocation
        let screenFrame = screenContaining(point: mouse) ?? currentScreenFrame()
        let shown = shownFrame(in: screenFrame)

        if isShown {
            let panelLeft = panel.frame.minX
            // Hide when mouse leaves the panel area (left, above, or below) with a small grace zone.
            let outsideLeft = mouse.x < panelLeft - hideHysteresis
            let outsideVertically =
                mouse.y < shown.minY - hideHysteresis ||
                mouse.y > shown.maxY + hideHysteresis
            if outsideLeft || outsideVertically {
                hide()
            }
        } else {
            // Trigger zone: thin slab at the right edge, matching panel's vertical range.
            let inTrigger =
                mouse.x >= screenFrame.maxX - edgeTriggerThickness &&
                mouse.y >= shown.minY &&
                mouse.y <= shown.maxY
            if inTrigger {
                show()
            }
        }
    }

    // MARK: - Show / hide

    func show() {
        guard !isShown, !isAnimating else { return }
        let screenFrame = screenContaining(point: NSEvent.mouseLocation) ?? currentScreenFrame()
        let target = shownFrame(in: screenFrame)
        // Park offscreen on the correct screen first to avoid first-show flicker.
        panel.setFrame(offscreenFrame(for: target, in: screenFrame), display: false)
        panel.orderFrontRegardless()

        isShown = true
        isAnimating = true
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = slideDuration
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().setFrame(target, display: true)
        }, completionHandler: { [weak self] in
            self?.panel.invalidateShadow()
            self?.isAnimating = false
        })
    }

    func hide() {
        guard isShown, !isAnimating else { return }
        let screenFrame = screenContaining(point: panel.frame.origin) ?? currentScreenFrame()
        let current = panel.frame
        let target = offscreenFrame(for: current, in: screenFrame)
        isAnimating = true
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = slideDuration
            ctx.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().setFrame(target, display: true)
        }, completionHandler: { [weak self] in
            self?.isShown = false
            self?.isAnimating = false
        })
    }

    // MARK: - Frame helpers

    private func shownFrame(in screenFrame: NSRect) -> NSRect {
        let h = (screenFrame.height * heightRatio).rounded()
        let y = (screenFrame.minY + (screenFrame.height - h) / 2).rounded()
        return NSRect(x: screenFrame.maxX - panelWidth, y: y, width: panelWidth, height: h)
    }

    private func offscreenFrame(for shown: NSRect, in screenFrame: NSRect) -> NSRect {
        NSRect(x: screenFrame.maxX, y: shown.minY, width: shown.width, height: shown.height)
    }

    // MARK: - Screen helpers

    private func currentScreenFrame() -> NSRect {
        (NSScreen.main ?? NSScreen.screens.first!).visibleFrame
    }

    private func screenContaining(point: NSPoint) -> NSRect? {
        for screen in NSScreen.screens where screen.frame.contains(point) {
            return screen.visibleFrame
        }
        return nil
    }
}
