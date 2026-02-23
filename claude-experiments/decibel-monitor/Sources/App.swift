import SwiftUI
import AppKit

@main
struct DecibelMonitorApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var monitor = AudioMonitor()

    var body: some Scene {
        MenuBarExtra {
            MonitorView(monitor: monitor)
        } label: {
            Image(nsImage: statusCircle)
        }
        .menuBarExtraStyle(.window)
    }

    private var statusCircle: NSImage {
        let color: NSColor
        if !monitor.isMonitoring {
            color = .gray
        } else if monitor.isAboveThreshold {
            color = .red
        } else {
            color = .white
        }

        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size, flipped: false) { rect in
            let circleDiameter: CGFloat = 14
            let circleRect = NSRect(
                x: (rect.width - circleDiameter) / 2,
                y: (rect.height - circleDiameter) / 2,
                width: circleDiameter,
                height: circleDiameter
            )
            color.setFill()
            NSBezierPath(ovalIn: circleRect).fill()
            return true
        }
        image.isTemplate = false
        return image
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
    }
}
