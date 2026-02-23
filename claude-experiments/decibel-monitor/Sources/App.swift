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
            Image(systemName: menuBarIcon)
        }
        .menuBarExtraStyle(.window)
    }

    private var menuBarIcon: String {
        if !monitor.isMonitoring {
            return "mic.slash"
        } else if monitor.isAboveThreshold {
            return "exclamationmark.triangle.fill"
        } else {
            return "mic.fill"
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
    }
}
