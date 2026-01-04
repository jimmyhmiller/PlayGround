import SwiftUI
#if os(macOS)
import AppKit
#endif

@main
struct ChessStudyApp: App {
    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    #endif

    var body: some Scene {
        WindowGroup {
            ContentView(dataDir: findDataDir())
        }
    }

    func findDataDir() -> String {
        let fm = FileManager.default

        // Check command line args first
        let args = CommandLine.arguments
        if args.count > 1 {
            return args[1]
        }

        // Try common locations relative to executable
        let candidates = [
            "./data",
            "../data",
            "../../data",
            "../../../data",
            // For running from Xcode or app bundle
            Bundle.main.bundlePath + "/Contents/Resources/data",
            Bundle.main.resourcePath.map { $0 + "/data" } ?? "",
            // Absolute path for development
            fm.currentDirectoryPath + "/data"
        ]

        for candidate in candidates {
            if fm.fileExists(atPath: candidate + "/courses.json") {
                return candidate
            }
        }

        // Default - will show error in UI
        return "./data"
    }
}

#if os(macOS)
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}
#endif
