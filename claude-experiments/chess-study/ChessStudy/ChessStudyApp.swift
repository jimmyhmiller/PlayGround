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

        // Check command line args first (for macOS CLI usage)
        let args = CommandLine.arguments
        if args.count > 1 {
            return args[1]
        }

        // Check bundle first (for iOS and macOS app builds)
        if let bundlePath = Bundle.main.resourcePath {
            let bundleData = bundlePath + "/data"
            if fm.fileExists(atPath: bundleData + "/courses.json") {
                return bundleData
            }
        }

        // Try common locations relative to executable (for macOS development)
        let candidates = [
            "./data",
            "../data",
            "../../data",
            "../../../data",
            Bundle.main.bundlePath + "/Contents/Resources/data",
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
