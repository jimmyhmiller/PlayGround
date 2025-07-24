import Cocoa

class AppDelegate: NSObject, NSApplicationDelegate {
    var mainWindow: NSWindow!
    var mainViewController: MainViewController!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMainWindow()
        mainWindow.makeKeyAndOrderFront(self)
        NSApp.activate(ignoringOtherApps: true)
        print("Window should be visible now")
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
    
    private func setupMainWindow() {
        let screenRect = NSScreen.main?.frame ?? NSRect(x: 0, y: 0, width: 1200, height: 800)
        let windowRect = NSRect(
            x: screenRect.width / 2 - 600,
            y: screenRect.height / 2 - 400,
            width: 1200,
            height: 800
        )
        
        mainWindow = NSWindow(
            contentRect: windowRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        
        mainWindow.title = "LightTable Clone"
        mainWindow.level = .normal
        mainWindow.backgroundColor = NSColor(red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0)
        
        mainViewController = MainViewController()
        mainWindow.contentViewController = mainViewController
        
        print("About to show window")
        mainWindow.makeKeyAndOrderFront(self)
        print("Window frame: \(mainWindow.frame)")
        print("Window is visible: \(mainWindow.isVisible)")
    }
}