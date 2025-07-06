import Cocoa
import SwiftUI
import SwiftFontAtlas

@MainActor
class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    
    func applicationDidFinishLaunching(_ aNotification: Notification) {
        NSApp.setActivationPolicy(.regular)
        
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1000, height: 700),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "SwiftFontAtlas - Real Text Rendering"
        window.center()
        
        // Use the simplified view with high-level API
        let contentView = NSHostingView(rootView: RealTextViewSimplified())
        window.contentView = contentView
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
        
        NSApp.activate(ignoringOtherApps: true)
        
        setupMenu()
    }
    
    private func setupMenu() {
        let mainMenu = NSMenu()
        
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        
        let appMenu = NSMenu()
        appMenuItem.submenu = appMenu
        
        let quitItem = NSMenuItem(title: "Quit RealText", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appMenu.addItem(quitItem)
        
        NSApplication.shared.mainMenu = mainMenu
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}