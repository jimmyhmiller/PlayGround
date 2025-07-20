import AppKit
import Foundation

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow?
    var mainWindowController: MainWindowController?
    
    func applicationDidFinishLaunching(_ aNotification: Notification) {
        NSApp.setActivationPolicy(.regular)
        
        mainWindowController = MainWindowController()
        mainWindowController?.showWindow(nil)
        
        NSApp.activate(ignoringOtherApps: true)
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}