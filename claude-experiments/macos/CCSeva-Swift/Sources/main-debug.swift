//
//  main-debug.swift
//  CCSeva - Debug version with window
//

import Cocoa
import SwiftUI

// Debug application delegate that shows both window and menu bar
class DebugAppDelegate: NSObject, NSApplicationDelegate {
    
    private var statusBarItem: NSStatusItem!
    private var popover: NSPopover!
    private var window: NSWindow!
    
    func applicationDidFinishLaunching(_ aNotification: Notification) {
        print("🚀 CCSeva DEBUG starting up...")
        
        // Create a regular window first for debugging
        let contentView = ContentView()
        
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 600),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.center()
        window.setFrameAutosaveName("CCSeva Debug Window")
        window.contentView = NSHostingController(rootView: contentView).view
        window.title = "CCSeva Debug"
        window.makeKeyAndOrderFront(nil)
        
        // Also create the status bar item
        statusBarItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        
        if let button = statusBarItem.button {
            button.title = "75%"
            button.action = #selector(togglePopover(_:))
            button.target = self
            print("✅ Status bar item created: \(button.title)")
        } else {
            print("❌ Failed to create status bar button")
        }
        
        // Create the popover
        popover = NSPopover()
        popover.contentSize = NSSize(width: 600, height: 600)
        popover.behavior = .transient
        popover.contentViewController = NSHostingController(rootView: contentView)
        
        print("✅ Debug setup complete - window and menu bar item should be visible")
    }
    
    @objc func togglePopover(_ sender: AnyObject?) {
        print("🖱️ Menu bar item clicked")
        
        guard let button = statusBarItem.button else { return }
        
        if popover.isShown {
            print("📴 Closing popover")
            popover.performClose(sender)
        } else {
            print("📱 Opening popover")
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: NSRectEdge.minY)
        }
    }
    
    func applicationWillTerminate(_ aNotification: Notification) {
        print("👋 CCSeva DEBUG shutting down...")
    }
    
    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        return true
    }
}

// Main entry point for debug version
print("🎯 Starting CCSeva DEBUG...")

let app = NSApplication.shared
let delegate = DebugAppDelegate()
app.delegate = delegate

// Show in dock for debugging
app.setActivationPolicy(.regular)

print("🔄 Running debug app...")
app.run()