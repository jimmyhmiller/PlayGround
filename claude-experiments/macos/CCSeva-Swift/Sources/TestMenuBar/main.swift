//
//  test-menubar.swift
//  Minimal menu bar test
//

import Cocoa

class MinimalAppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("🚀 Creating status bar item...")
        
        // Create status item
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        
        if let button = statusItem.button {
            button.title = "TEST"
            button.action = #selector(statusItemClicked)
            button.target = self
            print("✅ Status item created with title: '\(button.title)'")
        } else {
            print("❌ Failed to create status item button")
        }
        
        // Force refresh status bar
        NSStatusBar.system.removeStatusItem(statusItem)
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        
        if let button = statusItem.button {
            button.title = "🔴 TEST"
            button.action = #selector(statusItemClicked)
            button.target = self
            print("✅ Recreated status item with title: '\(button.title)'")
        }
    }
    
    @objc func statusItemClicked() {
        print("🖱️ Status item clicked!")
        let alert = NSAlert()
        alert.messageText = "Menu Bar Working!"
        alert.informativeText = "The menu bar item is working correctly."
        alert.runModal()
    }
}

// Main
print("🎯 Starting minimal menu bar test...")

let app = NSApplication.shared
let delegate = MinimalAppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)  // Show in dock for testing

print("🔄 App running...")
app.run()