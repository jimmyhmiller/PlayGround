//
//  main.swift
//  CCSeva
//
//  A native macOS Swift port of CCSeva - Claude Code usage tracking
//  Original: https://github.com/Iamshankhadeep/ccseva
//

import Cocoa
import SwiftUI


// Application delegate class
class AppDelegate: NSObject, NSApplicationDelegate {
    
    private var statusBarItem: NSStatusItem!
    private var popover: NSPopover!
    private var usageService = CCUsageService.shared
    
    func applicationDidFinishLaunching(_ aNotification: Notification) {
        print("🚀 CCSeva starting up...")
        
        // Create the status bar item first
        statusBarItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        
        guard let button = statusBarItem.button else {
            print("❌ Failed to create status bar button")
            return
        }
        
        // Configure the button
        button.title = "0%"
        button.action = #selector(togglePopover(_:))
        button.target = self
        
        print("✅ Status bar item created")
        
        // Create the SwiftUI view
        let contentView = ContentView()
        
        // Create the popover
        popover = NSPopover()
        popover.contentSize = NSSize(width: 600, height: 600)
        popover.behavior = .transient
        popover.contentViewController = NSHostingController(rootView: contentView)
        
        print("✅ Popover created")
        
        // Start usage monitoring
        usageService.startMonitoring()
        
        // Set up observers for usage data updates
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(updateMenuBarDisplay),
            name: .usageDataUpdated,
            object: nil
        )
        
        // Update display once initially
        updateMenuBarDisplay()
        
        // Keep the app running as menu bar only app
        NSApp.setActivationPolicy(.accessory)
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
            popover.contentViewController?.view.window?.becomeKey()
        }
    }
    
    
    @objc func updateMenuBarDisplay() {
        DispatchQueue.main.async {
            guard let button = self.statusBarItem.button else { return }
            
            if self.usageService.isLoading {
                button.title = "..."
            } else if let stats = self.usageService.currentStats {
                button.title = String(format: "%.0f%%", stats.percentageUsed)
            } else if self.usageService.lastError != nil {
                button.title = "Error"
            } else {
                button.title = "0%"
            }
        }
    }
    
    
    func applicationWillTerminate(_ aNotification: Notification) {
        print("👋 CCSeva shutting down...")
        usageService.stopMonitoring()
    }
    
    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        return true
    }
}

// Main entry point
print("🎯 Starting CCSeva...")

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate

// This is crucial for menu bar apps
app.setActivationPolicy(.accessory)

print("🔄 Running app...")
app.run()