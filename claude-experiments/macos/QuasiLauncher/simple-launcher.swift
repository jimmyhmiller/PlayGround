#!/usr/bin/env swift

import SwiftUI
import AppKit
import Carbon
import IOKit.hid

class SimpleAppDelegate: NSObject, NSApplicationDelegate {
    var statusWindow: NSWindow?
    var overlayWindow: NSWindow?
    var statusLabel: NSTextField?
    var commandField: NSTextField?
    var eventTap: CFMachPort?
    var checkTimer: Timer?
    var isEscapePressed = false
    var showingOverlay = false
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide dock icon
        NSApp.setActivationPolicy(.accessory)
        
        // Create status window
        createStatusWindow()
        
        // Create overlay window (hidden initially)
        createOverlayWindow()
        
        // Start checking permissions
        checkPermissions()
        
        // Set up timer to check every 2 seconds
        checkTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            self.checkPermissions()
        }
    }
    
    func checkPermissions() {
        var status = "🧪 QuasiLauncher Permission Status\n"
        status += "==================================\n\n"
        
        // Check accessibility
        let accessibility = AXIsProcessTrusted()
        status += "1️⃣ Accessibility: \(accessibility ? "✅" : "❌")\n"
        
        // Check input monitoring
        let inputMonitoring = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
        status += "2️⃣ Input Monitoring: \(inputMonitoring == kIOHIDAccessTypeGranted ? "✅" : "❌")\n"
        
        // Test event tap
        let eventMask = CGEventMask(1 << CGEventType.keyDown.rawValue)
        let testTap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: eventMask,
            callback: { _, _, _, _ in return nil },
            userInfo: nil
        )
        
        let eventTapWorks = testTap != nil
        if let tap = testTap {
            CFMachPortInvalidate(tap)
        }
        
        status += "3️⃣ Event Tap: \(eventTapWorks ? "✅" : "❌")\n\n"
        
        // If everything works, create the real event tap
        if accessibility && eventTapWorks {
            status += "🎉 All permissions granted!\n\n"
            status += "Press Escape to test key detection...\n\n"
            
            if eventTap == nil {
                createEventTap()
                status += "Event tap created and listening...\n"
            }
            
            status += "\nKey Events:\n"
        } else {
            status += "⚠️  Please grant permissions in:\n"
            status += "System Preferences > Security & Privacy > Privacy\n"
            status += "• Accessibility\n"
            status += "• Input Monitoring\n"
        }
        
        statusLabel?.stringValue = status
    }
    
    func createStatusWindow() {
        statusWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 300),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        
        statusWindow?.title = "QuasiLauncher Status"
        statusWindow?.center()
        
        // Create status label
        statusLabel = NSTextField(frame: NSRect(x: 20, y: 20, width: 360, height: 260))
        statusLabel?.isEditable = false
        statusLabel?.isBordered = false
        statusLabel?.backgroundColor = .clear
        statusLabel?.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        
        statusWindow?.contentView?.addSubview(statusLabel!)
        statusWindow?.makeKeyAndOrderFront(nil)
    }
    
    func createOverlayWindow() {
        // Create a transparent overlay window
        overlayWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 500, height: 80),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        
        overlayWindow?.level = .floating
        overlayWindow?.backgroundColor = NSColor.black.withAlphaComponent(0.8)
        overlayWindow?.isOpaque = false
        overlayWindow?.hasShadow = true
        overlayWindow?.ignoresMouseEvents = false
        
        // Center on screen
        if let screen = NSScreen.main {
            let screenFrame = screen.frame
            let windowFrame = overlayWindow!.frame
            let x = (screenFrame.width - windowFrame.width) / 2
            let y = (screenFrame.height - windowFrame.height) / 2 + 100 // Slightly above center
            overlayWindow?.setFrameOrigin(NSPoint(x: x, y: y))
        }
        
        // Create command input field
        commandField = NSTextField(frame: NSRect(x: 20, y: 20, width: 460, height: 40))
        commandField?.font = .systemFont(ofSize: 18, weight: .medium)
        commandField?.backgroundColor = .white
        commandField?.textColor = .black
        commandField?.placeholderString = "Type command..."
        commandField?.isBordered = true
        commandField?.focusRingType = .default
        commandField?.layer?.cornerRadius = 8
        
        overlayWindow?.contentView?.addSubview(commandField!)
        overlayWindow?.orderOut(nil) // Hidden initially
    }
    
    func showOverlay() {
        if !showingOverlay {
            showingOverlay = true
            commandField?.stringValue = ""
            overlayWindow?.makeKeyAndOrderFront(nil)
            overlayWindow?.makeFirstResponder(commandField)
        }
    }
    
    func hideOverlay() {
        if showingOverlay {
            showingOverlay = false
            overlayWindow?.orderOut(nil)
            statusWindow?.makeKeyAndOrderFront(nil)
        }
    }
    
    func createEventTap() {
        let eventMask = CGEventMask(
            1 << CGEventType.keyDown.rawValue |
            1 << CGEventType.keyUp.rawValue |
            1 << CGEventType.flagsChanged.rawValue
        )
        
        eventTap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: eventMask,
            callback: { proxy, type, event, refcon in
                let delegate = Unmanaged<SimpleAppDelegate>.fromOpaque(refcon!).takeUnretainedValue()
                return delegate.handleEvent(proxy: proxy, type: type, event: event)
            },
            userInfo: Unmanaged.passUnretained(self).toOpaque()
        )
        
        if let tap = eventTap {
            let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
            CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
            CGEvent.tapEnable(tap: tap, enable: true)
        }
    }
    
    func handleEvent(proxy: CGEventTapProxy, type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        let keyCode = event.getIntegerValueField(.keyboardEventKeycode)
        
        // Check for Escape key (keycode 53)
        if keyCode == 53 {
            let isPressed = (type == .keyDown)
            
            DispatchQueue.main.async {
                if isPressed && !self.isEscapePressed {
                    // Escape pressed - show overlay
                    self.isEscapePressed = true
                    self.showOverlay()
                    
                    // Update status
                    var currentText = self.statusLabel?.stringValue ?? ""
                    if !currentText.contains("Key Events:") {
                        currentText += "\nKey Events:\n"
                    }
                    currentText += "Escape pressed - overlay shown at \(Date())\n"
                    
                    // Keep only last 5 events
                    let lines = currentText.components(separatedBy: .newlines)
                    let keyEventIndex = lines.firstIndex(of: "Key Events:") ?? lines.count
                    let beforeKeyEvents = lines[0...keyEventIndex]
                    let keyEvents = Array(lines[(keyEventIndex + 1)...].suffix(5))
                    
                    self.statusLabel?.stringValue = (beforeKeyEvents + keyEvents).joined(separator: "\n")
                    
                } else if !isPressed && self.isEscapePressed {
                    // Escape released - hide overlay and execute command
                    self.isEscapePressed = false
                    let command = self.commandField?.stringValue ?? ""
                    self.hideOverlay()
                    
                    if !command.isEmpty {
                        self.executeCommand(command)
                    }
                    
                    // Update status
                    var currentText = self.statusLabel?.stringValue ?? ""
                    if !currentText.contains("Key Events:") {
                        currentText += "\nKey Events:\n"
                    }
                    currentText += "Escape released - overlay hidden\(command.isEmpty ? "" : ", executing: \(command)") at \(Date())\n"
                    
                    // Keep only last 5 events
                    let lines = currentText.components(separatedBy: .newlines)
                    let keyEventIndex = lines.firstIndex(of: "Key Events:") ?? lines.count
                    let beforeKeyEvents = lines[0...keyEventIndex]
                    let keyEvents = Array(lines[(keyEventIndex + 1)...].suffix(5))
                    
                    self.statusLabel?.stringValue = (beforeKeyEvents + keyEvents).joined(separator: "\n")
                }
            }
            
            // Block the escape key to prevent beeping
            return nil
        }
        
        // If overlay is showing, intercept all keystrokes except escape
        if showingOverlay && type == .keyDown && keyCode != 53 {
            // Handle typing into the command field
            DispatchQueue.main.async {
                self.handleTyping(keyCode: keyCode, event: event)
            }
            return nil // Block from system
        }
        
        return Unmanaged.passUnretained(event)
    }
    
    func handleTyping(keyCode: Int64, event: CGEvent) {
        guard let commandField = self.commandField else { return }
        
        // Convert keycode to character
        let currentText = commandField.stringValue
        
        // Handle common keys
        switch keyCode {
        case 51: // Delete/Backspace
            if !currentText.isEmpty {
                commandField.stringValue = String(currentText.dropLast())
            }
        case 36: // Return/Enter
            // Execute command immediately on Enter
            let command = commandField.stringValue
            hideOverlay()
            if !command.isEmpty {
                executeCommand(command)
            }
        case 49: // Space
            commandField.stringValue = currentText + " "
        default:
            // Convert keycode to character using Carbon
            let chars = keyCodeToString(keyCode: keyCode, event: event)
            commandField.stringValue = currentText + chars
        }
    }
    
    func keyCodeToString(keyCode: Int64, event: CGEvent) -> String {
        // Create a keyboard layout
        let keyboard = TISCopyCurrentKeyboardInputSource().takeRetainedValue()
        let layoutData = TISGetInputSourceProperty(keyboard, kTISPropertyUnicodeKeyLayoutData)
        
        if layoutData != nil {
            let keyboardLayout = unsafeBitCast(layoutData, to: CFData.self)
            let keyLayoutPtr = CFDataGetBytePtr(keyboardLayout)
            
            var deadKeyState: UInt32 = 0
            var length = 0
            var chars = [UniChar](repeating: 0, count: 4)
            
            let status = UCKeyTranslate(
                UnsafePointer<UCKeyboardLayout>(OpaquePointer(keyLayoutPtr)),
                UInt16(keyCode),
                UInt16(kUCKeyActionDown),
                0, // No modifiers for now
                UInt32(LMGetKbdType()),
                UInt32(kUCKeyTranslateNoDeadKeysBit),
                &deadKeyState,
                4,
                &length,
                &chars
            )
            
            if status == noErr && length > 0 {
                return String(utf16CodeUnits: chars, count: length)
            }
        }
        
        // Fallback for common keys
        switch keyCode {
        case 0: return "a"
        case 1: return "s"
        case 2: return "d"
        case 3: return "f"
        case 4: return "h"
        case 5: return "g"
        case 6: return "z"
        case 7: return "x"
        case 8: return "c"
        case 9: return "v"
        case 11: return "b"
        case 12: return "q"
        case 13: return "w"
        case 14: return "e"
        case 15: return "r"
        case 16: return "y"
        case 17: return "t"
        case 18: return "1"
        case 19: return "2"
        case 20: return "3"
        case 21: return "4"
        case 22: return "6"
        case 23: return "5"
        case 24: return "="
        case 25: return "9"
        case 26: return "7"
        case 27: return "-"
        case 28: return "8"
        case 29: return "0"
        case 30: return "]"
        case 31: return "o"
        case 32: return "u"
        case 33: return "["
        case 34: return "i"
        case 35: return "p"
        case 37: return "l"
        case 38: return "j"
        case 39: return "'"
        case 40: return "k"
        case 41: return ";"
        case 42: return "\\"
        case 43: return ","
        case 44: return "/"
        case 45: return "n"
        case 46: return "m"
        case 47: return "."
        default: return ""
        }
    }
    
    func executeCommand(_ command: String) {
        // Simple command execution
        let trimmedCommand = command.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        
        if trimmedCommand.isEmpty {
            return
        }
        
        // Try to launch as application
        let workspace = NSWorkspace.shared
        
        // First try direct app launch by bundle ID
        if let appURL = workspace.urlForApplication(withBundleIdentifier: trimmedCommand) {
            workspace.openApplication(at: appURL, configuration: NSWorkspace.OpenConfiguration()) { _, _ in }
            return
        }
        
        // Try fuzzy app name matching
        if let apps = try? FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: "/Applications"), includingPropertiesForKeys: nil) {
            for appURL in apps {
                let appName = appURL.deletingPathExtension().lastPathComponent.lowercased()
                if appName.contains(trimmedCommand) || trimmedCommand.contains(appName) {
                    workspace.openApplication(at: appURL, configuration: NSWorkspace.OpenConfiguration()) { _, _ in }
                    return
                }
            }
        }
        
        // Try as shell command
        let task = Process()
        task.launchPath = "/bin/bash"
        task.arguments = ["-c", command]
        try? task.run()
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// Entry point
let app = NSApplication.shared
let delegate = SimpleAppDelegate()
app.delegate = delegate
app.run()