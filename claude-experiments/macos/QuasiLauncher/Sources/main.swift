import SwiftUI
import AppKit
import Carbon
import IOKit.hid
import QuartzCore

class QuasiLauncherAppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    var overlayWindow: NSWindow?
    var commandField: NSTextField?
    var suggestionLabel: NSTextField?
    var eventTap: CFMachPort?
    var checkTimer: Timer?
    var isEscapePressed = false
    var showingOverlay = false
    var permissionsGranted = false
    var availableApps: [String] = []
    var animationsEnabled = true
    var animationDuration: Double = 0.25
    var slideDistance: CGFloat = 80
    var centerPosition = false
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide dock icon - this is a menu bar app
        NSApp.setActivationPolicy(.accessory)
        
        // Create menu bar item
        createMenuBarItem()
        
        // Create overlay window (hidden initially)
        createOverlayWindow()
        
        // Load available apps
        loadAvailableApps()
        
        // Start checking permissions
        checkPermissions()
        
        // Set up timer to check every 5 seconds
        checkTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            self.checkPermissions()
        }
    }
    
    func createMenuBarItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        
        if let statusButton = statusItem?.button {
            statusButton.title = "🚀"
            statusButton.toolTip = "QuasiLauncher - Hold Escape to launch"
        }
        
        let menu = NSMenu()
        
        let statusMenuItem = NSMenuItem(title: "Checking permissions...", action: nil, keyEquivalent: "")
        statusMenuItem.tag = 100 // We'll update this
        menu.addItem(statusMenuItem)
        
        menu.addItem(NSMenuItem.separator())
        
        let testMenuItem = NSMenuItem(title: "Test Launcher", action: #selector(testLauncher), keyEquivalent: "")
        testMenuItem.target = self
        menu.addItem(testMenuItem)
        
        let permissionsMenuItem = NSMenuItem(title: "Open System Preferences", action: #selector(openSystemPreferences), keyEquivalent: "")
        permissionsMenuItem.target = self
        menu.addItem(permissionsMenuItem)
        
        menu.addItem(NSMenuItem.separator())
        
        let animationMenuItem = NSMenuItem(title: "✅ Animations Enabled", action: #selector(toggleAnimations), keyEquivalent: "")
        animationMenuItem.target = self
        animationMenuItem.tag = 200 // We'll update this
        menu.addItem(animationMenuItem)
        
        // Animation controls submenu
        let animControlsMenu = NSMenu()
        
        let speedFastItem = NSMenuItem(title: "Speed: Fast (0.15s)", action: #selector(setSpeedFast), keyEquivalent: "")
        speedFastItem.target = self
        animControlsMenu.addItem(speedFastItem)
        
        let speedNormalItem = NSMenuItem(title: "Speed: Normal (0.25s)", action: #selector(setSpeedNormal), keyEquivalent: "")
        speedNormalItem.target = self
        speedNormalItem.state = .on
        animControlsMenu.addItem(speedNormalItem)
        
        let speedSlowItem = NSMenuItem(title: "Speed: Slow (0.4s)", action: #selector(setSpeedSlow), keyEquivalent: "")
        speedSlowItem.target = self
        animControlsMenu.addItem(speedSlowItem)
        
        animControlsMenu.addItem(NSMenuItem.separator())
        
        let slideSmallItem = NSMenuItem(title: "Slide: Small (40px)", action: #selector(setSlideSmall), keyEquivalent: "")
        slideSmallItem.target = self
        animControlsMenu.addItem(slideSmallItem)
        
        let slideNormalItem = NSMenuItem(title: "Slide: Normal (80px)", action: #selector(setSlideNormal), keyEquivalent: "")
        slideNormalItem.target = self
        slideNormalItem.state = .on
        animControlsMenu.addItem(slideNormalItem)
        
        let slideLargeItem = NSMenuItem(title: "Slide: Large (150px)", action: #selector(setSlideLarge), keyEquivalent: "")
        slideLargeItem.target = self
        animControlsMenu.addItem(slideLargeItem)
        
        let animControlsItem = NSMenuItem(title: "Animation Settings", action: nil, keyEquivalent: "")
        animControlsItem.submenu = animControlsMenu
        menu.addItem(animControlsItem)
        
        menu.addItem(NSMenuItem.separator())
        
        let positionMenuItem = NSMenuItem(title: "❌ Center Position", action: #selector(togglePosition), keyEquivalent: "")
        positionMenuItem.target = self
        positionMenuItem.tag = 300 // We'll update this
        menu.addItem(positionMenuItem)
        
        menu.addItem(NSMenuItem.separator())
        
        let quitMenuItem = NSMenuItem(title: "Quit QuasiLauncher", action: #selector(quit), keyEquivalent: "q")
        quitMenuItem.target = self
        menu.addItem(quitMenuItem)
        
        statusItem?.menu = menu
    }
    
    @objc func testLauncher() {
        if permissionsGranted {
            showOverlay()
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.hideOverlay()
            }
        } else {
            let alert = NSAlert()
            alert.messageText = "Permissions Required"
            alert.informativeText = "Please grant Accessibility and Input Monitoring permissions in System Preferences."
            alert.addButton(withTitle: "Open System Preferences")
            alert.addButton(withTitle: "Cancel")
            
            if alert.runModal() == .alertFirstButtonReturn {
                openSystemPreferences()
            }
        }
    }
    
    @objc func openSystemPreferences() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
    }
    
    @objc func toggleAnimations() {
        animationsEnabled.toggle()
        
        // Update menu item text
        if let menu = statusItem?.menu,
           let animationMenuItem = menu.item(withTag: 200) {
            animationMenuItem.title = animationsEnabled ? "✅ Animations Enabled" : "❌ Animations Disabled"
        }
    }
    
    @objc func setSpeedFast() {
        animationDuration = 0.15
        updateSpeedMenuStates()
    }
    
    @objc func setSpeedNormal() {
        animationDuration = 0.25
        updateSpeedMenuStates()
    }
    
    @objc func setSpeedSlow() {
        animationDuration = 0.4
        updateSpeedMenuStates()
    }
    
    @objc func setSlideSmall() {
        slideDistance = 40
        updateSlideMenuStates()
    }
    
    @objc func setSlideNormal() {
        slideDistance = 80
        updateSlideMenuStates()
    }
    
    @objc func setSlideLarge() {
        slideDistance = 150
        updateSlideMenuStates()
    }
    
    @objc func togglePosition() {
        centerPosition.toggle()
        
        // Update menu item text
        if let menu = statusItem?.menu,
           let positionMenuItem = menu.item(withTag: 300) {
            positionMenuItem.title = centerPosition ? "✅ Center Position" : "❌ Center Position"
        }
    }
    
    func updateSpeedMenuStates() {
        guard let menu = statusItem?.menu else { return }
        for item in menu.items {
            if let submenu = item.submenu {
                for subItem in submenu.items {
                    if subItem.title.contains("Speed:") {
                        subItem.state = .off
                        if (subItem.title.contains("Fast") && animationDuration == 0.15) ||
                           (subItem.title.contains("Normal") && animationDuration == 0.25) ||
                           (subItem.title.contains("Slow") && animationDuration == 0.4) {
                            subItem.state = .on
                        }
                    }
                }
            }
        }
    }
    
    func updateSlideMenuStates() {
        guard let menu = statusItem?.menu else { return }
        for item in menu.items {
            if let submenu = item.submenu {
                for subItem in submenu.items {
                    if subItem.title.contains("Slide:") {
                        subItem.state = .off
                        if (subItem.title.contains("Small") && slideDistance == 40) ||
                           (subItem.title.contains("Normal") && slideDistance == 80) ||
                           (subItem.title.contains("Large") && slideDistance == 150) {
                            subItem.state = .on
                        }
                    }
                }
            }
        }
    }
    
    @objc func quit() {
        NSApplication.shared.terminate(nil)
    }
    
    func checkPermissions() {
        // Check accessibility
        let accessibility = AXIsProcessTrusted()
        
        // For the app bundle, we'll assume input monitoring is granted if accessibility is granted
        // and we can create an event tap. The IOHIDCheckAccess seems unreliable in app bundles.
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
        
        // Both accessibility AND event tap creation must work
        let bothGranted = accessibility && eventTapWorks
        
        
        // Update menu bar status
        DispatchQueue.main.async {
            if let menu = self.statusItem?.menu,
               let statusMenuItem = menu.item(withTag: 100) {
                if bothGranted {
                    statusMenuItem.title = "✅ Ready - Hold Escape to launch"
                    self.statusItem?.button?.title = "🚀"
                } else {
                    statusMenuItem.title = "❌ Permissions needed"
                    self.statusItem?.button?.title = "⚠️"
                }
            }
        }
        
        // Set up event tap if permissions are granted and not already set up
        if bothGranted && !permissionsGranted {
            permissionsGranted = true
            createEventTap()
        } else if !bothGranted && permissionsGranted {
            permissionsGranted = false
            // Disable event tap
            if let tap = eventTap {
                CGEvent.tapEnable(tap: tap, enable: false)
                CFMachPortInvalidate(tap)
                eventTap = nil
            }
        }
        
        // Request permissions if needed
        if !accessibility {
            let _ = AXIsProcessTrustedWithOptions([kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true] as CFDictionary)
        }
    }
    
    func createOverlayWindow() {
        // Create sleek macOS Sequoia-style window
        overlayWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 420, height: 70),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        
        overlayWindow?.level = .floating
        overlayWindow?.backgroundColor = .clear
        overlayWindow?.isOpaque = false
        overlayWindow?.hasShadow = true
        overlayWindow?.ignoresMouseEvents = false
        
        // Position in top-left corner
        if let screen = NSScreen.main {
            let screenFrame = screen.frame
            let windowFrame = overlayWindow!.frame
            let x = screenFrame.minX + 40 // 40pt from left edge
            let y = screenFrame.maxY - windowFrame.height - 80 // 80pt from top (below menu bar)
            overlayWindow?.setFrameOrigin(NSPoint(x: x, y: y))
            print("DEBUG: Positioned overlay at (\(x), \(y)) on screen \(screenFrame)")
        }
        
        // Create the main container
        let containerView = NSView(frame: NSRect(x: 0, y: 0, width: 420, height: 70))
        containerView.wantsLayer = true
        
        // Create simple background that works
        let backgroundView = NSView(frame: containerView.bounds)
        backgroundView.wantsLayer = true
        backgroundView.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        backgroundView.layer?.cornerRadius = 12
        backgroundView.layer?.masksToBounds = true
        backgroundView.layer?.borderWidth = 1
        backgroundView.layer?.borderColor = NSColor.separatorColor.cgColor
        
        containerView.addSubview(backgroundView)
        
        // Create clean input field (simplified)
        commandField = NSTextField(frame: NSRect(x: 20, y: 20, width: 380, height: 30))
        commandField?.font = NSFont.systemFont(ofSize: 15, weight: .regular)
        commandField?.backgroundColor = .clear
        commandField?.textColor = .labelColor
        commandField?.placeholderString = "Search apps and commands..."
        commandField?.isBordered = false
        commandField?.focusRingType = .none
        commandField?.cell?.usesSingleLineMode = true
        commandField?.cell?.wraps = false
        commandField?.cell?.isScrollable = true
        
        containerView.addSubview(commandField!)
        
        // Add suggestion label at bottom
        suggestionLabel = NSTextField(frame: NSRect(x: 20, y: 5, width: 380, height: 14))
        suggestionLabel?.stringValue = ""
        suggestionLabel?.font = NSFont.systemFont(ofSize: 11, weight: .regular)
        suggestionLabel?.textColor = .tertiaryLabelColor
        suggestionLabel?.backgroundColor = .clear
        suggestionLabel?.isBordered = false
        suggestionLabel?.isEditable = false
        suggestionLabel?.isSelectable = false
        
        containerView.addSubview(suggestionLabel!)
        
        overlayWindow?.contentView = containerView
        overlayWindow?.orderOut(nil) // Hidden initially
    }
    
    func loadAvailableApps() {
        DispatchQueue.global(qos: .background).async {
            var apps: [String] = []
            
            if let appURLs = try? FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: "/Applications"), includingPropertiesForKeys: nil) {
                for appURL in appURLs {
                    let appName = appURL.deletingPathExtension().lastPathComponent
                    apps.append(appName)
                }
            }
            
            DispatchQueue.main.async {
                self.availableApps = apps.sorted()
            }
        }
    }
    
    func updateSuggestion(for input: String) {
        guard !input.isEmpty else {
            suggestionLabel?.stringValue = ""
            return
        }
        
        let lowercaseInput = input.lowercased()
        
        // Find first matching app
        if let match = availableApps.first(where: { $0.lowercased().hasPrefix(lowercaseInput) || $0.lowercased().contains(lowercaseInput) }) {
            suggestionLabel?.stringValue = "→ \(match)"
        } else {
            suggestionLabel?.stringValue = "→ Run as shell command"
        }
    }
    
    func showOverlay() {
        if !showingOverlay {
            showingOverlay = true
            commandField?.stringValue = ""
            suggestionLabel?.stringValue = ""
            
            if animationsEnabled {
                // Animated show - slide down from above using parameters
                if let screen = NSScreen.main {
                    let screenFrame = screen.frame
                    let windowWidth: CGFloat = 420
                    
                    let finalX = centerPosition ? 
                        (screenFrame.width - windowWidth) / 2 + screenFrame.minX : 
                        screenFrame.minX + 40
                    let finalY = centerPosition ?
                        (screenFrame.height - 70) / 2 + screenFrame.minY :
                        screenFrame.maxY - 130
                    
                    // Start position should be ABOVE the final position to slide down
                    let startY = finalY + slideDistance // Start higher up
                    
                    overlayWindow?.setFrameOrigin(NSPoint(x: finalX, y: startY))
                    print("DEBUG: Position=\(centerPosition ? "CENTER" : "TOP-LEFT")")
                    print("DEBUG: Sliding DOWN from y=\(startY) to y=\(finalY), distance=\(slideDistance)")
                }
                
                overlayWindow?.alphaValue = 1.0
                overlayWindow?.makeKeyAndOrderFront(nil)
                overlayWindow?.makeFirstResponder(commandField)
                
                // Slide down animation with configurable parameters
                print("DEBUG: Starting animation with duration \(animationDuration)")
                NSAnimationContext.runAnimationGroup({ context in
                    context.duration = animationDuration
                    context.timingFunction = CAMediaTimingFunction(name: .easeOut)
                    context.allowsImplicitAnimation = true
                    
                    print("DEBUG: Animation context created, duration=\(context.duration)")
                    
                    if let screen = NSScreen.main {
                        let screenFrame = screen.frame
                        let windowWidth: CGFloat = 420
                        
                        let finalX = centerPosition ? 
                            (screenFrame.width - windowWidth) / 2 + screenFrame.minX : 
                            screenFrame.minX + 40
                        let finalY = centerPosition ?
                            (screenFrame.height - 70) / 2 + screenFrame.minY :
                            screenFrame.maxY - 130
                            
                        print("DEBUG: Animating to final position (\(finalX), \(finalY))")
                        overlayWindow?.animator().setFrameOrigin(NSPoint(x: finalX, y: finalY))
                    }
                }, completionHandler: {
                    print("DEBUG: Animation completed")
                })
            } else {
                // Instant show
                if let screen = NSScreen.main {
                    let screenFrame = screen.frame
                    let windowWidth: CGFloat = 420
                    
                    let x = centerPosition ? 
                        (screenFrame.width - windowWidth) / 2 + screenFrame.minX : 
                        screenFrame.minX + 40
                    let y = centerPosition ?
                        (screenFrame.height - 70) / 2 + screenFrame.minY :
                        screenFrame.maxY - 130
                    overlayWindow?.setFrameOrigin(NSPoint(x: x, y: y))
                }
                
                overlayWindow?.alphaValue = 1.0
                overlayWindow?.makeKeyAndOrderFront(nil)
                overlayWindow?.makeFirstResponder(commandField)
            }
        }
    }
    
    func hideOverlay() {
        if showingOverlay {
            showingOverlay = false
            
            if animationsEnabled {
                // Animated hide - slide back up using parameters (no fade)
                NSAnimationContext.runAnimationGroup({ context in
                    context.duration = animationDuration * 0.8 // Slightly faster hide
                    context.timingFunction = CAMediaTimingFunction(name: .easeIn)
                    
                    if let screen = NSScreen.main {
                        let screenFrame = screen.frame
                        let windowWidth: CGFloat = 420
                        
                        let currentX = centerPosition ? 
                            (screenFrame.width - windowWidth) / 2 + screenFrame.minX : 
                            screenFrame.minX + 40
                        let currentY = centerPosition ?
                            (screenFrame.height - 70) / 2 + screenFrame.minY :
                            screenFrame.maxY - 130
                        let finalY = currentY + slideDistance // Slide back up (higher Y)
                        
                        overlayWindow?.animator().setFrameOrigin(NSPoint(x: currentX, y: finalY))
                        print("DEBUG: Sliding UP from y=\(currentY) to y=\(finalY)")
                    }
                }) {
                    self.overlayWindow?.orderOut(nil)
                }
            } else {
                // Instant hide
                overlayWindow?.orderOut(nil)
            }
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
                let delegate = Unmanaged<QuasiLauncherAppDelegate>.fromOpaque(refcon!).takeUnretainedValue()
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
                    
                } else if !isPressed && self.isEscapePressed {
                    // Escape released - hide overlay and execute command
                    self.isEscapePressed = false
                    let command = self.commandField?.stringValue ?? ""
                    self.hideOverlay()
                    
                    if !command.isEmpty {
                        self.executeCommand(command)
                    }
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
        
        // Update suggestions after any text change
        updateSuggestion(for: commandField.stringValue)
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
        return false // Keep running even if no windows are open
    }
}

// Entry point
let app = NSApplication.shared
let delegate = QuasiLauncherAppDelegate()
app.delegate = delegate
app.run()