#!/usr/bin/env swift

import AppKit
import Carbon

class TestDelegate: NSObject, NSApplicationDelegate {
    var eventTap: CFMachPort?
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        print("App launched, testing escape key...")
        
        let eventMask = CGEventMask(1 << CGEventType.keyDown.rawValue | 1 << CGEventType.keyUp.rawValue)
        
        eventTap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: eventMask,
            callback: { proxy, type, event, refcon in
                let keyCode = event.getIntegerValueField(.keyboardEventKeycode)
                if keyCode == 53 { // Escape
                    let isPressed = (type == .keyDown)
                    print("Escape key \(isPressed ? "pressed" : "released")")
                    return nil // Block escape
                }
                return Unmanaged.passUnretained(event)
            },
            userInfo: nil
        )
        
        if let tap = eventTap {
            let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
            CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
            CGEvent.tapEnable(tap: tap, enable: true)
            print("Event tap created successfully")
        } else {
            print("Failed to create event tap")
        }
    }
}

let app = NSApplication.shared
let delegate = TestDelegate()
app.delegate = delegate
app.run()