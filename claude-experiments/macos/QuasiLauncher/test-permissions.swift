#!/usr/bin/env swift

import Foundation
import Carbon
import IOKit.hid

print("🧪 Permission Test Tool")
print("======================")

// Test 1: Basic Accessibility Check
print("\n1️⃣ Testing Accessibility Permission:")
let accessibility = AXIsProcessTrusted()
print("   AXIsProcessTrusted(): \(accessibility)")

if !accessibility {
    print("   ⚠️  Requesting accessibility permission...")
    let _ = AXIsProcessTrustedWithOptions([kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true] as CFDictionary)
}

// Test 2: Input Monitoring Check
print("\n2️⃣ Testing Input Monitoring Permission:")
let inputMonitoring = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
print("   IOHIDCheckAccess(): \(inputMonitoring.rawValue)")
print("   Constants: granted=\(kIOHIDAccessTypeGranted.rawValue), denied=\(kIOHIDAccessTypeDenied.rawValue)")

if inputMonitoring != kIOHIDAccessTypeGranted {
    print("   ⚠️  Requesting input monitoring permission...")
    let _ = IOHIDRequestAccess(kIOHIDRequestTypeListenEvent)
}

// Test 3: Event Tap Test (the real test)
print("\n3️⃣ Testing Event Tap Creation:")
let eventMask = CGEventMask(1 << CGEventType.keyDown.rawValue)

let eventTap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .defaultTap,
    eventsOfInterest: eventMask,
    callback: { proxy, type, event, refcon in
        print("🔥 Key detected! keyCode: \(event.getIntegerValueField(.keyboardEventKeycode))")
        return Unmanaged.passUnretained(event)
    },
    userInfo: nil
)

if let tap = eventTap {
    print("   ✅ Event tap created successfully!")
    
    // Enable it
    CGEvent.tapEnable(tap: tap, enable: true)
    print("   ✅ Event tap enabled!")
    
    // Test for 5 seconds
    print("   ⏱️  Testing for 5 seconds - press any key...")
    
    let runLoop = CFRunLoopGetCurrent()
    let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
    CFRunLoopAddSource(runLoop, source, .defaultMode)
    
    // Run for 5 seconds
    CFRunLoopRunInMode(.defaultMode, 5.0, false)
    
    // Cleanup
    CGEvent.tapEnable(tap: tap, enable: false)
    CFRunLoopRemoveSource(runLoop, source, .defaultMode)
    CFMachPortInvalidate(tap)
    
    print("   ✅ Test complete!")
} else {
    print("   ❌ Event tap creation FAILED")
    print("   This means permissions are not properly granted or app is not properly signed")
}

// Test 4: Code Signature Check
print("\n4️⃣ Checking Code Signature:")
let task = Process()
task.launchPath = "/usr/bin/codesign"
task.arguments = ["-dv", ProcessInfo.processInfo.arguments[0]]

let pipe = Pipe()
task.standardError = pipe

do {
    try task.run()
    task.waitUntilExit()
    
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    if let output = String(data: data, encoding: .utf8) {
        print("   Code signature info:")
        for line in output.components(separatedBy: .newlines) {
            if !line.isEmpty && (line.contains("Signature") || line.contains("TeamIdentifier") || line.contains("Identifier")) {
                print("     \(line)")
            }
        }
    }
} catch {
    print("   ❌ Failed to check signature: \(error)")
}

// Summary
print("\n📊 Summary:")
print("   Accessibility: \(accessibility ? "✅" : "❌")")
print("   Input Monitoring: \(inputMonitoring == kIOHIDAccessTypeGranted ? "✅" : "❌")")
print("   Event Tap: \(eventTap != nil ? "✅" : "❌")")
print("")
print("If any are ❌, check System Preferences > Security & Privacy > Privacy")
print("You may need to remove and re-add this app after proper code signing.")