#!/usr/bin/env swift

import Foundation
import Carbon
import IOKit.hid

print("🔍 Debug Permission Check")
print("========================")

// Test 1: Accessibility
let accessibility = AXIsProcessTrusted()
print("AXIsProcessTrusted(): \(accessibility)")

// Test 2: Input Monitoring  
let inputMonitoring = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
print("IOHIDCheckAccess(): \(inputMonitoring.rawValue)")
print("  granted=\(kIOHIDAccessTypeGranted.rawValue), denied=\(kIOHIDAccessTypeDenied.rawValue)")

// Test 3: Event Tap Creation
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

print("Event tap creation: \(eventTapWorks)")

// Test 4: Code signature
let executablePath = CommandLine.arguments[0]
print("Executable: \(executablePath)")

let task = Process()
task.launchPath = "/usr/bin/codesign"
task.arguments = ["-dv", executablePath]

let pipe = Pipe()
task.standardError = pipe

do {
    try task.run()
    task.waitUntilExit()
    
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    if let output = String(data: data, encoding: .utf8) {
        for line in output.components(separatedBy: .newlines) {
            if line.contains("Identifier") || line.contains("TeamIdentifier") {
                print(line)
            }
        }
    }
} catch {
    print("Failed to check signature: \(error)")
}

print("\nResult: \(accessibility && eventTapWorks ? "SHOULD WORK" : "PERMISSIONS MISSING")")