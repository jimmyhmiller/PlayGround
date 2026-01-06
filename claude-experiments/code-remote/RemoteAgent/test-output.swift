#!/usr/bin/env swift

import Foundation

print("Testing process output streaming...")

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
// Use a command that produces output over time
process.arguments = ["bash", "-c", "for i in 1 2 3 4 5; do echo \"Line $i\"; sleep 0.5; done"]

let stdoutPipe = Pipe()
let stderrPipe = Pipe()
process.standardOutput = stdoutPipe
process.standardError = stderrPipe

var outputReceived = false

// Set up async reading for stdout
stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
    let data = handle.availableData
    if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
        outputReceived = true
        print("STDOUT: \(output)", terminator: "")
    }
}

// Set up async reading for stderr
stderrPipe.fileHandleForReading.readabilityHandler = { handle in
    let data = handle.availableData
    if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
        outputReceived = true
        print("STDERR: \(output)", terminator: "")
    }
}

do {
    try process.run()
    print("Process started, waiting...")
    process.waitUntilExit()

    stdoutPipe.fileHandleForReading.readabilityHandler = nil
    stderrPipe.fileHandleForReading.readabilityHandler = nil

    print("\nProcess finished with status: \(process.terminationStatus)")
    print("Output was received: \(outputReceived)")
} catch {
    print("Error: \(error)")
}
