#!/usr/bin/env swift

import Foundation

print("Testing ACP over SSH...")

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
process.arguments = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "jimmyhmiller@192.168.0.55",
    "cd /home/jimmyhmiller/Documents/Code/beagle && claude-code-acp"
]

let stdinPipe = Pipe()
let stdoutPipe = Pipe()
let stderrPipe = Pipe()

process.standardInput = stdinPipe
process.standardOutput = stdoutPipe
process.standardError = stderrPipe

var gotResponse = false

// Read stdout async
stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
    let data = handle.availableData
    if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
        print("[STDOUT] \(str)")
        gotResponse = true
    }
}

// Read stderr async
stderrPipe.fileHandleForReading.readabilityHandler = { handle in
    let data = handle.availableData
    if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
        print("[STDERR] \(str)")
    }
}

do {
    try process.run()
    print("Process started with PID: \(process.processIdentifier)")
} catch {
    print("Failed to start: \(error)")
    exit(1)
}

// Wait a moment for process to start
RunLoop.main.run(until: Date(timeIntervalSinceNow: 1.0))

// Send initialize request
let initRequest = """
{"jsonrpc":"2.0","id":"0","method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{"fs":{"readTextFile":true,"writeTextFile":true},"terminal":true},"clientInfo":{"name":"test","title":"Test","version":"1.0.0"}}}

"""

print("Sending initialize request...")

if let data = initRequest.data(using: .utf8) {
    stdinPipe.fileHandleForWriting.write(data)
    print("Request sent!")
} else {
    print("Failed to encode request")
}

// Run the run loop to let handlers fire
print("Waiting for response (running RunLoop)...")
for _ in 0..<50 {  // 5 seconds
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.1))
    if gotResponse {
        break
    }
}

print("Checking process status...")
print("Is running: \(process.isRunning)")
print("Got response: \(gotResponse)")

if !process.isRunning {
    print("Exit code: \(process.terminationStatus)")
}

// Clean up
stdoutPipe.fileHandleForReading.readabilityHandler = nil
stderrPipe.fileHandleForReading.readabilityHandler = nil
stdinPipe.fileHandleForWriting.closeFile()
process.terminate()
print("Done!")
