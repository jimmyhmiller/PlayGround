#!/usr/bin/env swift

import Foundation

print("Testing npm output streaming...")

var collectedOutput = ""
let lock = NSLock()

func runInstall(onOutput: @escaping @Sendable (String) -> Void) async -> Bool {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    // Test with npm - just list global packages (quick operation with output)
    process.arguments = ["npm", "list", "-g", "--depth=0"]

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
            DispatchQueue.main.async {
                onOutput("[STDOUT] \(output)")
            }
        }
    }

    stderrPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
            DispatchQueue.main.async {
                onOutput("[STDERR] \(output)")
            }
        }
    }

    do {
        try process.run()
        print("Process started with PID: \(process.processIdentifier)")
    } catch {
        print("Failed to run: \(error)")
        return false
    }

    return await Task.detached {
        process.waitUntilExit()
        // Give a moment for final output
        Thread.sleep(forTimeInterval: 0.1)
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
        return process.terminationStatus == 0
    }.value
}

// Run the test
print("Starting npm list...")

Task {
    let success = await runInstall { output in
        lock.lock()
        collectedOutput += output
        lock.unlock()
        print("RECEIVED: \(output)", terminator: "")
    }

    print("\n---")
    print("Success: \(success)")
    print("Output received: \(!collectedOutput.isEmpty)")
    exit(0)
}

// Keep the run loop alive for callbacks
RunLoop.main.run()
