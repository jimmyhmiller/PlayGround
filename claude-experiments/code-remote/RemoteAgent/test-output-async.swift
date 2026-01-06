#!/usr/bin/env swift

import Foundation

print("Testing async process output streaming...")

var collectedOutput = ""
let lock = NSLock()

func runInstall(onOutput: @escaping @Sendable (String) -> Void) async -> Bool {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["bash", "-c", "for i in 1 2 3 4 5; do echo \"Line $i\"; sleep 0.3; done"]

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
            DispatchQueue.main.async {
                onOutput(output)
            }
        }
    }

    stderrPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
            DispatchQueue.main.async {
                onOutput(output)
            }
        }
    }

    do {
        try process.run()
    } catch {
        print("Failed to run: \(error)")
        return false
    }

    return await Task.detached {
        process.waitUntilExit()
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
        return process.terminationStatus == 0
    }.value
}

// Run the test
print("Starting install simulation...")

Task {
    let success = await runInstall { output in
        lock.lock()
        collectedOutput += output
        lock.unlock()
        print("RECEIVED: \(output)", terminator: "")
    }

    print("\n---")
    print("Success: \(success)")
    print("Total collected:\n\(collectedOutput)")
    exit(0)
}

// Keep the run loop alive for callbacks
RunLoop.main.run()
