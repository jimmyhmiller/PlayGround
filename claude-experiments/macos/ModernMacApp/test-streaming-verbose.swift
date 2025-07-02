#!/usr/bin/env swift

import Foundation

print("🧪 Testing Claude Streaming with --verbose")
print(String(repeating: "=", count: 40))

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
process.arguments = [
    "claude",
    "--print",
    "--verbose",
    "--output-format", "stream-json",
    "Say hello and count to 3"
]

let outputPipe = Pipe()
let errorPipe = Pipe()

process.standardOutput = outputPipe
process.standardError = errorPipe

do {
    try process.run()
    
    let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
    let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
    
    process.waitUntilExit()
    
    print("📊 Exit code: \(process.terminationStatus)")
    
    if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
        print("📝 Streaming JSON Output:")
        // Split by lines to see the JSON structure
        let lines = output.components(separatedBy: .newlines)
        for (i, line) in lines.enumerated() {
            if !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                print("Line \(i): \(line)")
            }
        }
    }
    
    if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
        print("⚠️ Error output:")
        print(error)
    }
    
} catch {
    print("❌ Failed to test: \(error)")
}

print("\n🏁 Test complete!")