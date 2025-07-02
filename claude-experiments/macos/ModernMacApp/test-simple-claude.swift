#!/usr/bin/env swift

import Foundation

// Simple test to isolate the Cocoa error
print("🧪 Testing Simple Claude Execution")

let process = Process()
process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/claude")
process.arguments = ["--print", "hello"]

let outputPipe = Pipe()
let errorPipe = Pipe()

process.standardOutput = outputPipe
process.standardError = errorPipe

do {
    print("📡 Starting process...")
    try process.run()
    print("✅ Process started successfully")
    
    process.waitUntilExit()
    print("📊 Process completed with exit code: \(process.terminationStatus)")
    
    let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
    let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
    
    if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
        print("📝 Output: \(output)")
    }
    
    if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
        print("⚠️ Error: \(error)")
    }
    
} catch {
    print("❌ Error executing process: \(error)")
    print("Error code: \((error as NSError).code)")
    print("Error domain: \((error as NSError).domain)")
}