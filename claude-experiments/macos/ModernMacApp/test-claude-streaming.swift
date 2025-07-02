#!/usr/bin/env swift

import Foundation

print("🧪 Testing Claude Streaming")
print(String(repeating: "=", count: 40))

// Test the correct streaming format based on the help output
func testStreamingWithCorrectFormat() {
    print("🔍 Testing claude with --print --output-format stream-json...")
    
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = [
        "claude",
        "--print",
        "--output-format", "stream-json",
        "Say hello and count to 3"
    ]
    
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    
    process.standardOutput = outputPipe
    process.standardError = errorPipe
    
    do {
        try process.run()
        
        let outputHandle = outputPipe.fileHandleForReading
        var isFinished = false
        
        DispatchQueue.global().async {
            while !isFinished {
                let data = outputHandle.availableData
                
                if data.isEmpty {
                    Thread.sleep(forTimeInterval: 0.01)
                    continue
                }
                
                if let output = String(data: data, encoding: .utf8) {
                    print("📡 Received: \(output.debugDescription)")
                }
            }
        }
        
        process.waitUntilExit()
        isFinished = true
        
        print("📊 Exit code: \(process.terminationStatus)")
        
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
            print("⚠️ Error:")
            print(error)
        }
        
    } catch {
        print("❌ Failed to test streaming: \(error)")
    }
}

// Test non-streaming first to see regular output
func testRegularOutput() {
    print("🔍 Testing regular claude output...")
    
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = [
        "claude",
        "--print",
        "Say hello and count to 3"
    ]
    
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    
    process.standardOutput = outputPipe
    process.standardError = errorPipe
    
    do {
        try process.run()
        process.waitUntilExit()
        
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        
        print("📊 Exit code: \(process.terminationStatus)")
        
        if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
            print("📝 Output:")
            print(output)
        }
        
        if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
            print("⚠️ Error:")
            print(error)
        }
        
    } catch {
        print("❌ Failed to test regular output: \(error)")
    }
}

// Run tests
testRegularOutput()
print("\n" + String(repeating: "-", count: 40))
testStreamingWithCorrectFormat()

print("\n🏁 Tests complete!")