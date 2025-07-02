#!/usr/bin/env swift

import Foundation

// Standalone test for claude command - can be run with: swift test-claude-command.swift

print("🧪 Testing Claude Command")
print(String(repeating: "=", count: 40))

// Test 1: Check if claude command exists
func testClaudeExists() -> Bool {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
    process.arguments = ["claude"]
    
    do {
        try process.run()
        process.waitUntilExit()
        let exists = process.terminationStatus == 0
        print("✅ Claude command \(exists ? "found" : "not found")")
        return exists
    } catch {
        print("❌ Error checking claude: \(error)")
        return false
    }
}

// Test 2: Get claude help output
func testClaudeHelp() {
    print("\n🔍 Testing claude --help...")
    
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["claude", "--help"]
    
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    
    process.standardOutput = outputPipe
    process.standardError = errorPipe
    
    do {
        try process.run()
        process.waitUntilExit()
        
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        
        print("Exit code: \(process.terminationStatus)")
        
        if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
            print("📋 Help output:")
            print(output)
        }
        
        if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
            print("⚠️ Error output:")
            print(error)
        }
        
    } catch {
        print("❌ Failed to run claude --help: \(error)")
    }
}

// Test 3: Simple claude message
func testSimpleMessage() {
    print("\n🔍 Testing simple message...")
    
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["claude", "Say hello"]
    
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    
    process.standardOutput = outputPipe
    process.standardError = errorPipe
    
    do {
        try process.run()
        process.waitUntilExit()
        
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        
        print("Exit code: \(process.terminationStatus)")
        
        if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
            print("📝 Response:")
            print(output)
        }
        
        if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
            print("⚠️ Error:")
            print(error)
        }
        
    } catch {
        print("❌ Failed to send message: \(error)")
    }
}

// Test 4: Test streaming
func testStreaming() {
    print("\n🔍 Testing streaming...")
    
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["claude", "--stream", "Count to 5"]
    
    let outputPipe = Pipe()
    let errorPipe = Pipe()
    
    process.standardOutput = outputPipe
    process.standardError = errorPipe
    
    do {
        try process.run()
        
        // Try to read streaming output
        let outputHandle = outputPipe.fileHandleForReading
        outputHandle.readabilityHandler = { handle in
            let data = handle.availableData
            
            if data.isEmpty {
                print("\n📊 Process finished")
                return
            }
            
            if let output = String(data: data, encoding: .utf8) {
                print("📡 Streaming chunk: \(output.debugDescription)")
            }
        }
        
        process.waitUntilExit()
        
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        
        print("Exit code: \(process.terminationStatus)")
        
        if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
            print("⚠️ Error:")
            print(error)
        }
        
    } catch {
        print("❌ Failed to test streaming: \(error)")
    }
}

// Run all tests
if testClaudeExists() {
    testClaudeHelp()
    testSimpleMessage()
    testStreaming()
} else {
    print("❌ Claude command not found. Please install claude CLI first.")
    print("   Try: npm install -g @anthropics/claude-cli")
}

print("\n🏁 Tests complete!")