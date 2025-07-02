import Foundation

// Simple test runner that can be called from the main app
struct TestRunner {
    
    static func runClaudeTests() {
        print("\n" + String(repeating: "=", count: 50))
        print("🧪 CLAUDE COMMAND TESTS")
        print(String(repeating: "=", count: 50))
        
        ClaudeCommandTester.runAllTests()
        
        print(String(repeating: "=", count: 50))
        print("🔚 END TESTS")
        print(String(repeating: "=", count: 50) + "\n")
    }
    
    static func testBasicProcess() {
        print("🔍 Testing basic process execution...")
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/echo")
        process.arguments = ["Hello from Process!"]
        
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                print("✅ Basic process test successful: \(output.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
        } catch {
            print("❌ Basic process test failed: \(error)")
        }
    }
}