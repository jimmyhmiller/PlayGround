import Foundation

// Simple standalone tester for claude command
class ClaudeCommandTester {
    
    static func testClaudeAvailability() -> Bool {
        print("🔍 Testing if claude command is available...")
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = ["claude"]
        
        do {
            try process.run()
            process.waitUntilExit()
            let isAvailable = process.terminationStatus == 0
            print(isAvailable ? "✅ claude command found" : "❌ claude command not found")
            return isAvailable
        } catch {
            print("❌ Error checking claude availability: \(error)")
            return false
        }
    }
    
    static func testClaudeHelp() {
        print("🔍 Testing claude --help...")
        
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
            
            if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
                print("✅ Claude help output:")
                print(output)
            }
            
            if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
                print("⚠️ Claude help stderr:")
                print(error)
            }
            
            print("📊 Exit code: \(process.terminationStatus)")
            
        } catch {
            print("❌ Error running claude --help: \(error)")
        }
    }
    
    static func testSimpleClaudeMessage() {
        print("🔍 Testing simple claude message...")
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["claude", "Hello, please respond with just 'Hi there!'"]
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            
            if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
                print("✅ Claude response:")
                print("📝 Output: \(output)")
            } else {
                print("❌ No output from claude")
            }
            
            if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
                print("⚠️ Claude stderr:")
                print("🚨 Error: \(error)")
            }
            
            print("📊 Exit code: \(process.terminationStatus)")
            
        } catch {
            print("❌ Error running claude command: \(error)")
        }
    }
    
    static func testClaudeWithStreamFlag() {
        print("🔍 Testing claude with --stream flag...")
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["claude", "--stream", "Hello, please respond with just 'Hi there!'"]
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            
            if let output = String(data: outputData, encoding: .utf8), !output.isEmpty {
                print("✅ Claude streaming response:")
                print("📝 Output: \(output)")
            } else {
                print("❌ No output from claude --stream")
            }
            
            if let error = String(data: errorData, encoding: .utf8), !error.isEmpty {
                print("⚠️ Claude --stream stderr:")
                print("🚨 Error: \(error)")
            }
            
            print("📊 Exit code: \(process.terminationStatus)")
            
        } catch {
            print("❌ Error running claude --stream: \(error)")
        }
    }
    
    static func runAllTests() {
        print("🧪 Running Claude Command Tests")
        print("================================")
        
        testClaudeAvailability()
        print("")
        
        testClaudeHelp()
        print("")
        
        testSimpleClaudeMessage()
        print("")
        
        testClaudeWithStreamFlag()
        print("")
        
        print("🏁 Tests complete!")
    }
}