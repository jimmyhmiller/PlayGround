#!/usr/bin/env swift

import Foundation

// Test script to run our usage reader standalone
class TestUsageReader {
    private let claudeDirectory: URL
    private let calendar = Calendar.current
    private let dateFormatter: DateFormatter
    
    init() {
        self.claudeDirectory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".claude")
        self.dateFormatter = DateFormatter()
        self.dateFormatter.dateFormat = "yyyy-MM-dd"
    }
    
    func testUsageReading() throws {
        print("Testing Claude usage reading...")
        
        let projectsDir = claudeDirectory.appendingPathComponent("projects")
        let projectDirs = try FileManager.default.contentsOfDirectory(at: projectsDir, includingPropertiesForKeys: nil)
        
        var totalTokens = 0
        var totalCost = 0.0
        var entryCount = 0
        
        for projectDir in projectDirs.prefix(3) {
            print("Project: \(projectDir.lastPathComponent)")
            
            let jsonlFiles = try FileManager.default.contentsOfDirectory(at: projectDir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "jsonl" }
            
            for jsonlFile in jsonlFiles.prefix(1) {
                let content = try String(contentsOf: jsonlFile, encoding: .utf8)
                let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
                
                for line in lines {
                    if line.contains("\"usage\":{") && line.contains("\"type\":\"assistant\"") {
                        // Parse basic usage from line
                        var lineTokens = 0
                        if let inputMatch = line.range(of: "\"input_tokens\":(\\d+)", options: .regularExpression) {
                            let inputStr = String(line[inputMatch]).replacingOccurrences(of: "\"input_tokens\":", with: "")
                            if let inputTokens = Int(inputStr) {
                                lineTokens += inputTokens
                            }
                        }
                        
                        if let outputMatch = line.range(of: "\"output_tokens\":(\\d+)", options: .regularExpression) {
                            let outputStr = String(line[outputMatch]).replacingOccurrences(of: "\"output_tokens\":", with: "")
                            if let outputTokens = Int(outputStr) {
                                lineTokens += outputTokens
                            }
                        }
                        
                        if let cacheCreateMatch = line.range(of: "\"cache_creation_input_tokens\":(\\d+)", options: .regularExpression) {
                            let cacheStr = String(line[cacheCreateMatch]).replacingOccurrences(of: "\"cache_creation_input_tokens\":", with: "")
                            if let cacheTokens = Int(cacheStr) {
                                lineTokens += cacheTokens
                            }
                        }
                        
                        if let cacheReadMatch = line.range(of: "\"cache_read_input_tokens\":(\\d+)", options: .regularExpression) {
                            let cacheStr = String(line[cacheReadMatch]).replacingOccurrences(of: "\"cache_read_input_tokens\":", with: "")
                            if let cacheTokens = Int(cacheStr) {
                                lineTokens += cacheTokens
                            }
                        }
                        
                        if lineTokens > 0 {
                            totalTokens += lineTokens
                            entryCount += 1
                        }
                    }
                }
            }
        }
        
        print("📊 Our implementation found:")
        print("   Total entries: \(entryCount)")
        print("   Total tokens (all types): \(totalTokens)")
        print("   Total cost: $\(String(format: "%.2f", totalCost))")
        
        print("\n📊 ccusage reports:")
        print("   Total tokens (including cache): 624,764,844")
        print("   Total cost: $367.38")
        print("   Today (June 29): 55,102,492 tokens, $52.69")
    }
}

do {
    let reader = TestUsageReader()
    try reader.testUsageReading()
} catch {
    print("Error: \(error)")
}