#!/usr/bin/env swift

import Foundation

// Debug script to check what's happening with Claude usage parsing
let claudeDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".claude")
let projectsDir = claudeDir.appendingPathComponent("projects")

print("Checking Claude directory: \(claudeDir.path)")
print("Projects directory exists: \(FileManager.default.fileExists(atPath: projectsDir.path))")

do {
    let projectDirs = try FileManager.default.contentsOfDirectory(at: projectsDir, includingPropertiesForKeys: nil)
    print("Found \(projectDirs.count) project directories")
    
    for projectDir in projectDirs.prefix(3) {
        print("\nProject: \(projectDir.lastPathComponent)")
        
        let jsonlFiles = try FileManager.default.contentsOfDirectory(at: projectDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "jsonl" }
        
        print("  JSONL files: \(jsonlFiles.count)")
        
        for jsonlFile in jsonlFiles.prefix(1) {
            print("    File: \(jsonlFile.lastPathComponent)")
            
            let content = try String(contentsOf: jsonlFile, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
            print("    Lines: \(lines.count)")
            
            var usageCount = 0
            var totalTokens = 0
            
            for (index, line) in lines.enumerated() {
                if line.contains("\"usage\":{") {
                    usageCount += 1
                    
                    // Extract token counts from the line
                    if let inputRange = line.range(of: "\"input_tokens\":(\\d+)", options: .regularExpression) {
                        let inputStr = String(line[inputRange]).replacingOccurrences(of: "\"input_tokens\":", with: "")
                        if let inputTokens = Int(inputStr) {
                            totalTokens += inputTokens
                        }
                    }
                    
                    if let outputRange = line.range(of: "\"output_tokens\":(\\d+)", options: .regularExpression) {
                        let outputStr = String(line[outputRange]).replacingOccurrences(of: "\"output_tokens\":", with: "")
                        if let outputTokens = Int(outputStr) {
                            totalTokens += outputTokens
                        }
                    }
                    
                    if index < 3 {
                        print("      Usage line \(index + 1): \(line.prefix(200))...")
                    }
                }
            }
            
            print("    Total usage entries: \(usageCount)")
            print("    Estimated total tokens: \(totalTokens)")
            
            // Check dates
            let today = Date()
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyy-MM-dd"
            let todayString = dateFormatter.string(from: today)
            print("    Today's date: \(todayString)")
            
            let todayEntries = lines.filter { $0.contains(todayString) }
            print("    Today's entries: \(todayEntries.count)")
        }
    }
} catch {
    print("Error: \(error)")
}