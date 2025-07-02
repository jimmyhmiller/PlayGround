#!/usr/bin/env swift

import Foundation

// Copy the exact ClaudeUsageReader code to test directly
// This will show us what our implementation actually produces

print("🔍 Testing our ClaudeUsageReader implementation...")

// Minimal test to see what we get
let claudeDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".claude")
let projectsDir = claudeDir.appendingPathComponent("projects")

do {
    let projectDirs = try FileManager.default.contentsOfDirectory(at: projectsDir, includingPropertiesForKeys: nil)
    print("Found \(projectDirs.count) project directories")
    
    // Just check one recent project 
    let recentProject = projectsDir.appendingPathComponent("-Users-jimmyhmiller-Documents-Code-PlayGround-claude-experiments-macos")
    
    if FileManager.default.fileExists(atPath: recentProject.path) {
        let jsonlFiles = try FileManager.default.contentsOfDirectory(at: recentProject, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "jsonl" }
        
        print("Found \(jsonlFiles.count) JSONL files in recent project")
        
        if let firstFile = jsonlFiles.first {
            let content = try String(contentsOf: firstFile, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
            
            print("File has \(lines.count) lines")
            
            // Count today's entries
            let todayStr = "2025-06-29"
            let todayLines = lines.filter { $0.contains(todayStr) && $0.contains("\"usage\":{") }
            
            print("Today's usage entries: \(todayLines.count)")
            
            if let firstTodayLine = todayLines.first {
                print("First today entry contains:")
                if firstTodayLine.contains("\"input_tokens\":") {
                    print("  - Has input_tokens ✓")
                }
                if firstTodayLine.contains("\"output_tokens\":") {
                    print("  - Has output_tokens ✓")
                }
                if firstTodayLine.contains("\"cache_creation_input_tokens\":") {
                    print("  - Has cache_creation_input_tokens ✓")
                }
                if firstTodayLine.contains("\"cache_read_input_tokens\":") {
                    print("  - Has cache_read_input_tokens ✓")
                }
                if firstTodayLine.contains("\"service_tier\":") {
                    print("  - Has service_tier ✓")
                }
            }
        }
    }
    
} catch {
    print("Error: \(error)")
}