#!/usr/bin/env swift

import Foundation

// Debug script to examine our entry counting vs ccusage

print("🔍 Debugging token counting...")

// First, let's see what ccusage says about today specifically
let todayExpected = 64677475 // from ccusage
let todayActual = 101719018   // from our implementation
let difference = todayActual - todayExpected

print("Expected (ccusage): \(todayExpected)")
print("Actual (ours): \(todayActual)")
print("Difference: +\(difference) (+\(Int(Double(difference)/Double(todayExpected)*100))%)")
print("")

// Let's examine one JSONL file in detail
let claudeDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".claude")
let projectsDir = claudeDir.appendingPathComponent("projects")
let targetProject = projectsDir.appendingPathComponent("-Users-jimmyhmiller-Documents-Code-PlayGround-claude-experiments-macos")

do {
    let jsonlFiles = try FileManager.default.contentsOfDirectory(at: targetProject, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "jsonl" }
    
    if let firstFile = jsonlFiles.first {
        let content = try String(contentsOf: firstFile, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        
        print("📄 Analyzing file: \(firstFile.lastPathComponent)")
        print("Total lines: \(lines.count)")
        
        // Count different types of entries
        var userMessages = 0
        var assistantMessages = 0
        var assistantWithUsage = 0
        var todayMessages = 0
        var todayUsageMessages = 0
        
        // Track token sums
        var totalInputTokens = 0
        var totalOutputTokens = 0
        var totalCacheCreateTokens = 0
        var totalCacheReadTokens = 0
        
        for line in lines {
            if line.contains("\"type\":\"user\"") {
                userMessages += 1
            }
            
            if line.contains("\"type\":\"assistant\"") {
                assistantMessages += 1
                
                if line.contains("\"usage\":{") {
                    assistantWithUsage += 1
                    
                    // Extract token counts
                    if let inputMatch = line.range(of: "\"input_tokens\":(\\d+)", options: .regularExpression) {
                        let inputStr = String(line[inputMatch]).replacingOccurrences(of: "\"input_tokens\":", with: "")
                        if let tokens = Int(inputStr) {
                            totalInputTokens += tokens
                        }
                    }
                    
                    if let outputMatch = line.range(of: "\"output_tokens\":(\\d+)", options: .regularExpression) {
                        let outputStr = String(line[outputMatch]).replacingOccurrences(of: "\"output_tokens\":", with: "")
                        if let tokens = Int(outputStr) {
                            totalOutputTokens += tokens
                        }
                    }
                    
                    if let cacheCreateMatch = line.range(of: "\"cache_creation_input_tokens\":(\\d+)", options: .regularExpression) {
                        let cacheStr = String(line[cacheCreateMatch]).replacingOccurrences(of: "\"cache_creation_input_tokens\":", with: "")
                        if let tokens = Int(cacheStr) {
                            totalCacheCreateTokens += tokens
                        }
                    }
                    
                    if let cacheReadMatch = line.range(of: "\"cache_read_input_tokens\":(\\d+)", options: .regularExpression) {
                        let cacheStr = String(line[cacheReadMatch]).replacingOccurrences(of: "\"cache_read_input_tokens\":", with: "")
                        if let tokens = Int(cacheStr) {
                            totalCacheReadTokens += tokens
                        }
                    }
                }
            }
            
            if line.contains("2025-06-29") {
                todayMessages += 1
                if line.contains("\"usage\":{") && line.contains("\"type\":\"assistant\"") {
                    todayUsageMessages += 1
                }
            }
        }
        
        print("")
        print("📊 Message breakdown:")
        print("  User messages: \(userMessages)")
        print("  Assistant messages: \(assistantMessages)")
        print("  Assistant with usage: \(assistantWithUsage)")
        print("  Today's messages: \(todayMessages)")
        print("  Today's usage messages: \(todayUsageMessages)")
        
        print("")
        print("📊 Token breakdown (this file only):")
        print("  Input tokens: \(totalInputTokens)")
        print("  Output tokens: \(totalOutputTokens)")
        print("  Cache create tokens: \(totalCacheCreateTokens)")
        print("  Cache read tokens: \(totalCacheReadTokens)")
        print("  Total: \(totalInputTokens + totalOutputTokens + totalCacheCreateTokens + totalCacheReadTokens)")
        
        // Let's look at a few today entries
        print("")
        print("📊 Sample today entries:")
        let todayLines = lines.filter { $0.contains("2025-06-29") && $0.contains("\"usage\":{") }
        for (i, line) in todayLines.prefix(3).enumerated() {
            print("Entry \(i+1):")
            
            // Extract and show token counts
            var tokens: [String] = []
            if let inputMatch = line.range(of: "\"input_tokens\":(\\d+)", options: .regularExpression) {
                let inputStr = String(line[inputMatch]).replacingOccurrences(of: "\"input_tokens\":", with: "")
                tokens.append("input: \(inputStr)")
            }
            if let outputMatch = line.range(of: "\"output_tokens\":(\\d+)", options: .regularExpression) {
                let outputStr = String(line[outputMatch]).replacingOccurrences(of: "\"output_tokens\":", with: "")
                tokens.append("output: \(outputStr)")
            }
            if let cacheCreateMatch = line.range(of: "\"cache_creation_input_tokens\":(\\d+)", options: .regularExpression) {
                let cacheStr = String(line[cacheCreateMatch]).replacingOccurrences(of: "\"cache_creation_input_tokens\":", with: "")
                tokens.append("cache_create: \(cacheStr)")
            }
            if let cacheReadMatch = line.range(of: "\"cache_read_input_tokens\":(\\d+)", options: .regularExpression) {
                let cacheStr = String(line[cacheReadMatch]).replacingOccurrences(of: "\"cache_read_input_tokens\":", with: "")
                tokens.append("cache_read: \(cacheStr)")
            }
            
            print("  \(tokens.joined(separator: ", "))")
            
            // Check if there are multiple usage blocks in one line
            let usageCount = line.components(separatedBy: "\"usage\":{").count - 1
            if usageCount > 1 {
                print("  ⚠️  Multiple usage blocks in one line: \(usageCount)")
            }
        }
    }
    
} catch {
    print("Error: \(error)")
}