#!/usr/bin/env swift

import Foundation

print("🔍 Looking for duplicate entries...")

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
        
        // Find duplicate usage entries
        var usageSignatures: [String: Int] = [:]
        var duplicateCount = 0
        
        for (lineNum, line) in lines.enumerated() {
            if line.contains("\"usage\":{") && line.contains("\"type\":\"assistant\"") {
                // Extract usage signature (all token counts)
                var signature = ""
                
                if let inputMatch = line.range(of: "\"input_tokens\":(\\d+)", options: .regularExpression) {
                    signature += String(line[inputMatch])
                }
                if let outputMatch = line.range(of: "\"output_tokens\":(\\d+)", options: .regularExpression) {
                    signature += String(line[outputMatch])
                }
                if let cacheCreateMatch = line.range(of: "\"cache_creation_input_tokens\":(\\d+)", options: .regularExpression) {
                    signature += String(line[cacheCreateMatch])
                }
                if let cacheReadMatch = line.range(of: "\"cache_read_input_tokens\":(\\d+)", options: .regularExpression) {
                    signature += String(line[cacheReadMatch])
                }
                
                if let existingLine = usageSignatures[signature] {
                    print("🔴 DUPLICATE found!")
                    print("  Line \(existingLine) and line \(lineNum+1) have identical usage: \(signature)")
                    duplicateCount += 1
                    
                    if duplicateCount <= 5 { // Show first few duplicates
                        // Check if they have different UUIDs
                        if let uuid1 = lines[existingLine-1].range(of: "\"uuid\":\"([^\"]+)\"", options: .regularExpression) {
                            let uuid1Str = String(lines[existingLine-1][uuid1])
                            if let uuid2 = line.range(of: "\"uuid\":\"([^\"]+)\"", options: .regularExpression) {
                                let uuid2Str = String(line[uuid2])
                                print("  UUID1: \(uuid1Str)")
                                print("  UUID2: \(uuid2Str)")
                                print("  Same UUID: \(uuid1Str == uuid2Str)")
                            }
                        }
                        
                        // Check timestamps
                        if let ts1 = lines[existingLine-1].range(of: "\"timestamp\":\"([^\"]+)\"", options: .regularExpression) {
                            let ts1Str = String(lines[existingLine-1][ts1])
                            if let ts2 = line.range(of: "\"timestamp\":\"([^\"]+)\"", options: .regularExpression) {
                                let ts2Str = String(line[ts2])
                                print("  Timestamp1: \(ts1Str)")
                                print("  Timestamp2: \(ts2Str)")
                                print("  Same timestamp: \(ts1Str == ts2Str)")
                            }
                        }
                        print("")
                    }
                } else {
                    usageSignatures[signature] = lineNum + 1
                }
            }
        }
        
        print("📊 Results:")
        print("  Total usage entries: \(usageSignatures.count + duplicateCount)")
        print("  Unique usage signatures: \(usageSignatures.count)")
        print("  Duplicates found: \(duplicateCount)")
        
        if duplicateCount > 0 {
            print("⚠️  We're likely double-counting \(duplicateCount) entries!")
            print("This could explain the token count discrepancy.")
        }
    }
    
} catch {
    print("Error: \(error)")
}