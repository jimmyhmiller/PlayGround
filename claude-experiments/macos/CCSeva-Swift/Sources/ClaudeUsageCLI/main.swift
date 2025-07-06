//
//  main.swift
//  ClaudeUsageCLI
//
//  Command line tool to test and verify Claude usage calculations
//  Compare with ccusage CLI tool output
//

import Foundation
import ClaudeUsageCore

func printUsage() {
    print("Claude Usage CLI Tool")
    print("Usage: claude-usage [command]")
    print("")
    print("Commands:")
    print("  stats     Show current usage statistics")
    print("  today     Show today's usage details")
    print("  compare   Compare with ccusage output")
    print("  debug     Show debug information")
    print("  help      Show this help message")
}

func showStats() throws {
    let reader = ClaudeUsageReader()
    let stats = try reader.generateUsageStats()
    
    print("=== Claude Usage Statistics ===")
    print("")
    print("Plan: \(stats.currentPlan)")
    print("Token Limit: \(formatNumber(stats.tokenLimit))")
    print("Tokens Used: \(formatNumber(stats.tokensUsed))")
    print("Tokens Remaining: \(formatNumber(stats.tokensRemaining))")
    print("Percentage Used: \(String(format: "%.1f", stats.percentageUsed))%")
    print("")
    print("=== Today's Usage ===")
    print("Date: \(stats.today.date)")
    print("Total Tokens: \(formatNumber(stats.today.totalTokens))")
    print("Total Cost: $\(String(format: "%.2f", stats.today.totalCost))")
    print("")
    print("Models Used:")
    for (model, usage) in stats.today.models {
        print("  \(model):")
        print("    Tokens: \(formatNumber(usage.tokens))")
        print("    Cost: $\(String(format: "%.2f", usage.cost))")
    }
    print("")
    print("=== Burn Rate & Velocity ===")
    print("Current Burn Rate: \(Int(stats.burnRate)) tokens/hour")
    print("24h Average: \(Int(stats.velocity.average24h)) tokens/hour")
    print("7d Average: \(Int(stats.velocity.average7d)) tokens/hour")
    print("Trend: \(stats.velocity.trend)")
    print("")
    print("=== Reset Information ===")
    print("Time Until Reset: \(stats.resetInfo.timeUntilReset)")
    print("Next Reset: \(stats.resetInfo.nextResetDate)")
    print("Timezone: \(stats.resetInfo.timezone)")
}

func showTodayDetails() throws {
    let reader = ClaudeUsageReader()
    let stats = try reader.generateUsageStats()
    
    print("=== Today's Detailed Usage ===")
    print("Date: \(stats.today.date)")
    print("Total Tokens: \(formatNumber(stats.today.totalTokens))")
    print("Total Cost: $\(String(format: "%.4f", stats.today.totalCost))")
    print("")
    
    for (model, usage) in stats.today.models.sorted(by: { $0.value.tokens > $1.value.tokens }) {
        print("\(model):")
        print("  Tokens: \(formatNumber(usage.tokens))")
        print("  Cost: $\(String(format: "%.4f", usage.cost))")
        print("")
    }
}

func compareWithCCUsage() throws {
    print("=== Comparison with ccusage ===")
    print("")
    
    // Run our implementation
    let reader = ClaudeUsageReader()
    let stats = try reader.generateUsageStats()
    
    print("Our Implementation:")
    print("  Today's Tokens: \(formatNumber(stats.today.totalTokens))")
    print("  Today's Cost: $\(String(format: "%.2f", stats.today.totalCost))")
    print("  Plan: \(stats.currentPlan)")
    print("")
    
    // Try to run ccusage
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/ccusage")
    process.arguments = ["--json"]
    
    let pipe = Pipe()
    process.standardOutput = pipe
    
    do {
        try process.run()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        if let output = String(data: data, encoding: .utf8),
           let jsonData = output.data(using: .utf8) {
            
            let decoder = JSONDecoder()
            let ccusageData = try decoder.decode(CCUsageResponse.self, from: jsonData)
            
            // Find today's data
            let today = DateFormatter().string(from: Date())
            if let todayData = ccusageData.daily.first(where: { $0.date.contains("2025-06-29") }) {
                print("ccusage CLI:")
                print("  Today's Tokens: \(formatNumber(todayData.totalTokens))")
                print("  Today's Cost: $\(String(format: "%.2f", todayData.totalCost))")
                print("")
                
                print("Difference:")
                let tokenDiff = stats.today.totalTokens - todayData.totalTokens
                let costDiff = stats.today.totalCost - todayData.totalCost
                print("  Tokens: \(tokenDiff >= 0 ? "+" : "")\(formatNumber(tokenDiff))")
                print("  Cost: $\(String(format: "%.2f", costDiff))")
            }
        }
    } catch {
        print("Could not run ccusage: \(error)")
        print("Make sure ccusage is installed: npm install -g ccusage")
    }
}

func showDebugInfo() throws {
    print("=== Debug Information ===")
    print("")
    
    let reader = ClaudeUsageReader()
    
    // This will print debug info due to our debug prints in the reader
    let startTime = CFAbsoluteTimeGetCurrent()
    let _ = try reader.generateUsageStats()
    let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
    
    print("")
    print("⏱️ Total processing time: \(String(format: "%.3f", timeElapsed))s")
    
    let cacheStats = reader.getCacheStats()
    print("💾 Cache stats: \(cacheStats.filesInCache) files cached, \(cacheStats.totalEntries) total entries")
}

func formatNumber(_ number: Int) -> String {
    let formatter = NumberFormatter()
    formatter.numberStyle = .decimal
    return formatter.string(from: NSNumber(value: number)) ?? "\(number)"
}

// MARK: - CCUsage JSON Models for comparison

struct CCUsageResponse: Codable {
    let daily: [CCUsageDailyData]
    let totals: CCUsageTotals
}

struct CCUsageDailyData: Codable {
    let date: String
    let inputTokens: Int
    let outputTokens: Int
    let cacheCreationTokens: Int
    let cacheReadTokens: Int
    let totalTokens: Int
    let totalCost: Double
    let modelsUsed: [String]
}

struct CCUsageTotals: Codable {
    let inputTokens: Int
    let outputTokens: Int
    let cacheCreationTokens: Int
    let cacheReadTokens: Int
    let totalTokens: Int
    let totalCost: Double
}

// MARK: - Main

let arguments = CommandLine.arguments

if arguments.count < 2 {
    printUsage()
    exit(1)
}

let command = arguments[1].lowercased()

do {
    switch command {
    case "stats":
        try showStats()
    case "today":
        try showTodayDetails()
    case "compare":
        try compareWithCCUsage()
    case "debug":
        try showDebugInfo()
    case "help", "--help", "-h":
        printUsage()
    default:
        print("Unknown command: \(command)")
        printUsage()
        exit(1)
    }
} catch {
    print("Error: \(error)")
    exit(1)
}