//
//  ClaudeUsageReader.swift
//  CCSeva
//
//  Native Swift implementation for reading Claude usage data
//  Deduplication logic derived from ccusage (https://github.com/ryoppippi/ccusage)
//  Copyright (c) 2025 ryoppippi - MIT License
//

import Foundation

// MARK: - Claude JSONL Data Models
struct ClaudeMessage: Codable {
    let type: String
    let message: MessageContent?
    let timestamp: String?
    let sessionId: String?
    let uuid: String?
    let model: String?
    let requestId: String?
    let costUSD: Double?  // Optional - missing for subscription plans
    
    private enum CodingKeys: String, CodingKey {
        case type, message, timestamp, sessionId, uuid, model, requestId, costUSD
    }
}

struct MessageContent: Codable {
    let model: String?
    let usage: TokenUsage?
    let role: String?
    let id: String?
    
    private enum CodingKeys: String, CodingKey {
        case model, usage, role, id
    }
}

struct TokenUsage: Codable {
    let inputTokens: Int?
    let outputTokens: Int?
    let cacheCreationInputTokens: Int?
    let cacheReadInputTokens: Int?
    let serviceTier: String?
    
    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case cacheCreationInputTokens = "cache_creation_input_tokens"
        case cacheReadInputTokens = "cache_read_input_tokens"
        case serviceTier = "service_tier"
    }
}

// MARK: - Usage Aggregation
struct UsageEntry {
    let timestamp: Date
    let model: String
    let inputTokens: Int
    let outputTokens: Int
    let cacheCreationTokens: Int
    let cacheReadTokens: Int
    let totalTokens: Int
    let cost: Double  // API-equivalent cost (always calculated)
    let actualCostUSD: Double?  // Actual cost from JSONL (missing for subscription plans)
    
    init(message: ClaudeMessage) {
        // Parse timestamp
        if let timestampStr = message.timestamp {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            self.timestamp = formatter.date(from: timestampStr) ?? Date()
        } else {
            self.timestamp = Date()
        }
        
        // Extract model and usage
        self.model = message.message?.model ?? "unknown"
        let usage = message.message?.usage
        let serviceTier = usage?.serviceTier ?? "unknown"
        
        self.inputTokens = usage?.inputTokens ?? 0
        self.outputTokens = usage?.outputTokens ?? 0
        self.cacheCreationTokens = usage?.cacheCreationInputTokens ?? 0
        self.cacheReadTokens = usage?.cacheReadInputTokens ?? 0
        
        // Total tokens should include all token types to match ccusage behavior
        self.totalTokens = inputTokens + outputTokens + cacheCreationTokens + cacheReadTokens
        
        // Store actual cost from JSONL data (if present)
        self.actualCostUSD = message.costUSD
        
        // Always calculate API-equivalent cost for comparison (like ccusage)
        self.cost = UsageEntry.calculateCost(
            model: self.model,
            inputTokens: inputTokens,
            outputTokens: outputTokens,
            cacheCreationTokens: cacheCreationTokens,
            cacheReadTokens: cacheReadTokens,
            serviceTier: serviceTier
        )
    }
    
    private static func calculateCost(model: String, inputTokens: Int, outputTokens: Int, cacheCreationTokens: Int, cacheReadTokens: Int, serviceTier: String) -> Double {
        // Calculate actual costs based on Claude's pricing
        let pricing: (input: Double, output: Double, cacheCreation: Double, cacheRead: Double)
        
        switch model {
        case let m where m.contains("opus"):
            pricing = (15.0, 75.0, 18.75, 1.50)  // Opus pricing
        case let m where m.contains("sonnet"):
            pricing = (3.0, 15.0, 3.75, 0.30)   // Sonnet pricing
        case let m where m.contains("haiku"):
            pricing = (0.25, 1.25, 0.30, 0.03)  // Haiku pricing
        default:
            pricing = (3.0, 15.0, 3.75, 0.30)   // Default to Sonnet pricing
        }
        
        let inputCost = (Double(inputTokens) / 1_000_000) * pricing.input
        let outputCost = (Double(outputTokens) / 1_000_000) * pricing.output
        let cacheCreationCost = (Double(cacheCreationTokens) / 1_000_000) * pricing.cacheCreation
        let cacheReadCost = (Double(cacheReadTokens) / 1_000_000) * pricing.cacheRead
        
        return inputCost + outputCost + cacheCreationCost + cacheReadCost
    }
}

// MARK: - File Cache
struct FileCache {
    var entries: [UsageEntry]
    var modificationDate: Date
    var fileSize: Int64
}

// MARK: - Claude Usage Reader
public class ClaudeUsageReader {
    private let claudeDirectory: URL
    private let calendar = Calendar.current
    private let dateFormatter: DateFormatter
    
    // File caching for incremental parsing
    private var fileCache: [String: FileCache] = [:]
    private let cacheQueue = DispatchQueue(label: "com.ccseva.cache", qos: .utility)
    
    public init() {
        self.claudeDirectory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".claude")
        self.dateFormatter = DateFormatter()
        self.dateFormatter.dateFormat = "yyyy-MM-dd"
    }
    
    func readUsageData() throws -> [UsageEntry] {
        guard FileManager.default.fileExists(atPath: claudeDirectory.path) else {
            throw ClaudeUsageError.claudeDirectoryNotFound
        }
        
        let projectsDir = claudeDirectory.appendingPathComponent("projects")
        guard FileManager.default.fileExists(atPath: projectsDir.path) else {
            throw ClaudeUsageError.noProjectsFound
        }
        
        var allEntries: [UsageEntry] = []
        let fileManager = FileManager.default
        
        // Scan all JSONL files in projects directory
        let projectDirs = try fileManager.contentsOfDirectory(at: projectsDir, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey])
        
        for projectDir in projectDirs {
            guard projectDir.hasDirectoryPath else { continue }
            
            let jsonlFiles = try fileManager.contentsOfDirectory(
                at: projectDir, 
                includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]
            ).filter { $0.pathExtension == "jsonl" }
            
            for jsonlFile in jsonlFiles {
                let entries = try parseJSONLFileWithCache(jsonlFile)
                allEntries.append(contentsOf: entries)
            }
        }
        
        // Sort by timestamp
        return allEntries.sorted { $0.timestamp < $1.timestamp }
    }
    
    private func parseJSONLFileWithCache(_ fileURL: URL) throws -> [UsageEntry] {
        let filePath = fileURL.path
        let fileManager = FileManager.default
        
        // Get file attributes for modification date and size
        let attributes = try fileManager.attributesOfItem(atPath: filePath)
        guard let modificationDate = attributes[.modificationDate] as? Date,
              let fileSize = attributes[.size] as? Int64 else {
            print("⚠️ Could not get file attributes for \(filePath), falling back to direct parsing")
            return try parseJSONLFile(fileURL)
        }
        
        // Check cache in a thread-safe manner
        let cachedEntries: [UsageEntry]? = cacheQueue.sync {
            if let cached = fileCache[filePath],
               cached.modificationDate == modificationDate,
               cached.fileSize == fileSize {
                // File hasn't changed, return cached entries
                print("✅ Using cached data for \(fileURL.lastPathComponent)")
                return cached.entries
            }
            return nil
        }
        
        if let entries = cachedEntries {
            return entries
        }
        
        // File has changed or not cached, parse it fresh
        print("🔄 Parsing file \(fileURL.lastPathComponent) (modified: \(modificationDate))")
        let entries = try parseJSONLFile(fileURL)
        
        // Cache the results in a thread-safe manner
        cacheQueue.sync {
            fileCache[filePath] = FileCache(
                entries: entries,
                modificationDate: modificationDate,
                fileSize: fileSize
            )
        }
        
        return entries
    }
    
    private func parseJSONLFile(_ fileURL: URL) throws -> [UsageEntry] {
        let content = try String(contentsOf: fileURL, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        
        var entries: [UsageEntry] = []
        let decoder = JSONDecoder()
        
        // Track processed message+request combinations for deduplication
        // This approach is derived from ccusage: https://github.com/ryoppippi/ccusage
        var processedHashes: Set<String> = []
        
        for line in lines {
            guard let data = line.data(using: .utf8) else { continue }
            
            do {
                let message = try decoder.decode(ClaudeMessage.self, from: data)
                
                // Only process assistant messages with usage data
                if message.type == "assistant",
                   let messageContent = message.message,
                   let usage = messageContent.usage,
                   (usage.inputTokens ?? 0) > 0 || (usage.outputTokens ?? 0) > 0 {
                    
                    // Create unique hash like ccusage does: messageId:requestId
                    let uniqueHash = createUniqueHash(message: message)
                    
                    // Skip duplicates using the same logic as ccusage
                    if let hash = uniqueHash, processedHashes.contains(hash) {
                        continue // Skip duplicate message
                    }
                    
                    // Mark this combination as processed
                    if let hash = uniqueHash {
                        processedHashes.insert(hash)
                    }
                    
                    let entry = UsageEntry(message: message)
                    entries.append(entry)
                }
            } catch {
                // Skip malformed lines (ccusage also does this)
                continue
            }
        }
        
        return entries
    }
    
    /// Create unique hash for deduplication using message ID and request ID
    /// This logic is derived from ccusage: https://github.com/ryoppippi/ccusage
    /// See: ccusage/src/data-loader.ts - createUniqueHash function
    private func createUniqueHash(message: ClaudeMessage) -> String? {
        guard let messageId = message.message?.id,
              let requestId = message.requestId else {
            return nil
        }
        
        // Create hash using simple concatenation (same as ccusage)
        return "\(messageId):\(requestId)"
    }
    
    public func generateUsageStats() throws -> UsageStats {
        let entries = try readUsageData()
        print("📊 ClaudeUsageReader found \(entries.count) total entries")
        
        // Get current date and calculate date ranges (use local timezone)
        let now = Date()
        var localCalendar = Calendar.current
        localCalendar.timeZone = TimeZone.current
        let today = localCalendar.startOfDay(for: now)
        let weekAgo = localCalendar.date(byAdding: .day, value: -7, to: today)!
        let monthAgo = localCalendar.date(byAdding: .day, value: -30, to: today)!
        
        print("📅 Today date range: \(today)")
        print("📅 Week ago: \(weekAgo)")
        print("📅 Month ago: \(monthAgo)")
        
        // Filter entries by date ranges
        let todayEntries = entries.filter { localCalendar.isDate($0.timestamp, inSameDayAs: today) }
        let weekEntries = entries.filter { $0.timestamp >= weekAgo }
        let monthEntries = entries.filter { $0.timestamp >= monthAgo }
        
        print("📊 Filtered entries - Today: \(todayEntries.count), Week: \(weekEntries.count), Month: \(monthEntries.count)")
        
        if !todayEntries.isEmpty {
            print("📊 Today's first entry: \(todayEntries.first?.timestamp ?? Date())")
            print("📊 Today's token total: \(todayEntries.reduce(0) { $0 + $1.totalTokens })")
        }
        
        // Calculate today's usage
        let todayStats = calculateDailyUsage(entries: todayEntries, date: dateFormatter.string(from: today))
        
        // Calculate weekly usage
        let weeklyStats = generateDailyBreakdown(entries: weekEntries, days: 7)
        
        // Calculate monthly usage
        let monthlyStats = generateDailyBreakdown(entries: monthEntries, days: 30)
        
        // Calculate current usage totals for plan detection
        let currentPeriodTotal = todayEntries.reduce(0) { $0 + $1.totalTokens }
        
        // Detect plan and limits
        let (plan, tokenLimit) = detectPlanAndLimit(currentUsage: currentPeriodTotal, allEntries: entries)
        
        // Calculate burn rate and other metrics
        let burnRate = calculateBurnRate(entries: entries)
        let velocity = calculateVelocity(entries: entries, burnRate: burnRate)
        let resetInfo = calculateResetInfo()
        let prediction = calculatePrediction(
            tokensUsed: currentPeriodTotal,
            tokenLimit: tokenLimit,
            velocity: velocity,
            resetInfo: resetInfo
        )
        
        return UsageStats(
            today: todayStats,
            thisWeek: weeklyStats,
            thisMonth: monthlyStats,
            burnRate: burnRate,
            velocity: velocity,
            prediction: prediction,
            resetInfo: resetInfo,
            predictedDepleted: prediction.depletionTime,
            currentPlan: plan,
            tokenLimit: tokenLimit,
            tokensUsed: currentPeriodTotal,
            tokensRemaining: max(0, tokenLimit - currentPeriodTotal),
            percentageUsed: min(100.0, Double(currentPeriodTotal) / Double(tokenLimit) * 100.0)
        )
    }
    
    // MARK: - Helper Methods
    
    private func calculateDailyUsage(entries: [UsageEntry], date: String) -> DailyUsage {
        let totalTokens = entries.reduce(0) { $0 + $1.totalTokens }
        let totalCost = entries.reduce(0.0) { $0 + $1.cost }
        
        // Group by model
        var modelUsage: [String: ModelUsage] = [:]
        for entry in entries {
            let existing = modelUsage[entry.model] ?? ModelUsage(tokens: 0, cost: 0.0)
            modelUsage[entry.model] = ModelUsage(
                tokens: existing.tokens + entry.totalTokens,
                cost: existing.cost + entry.cost
            )
        }
        
        return DailyUsage(
            date: date,
            totalTokens: totalTokens,
            totalCost: totalCost,
            models: modelUsage
        )
    }
    
    private func generateDailyBreakdown(entries: [UsageEntry], days: Int) -> [DailyUsage] {
        var dailyStats: [DailyUsage] = []
        let today = Date()
        
        for i in 0..<days {
            let date = calendar.date(byAdding: .day, value: -i, to: today)!
            let dateString = dateFormatter.string(from: date)
            let dayEntries = entries.filter { calendar.isDate($0.timestamp, inSameDayAs: date) }
            
            dailyStats.append(calculateDailyUsage(entries: dayEntries, date: dateString))
        }
        
        return dailyStats.reversed()
    }
    
    private func detectPlanAndLimit(currentUsage: Int, allEntries: [UsageEntry]) -> (String, Int) {
        // Check if we have any entries with actual costs (indicates API plan vs subscription)
        let entriesWithActualCosts = allEntries.filter { $0.actualCostUSD != nil && $0.actualCostUSD! > 0 }
        let hasActualCosts = !entriesWithActualCosts.isEmpty
        
        // Calculate API-equivalent costs for comparison
        let calculatedCost = allEntries.reduce(0.0) { $0 + $1.cost }
        let totalTokens = allEntries.reduce(0) { $0 + $1.totalTokens }
        
        print("📊 Plan detection - Total tokens: \(totalTokens), API equivalent cost: $\(String(format: "%.4f", calculatedCost))")
        print("📊 Has actual costUSD entries: \(hasActualCosts)")
        
        // Find the highest daily usage in recent history 
        let recentDays = calendar.date(byAdding: .day, value: -30, to: Date())!
        let recentEntries = allEntries.filter { $0.timestamp >= recentDays }
        
        var dailyTotals: [Int] = []
        for i in 0..<30 {
            let date = calendar.date(byAdding: .day, value: -i, to: Date())!
            let dayTotal = recentEntries
                .filter { calendar.isDate($0.timestamp, inSameDayAs: date) }
                .reduce(0) { $0 + $1.totalTokens }
            dailyTotals.append(dayTotal)
        }
        
        let maxDailyUsage = dailyTotals.max() ?? currentUsage
        print("📊 Max daily usage in last 30 days: \(maxDailyUsage) tokens")
        
        // Claude subscription plans don't have hard daily limits
        // They use session-based rate limiting (5-hour blocks with dynamic limits)
        // The "percentage" is relative to typical session usage, not a hard cap
        
        if !hasActualCosts {
            // Subscription plan - no hard daily limits, just session rate limiting
            // Use historical max as reference for percentage (not a hard limit)
            if maxDailyUsage > 50000000 {
                return ("Max", maxDailyUsage)    // Use historical max for percentage context
            } else if maxDailyUsage > 10000000 {
                return ("Pro", maxDailyUsage)    // Use historical max for percentage context  
            } else {
                return ("Free", max(10000000, maxDailyUsage))   // At least 10M for context
            }
        } else {
            // API plan - has actual costs and may have real limits
            if maxDailyUsage > 100000000 {
                return ("API", 500000000)    // High-volume API usage
            } else {
                return ("API", 100000000)    // Standard API usage
            }
        }
    }
    
    private func calculateBurnRate(entries: [UsageEntry]) -> Double {
        let hourAgo = Date().addingTimeInterval(-3600)
        let recentEntries = entries.filter { $0.timestamp >= hourAgo }
        let recentTokens = recentEntries.reduce(0) { $0 + $1.totalTokens }
        return Double(recentTokens) // tokens per hour
    }
    
    private func calculateVelocity(entries: [UsageEntry], burnRate: Double) -> VelocityInfo {
        let now = Date()
        let dayAgo = now.addingTimeInterval(-24 * 3600)
        let weekAgo = now.addingTimeInterval(-7 * 24 * 3600)
        
        let last24h = entries.filter { $0.timestamp >= dayAgo }.reduce(0) { $0 + $1.totalTokens }
        let last7d = entries.filter { $0.timestamp >= weekAgo }.reduce(0) { $0 + $1.totalTokens }
        
        let avg24h = Double(last24h) / 24.0
        let avg7d = Double(last7d) / (7.0 * 24.0)
        
        let trendPercent = avg24h > 0 ? ((burnRate - avg24h) / avg24h) * 100 : 0
        let trend: String
        if abs(trendPercent) > 15 {
            trend = trendPercent > 0 ? "increasing" : "decreasing"
        } else {
            trend = "stable"
        }
        
        return VelocityInfo(
            current: burnRate,
            average24h: avg24h,
            average7d: avg7d,
            trend: trend,
            trendPercent: trendPercent,
            peakHour: 14, // Simplified
            isAccelerating: trend == "increasing" && trendPercent > 20
        )
    }
    
    private func calculateResetInfo() -> ResetTimeInfo {
        // Claude resets at midnight PST/PDT, but show time in user's timezone
        let claudeTimeZone = TimeZone(identifier: "America/Los_Angeles")!
        let userTimeZone = TimeZone.current
        
        // Calculate reset time in Claude's timezone (PST/PDT)
        var claudeCalendar = Calendar.current
        claudeCalendar.timeZone = claudeTimeZone
        
        let now = Date()
        let tomorrow = claudeCalendar.date(byAdding: .day, value: 1, to: claudeCalendar.startOfDay(for: now))!
        let timeUntilReset = tomorrow.timeIntervalSince(now)
        let hoursUntilReset = timeUntilReset / 3600
        
        // Format reset time in user's timezone - just show the time it happens
        let userFormatter = DateFormatter()
        userFormatter.timeZone = userTimeZone
        userFormatter.dateFormat = "h:mm a"  // e.g., "3:00 AM"
        
        let resetTimeString = userFormatter.string(from: tomorrow)
        
        return ResetTimeInfo(
            timeUntilReset: String(format: "%.0fh %.0fm", floor(hoursUntilReset), (hoursUntilReset.truncatingRemainder(dividingBy: 1) * 60)),
            nextResetDate: resetTimeString,
            hoursUntilReset: hoursUntilReset,
            timezone: resetTimeString  // Use the reset time as the "timezone" field
        )
    }
    
    private func calculatePrediction(tokensUsed: Int, tokenLimit: Int, velocity: VelocityInfo, resetInfo: ResetTimeInfo) -> PredictionInfo {
        let tokensRemaining = max(0, tokenLimit - tokensUsed)
        
        // For daily reset plans (like Claude Max), predictions should be based on time until reset
        // Not when you'll hit your total limit (since it resets daily)
        
        let hoursUntilReset = resetInfo.hoursUntilReset
        let daysRemaining = hoursUntilReset / 24.0
        
        // Calculate if you'll exceed limit before reset using current pace
        let depletionTime: String?
        let willExceedBeforeReset: Bool
        
        if velocity.average24h > 0 {
            // Use 24h average for more stable prediction than current burn rate
            let projectedTokensAtReset = Double(tokensUsed) + (velocity.average24h * hoursUntilReset)
            willExceedBeforeReset = projectedTokensAtReset > Double(tokenLimit)
            
            if willExceedBeforeReset {
                // Calculate when you'd hit the limit
                let hoursToLimit = Double(tokensRemaining) / velocity.average24h
                depletionTime = ISO8601DateFormatter().string(from: Date().addingTimeInterval(hoursToLimit * 3600))
            } else {
                depletionTime = nil
            }
        } else {
            willExceedBeforeReset = false
            depletionTime = nil
        }
        
        // Confidence based on data stability
        let confidence = velocity.average24h > 0 && velocity.average7d > 0 ? 75 : 50
        
        // Recommended pace to avoid hitting limit before reset
        let recommendedHourlyLimit = Double(tokensRemaining) / max(1.0, hoursUntilReset)
        let recommendedDailyLimit = Int(recommendedHourlyLimit * 24.0)
        
        // You're on track if current usage rate won't exceed limit before reset
        let onTrackForReset = !willExceedBeforeReset
        
        return PredictionInfo(
            depletionTime: depletionTime,
            confidence: confidence,
            daysRemaining: daysRemaining, // Hours until reset converted to days
            recommendedDailyLimit: recommendedDailyLimit,
            onTrackForReset: onTrackForReset
        )
    }
    
    // MARK: - Cache Management
    
    /// Clear all cached file data - useful for testing or when cache becomes stale
    public func clearCache() {
        cacheQueue.sync {
            fileCache.removeAll()
            print("🗑️ File cache cleared")
        }
    }
    
    /// Get cache statistics for debugging
    public func getCacheStats() -> (filesInCache: Int, totalEntries: Int) {
        return cacheQueue.sync {
            let totalEntries = fileCache.values.reduce(0) { $0 + $1.entries.count }
            return (filesInCache: fileCache.count, totalEntries: totalEntries)
        }
    }
}

// MARK: - Errors
enum ClaudeUsageError: Error, LocalizedError {
    case claudeDirectoryNotFound
    case noProjectsFound
    case fileReadError(String)
    
    var errorDescription: String? {
        switch self {
        case .claudeDirectoryNotFound:
            return "Claude directory not found at ~/.claude"
        case .noProjectsFound:
            return "No Claude projects found"
        case .fileReadError(let message):
            return "Failed to read usage files: \(message)"
        }
    }
}