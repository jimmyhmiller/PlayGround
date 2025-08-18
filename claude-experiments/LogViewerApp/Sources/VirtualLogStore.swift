// Virtual log store that provides on-demand access to log entries without loading entire file
// Based on HexFiend's lazy loading architecture

import Foundation
import SwiftUI

class VirtualLogStore: ObservableObject {
    internal let lineIndex: LogLineIndex
    internal let slice: LogSlice
    private var parsedEntries: [Int: LogEntry] = [:] // Cache parsed entries
    private let dateFormatter: DateFormatter
    private let isoFormatter: ISO8601DateFormatter
    
    @Published var selectedIndex: Int? = nil
    @Published var isInitializing = true
    @Published var estimatedLineCount: Int = 0
    
    private var cachedTimeRange: (start: Date, end: Date)?
    
    init?(url: URL) {
        guard let fileRef = LogFileReference(url: url) else { return nil }
        
        self.slice = LogFileSlice(fileRef: fileRef)
        self.lineIndex = LogLineIndex(slice: slice)
        
        // Initialize formatters
        self.dateFormatter = DateFormatter()
        self.dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        
        self.isoFormatter = ISO8601DateFormatter()
        self.isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        
        // Do minimal sync initialization - just estimate
        self.estimatedLineCount = estimateQuickLineCount()
        self.isInitializing = false // Start ready immediately
        
        // Start background initialization for better estimates
        Task {
            await refineEstimatesInBackground()
        }
    }
    
    /// Get total number of lines in the file (returns estimate during initialization)
    var totalLines: Int {
        // Always use estimate if available, never force full indexing
        if estimatedLineCount > 0 {
            return estimatedLineCount
        }
        // Fallback to current indexed count, but don't trigger full indexing
        return lineIndex.currentLineCount()
    }
    
    /// Quick synchronous estimate (very fast)
    private func estimateQuickLineCount() -> Int {
        // Quick estimate based on file size and typical line length
        let avgLineLength: Double = 80 // Reasonable estimate
        return max(1, Int(Double(slice.length) / avgLineLength))
    }
    
    /// Background refinement of estimates
    @MainActor
    private func refineEstimatesInBackground() async {
        await withTaskGroup(of: Void.self) { group in
            // Task 1: Better line count estimation from sampling
            group.addTask { [weak self] in
                await self?.estimateLineCount()
            }
            
            // Task 2: Parse first and last entries for time range
            group.addTask { [weak self] in
                await self?.loadTimeRange()
            }
            
            // Wait for both tasks
            await group.waitForAll()
        }
    }
    
    private func estimateLineCount() async {
        // Sample the first 1MB to estimate average line length
        let sampleSize = min(1024 * 1024, slice.length)
        guard let sampleData = slice.data(in: 0..<UInt64(sampleSize)) else { return }
        
        let newlineCount = sampleData.reduce(0) { count, byte in
            count + (byte == 0x0A ? 1 : 0)
        }
        
        if newlineCount > 0 {
            let avgLineLength = Double(sampleSize) / Double(newlineCount)
            let estimated = Int(Double(slice.length) / avgLineLength)
            
            await MainActor.run {
                self.estimatedLineCount = estimated
            }
        }
    }
    
    private func loadTimeRange() async {
        // Parse first entry
        guard let firstLineInfo = await lineIndex.lineInfoAsync(at: 0),
              let firstLineText = slice.substring(in: firstLineInfo.byteRange) else { return }
        
        let firstEntry = parseLogLine(firstLineText.trimmingCharacters(in: .whitespacesAndNewlines), lineNumber: 0)
        
        // For last entry, sample backwards to avoid full indexing
        let lastSampleOffset = max(0, Int64(slice.length) - 10000)
        guard let lastSampleData = slice.data(in: UInt64(lastSampleOffset)..<slice.length),
              let lastSampleText = String(data: lastSampleData, encoding: .utf8) else { return }
        
        // Find the last complete line in the sample
        let lines = lastSampleText.components(separatedBy: .newlines)
        for line in lines.reversed() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                let lastEntry = parseLogLine(trimmed, lineNumber: -1)
                
                await MainActor.run {
                    self.cachedTimeRange = (firstEntry.timestamp, lastEntry.timestamp)
                }
                break
            }
        }
    }
    
    /// Get log entry for a specific line number (non-blocking)
    func entry(at lineNumber: Int) -> LogEntry? {
        // Check cache first
        if let cached = parsedEntries[lineNumber] {
            return cached
        }
        
        // Get line info but don't force indexing
        guard let lineInfo = lineIndex.lineInfoNonBlocking(at: lineNumber),
              let lineText = slice.substring(in: lineInfo.byteRange) else {
            // Trigger background indexing for this area
            lineIndex.indexInBackgroundIfNeeded(around: lineNumber)
            return nil
        }
        
        let entry = parseLogLine(lineText.trimmingCharacters(in: .whitespacesAndNewlines), 
                                lineNumber: lineNumber)
        
        // Cache it
        parsedEntries[lineNumber] = entry
        return entry
    }
    
    /// Get multiple entries efficiently
    func entries(in range: Range<Int>) -> [LogEntry] {
        var result: [LogEntry] = []
        result.reserveCapacity(range.count)
        
        // Get line infos in batch
        let lineInfos = lineIndex.lines(in: range)
        
        for (index, lineInfo) in lineInfos.enumerated() {
            let lineNumber = range.lowerBound + index
            
            // Check cache first
            if let cached = parsedEntries[lineNumber] {
                result.append(cached)
                continue
            }
            
            // Parse new entry
            guard let lineText = slice.substring(in: lineInfo.byteRange) else { continue }
            
            let entry = parseLogLine(lineText.trimmingCharacters(in: .whitespacesAndNewlines),
                                   lineNumber: lineNumber)
            
            // Cache and add
            parsedEntries[lineNumber] = entry
            result.append(entry)
        }
        
        return result
    }
    
    /// Get the time range of the entire file (efficiently)
    func getTimeRange() -> (start: Date, end: Date)? {
        // Return cached time range if available
        if let cached = cachedTimeRange {
            return cached
        }
        
        // Force indexing for first and last entries to get time range
        guard let firstLineInfo = lineIndex.lineInfo(at: 0),
              let firstLineText = slice.substring(in: firstLineInfo.byteRange) else { 
            return nil 
        }
        
        let firstEntry = parseLogLine(firstLineText.trimmingCharacters(in: .whitespacesAndNewlines), lineNumber: 0)
        
        let lastLineIndex = totalLines - 1
        guard lastLineIndex > 0,
              let lastLineInfo = lineIndex.lineInfo(at: lastLineIndex),
              let lastLineText = slice.substring(in: lastLineInfo.byteRange) else {
            return (firstEntry.timestamp, firstEntry.timestamp)
        }
        
        let lastEntry = parseLogLine(lastLineText.trimmingCharacters(in: .whitespacesAndNewlines), lineNumber: lastLineIndex)
        
        let timeRange = (firstEntry.timestamp, lastEntry.timestamp)
        
        // Cache it for future use
        cachedTimeRange = timeRange
        
        return timeRange
    }
    
    /// Find entry closest to a given timestamp
    func findEntryNear(timestamp: Date) -> Int? {
        // Binary search approach - sample entries to find approximate location
        var left = 0
        var right = totalLines - 1
        
        while left < right {
            let mid = (left + right) / 2
            guard let midEntry = entry(at: mid) else { break }
            
            if midEntry.timestamp <= timestamp {
                left = mid + 1
            } else {
                right = mid
            }
        }
        
        return max(0, left - 1)
    }
    
    /// Jump to a specific time
    func jumpToTime(_ targetTime: Date) {
        selectedIndex = findEntryNear(timestamp: targetTime)
    }
    
    /// Get count of entries in a time range (for timeline)
    func getEntriesCount(in timeRange: Range<Date>) -> Int {
        // For now, sample the range - could be optimized with time indexing
        guard let overallRange = getTimeRange() else { return 0 }
        
        let totalDuration = overallRange.end.timeIntervalSince(overallRange.start)
        let startProgress = timeRange.lowerBound.timeIntervalSince(overallRange.start) / totalDuration
        let endProgress = timeRange.upperBound.timeIntervalSince(overallRange.start) / totalDuration
        
        let startLine = Int(Double(totalLines) * startProgress)
        let endLine = Int(Double(totalLines) * endProgress)
        
        return max(0, endLine - startLine)
    }
    
    internal func parseLogLine(_ line: String, lineNumber: Int) -> LogEntry {
        // Try JSON format first
        if line.starts(with: "{") {
            if let entry = parseJSONLog(line, lineNumber: lineNumber) {
                return entry
            }
        }
        
        // Try timestamped format
        if let entry = parseTimestampedLog(line, lineNumber: lineNumber) {
            return entry
        }
        
        // Fallback to simple format
        return LogEntry(timestamp: Date(), 
                       level: .info, 
                       message: line,
                       source: nil,
                       lineNumber: lineNumber)
    }
    
    private func parseJSONLog(_ line: String, lineNumber: Int) -> LogEntry? {
        guard let data = line.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        
        let timestamp = parseJSONTimestamp(json) ?? Date()
        let level = parseJSONLevel(json)
        let message = json["message"] as? String ?? json["msg"] as? String ?? line
        let source = json["source"] as? String ?? json["logger"] as? String
        
        return LogEntry(timestamp: timestamp, level: level, message: message, 
                       source: source, lineNumber: lineNumber)
    }
    
    private func parseJSONTimestamp(_ json: [String: Any]) -> Date? {
        if let timestampStr = json["timestamp"] as? String ?? 
                             json["time"] as? String ?? 
                             json["@timestamp"] as? String {
            return isoFormatter.date(from: timestampStr) ?? dateFormatter.date(from: timestampStr)
        }
        return nil
    }
    
    private func parseJSONLevel(_ json: [String: Any]) -> LogEntry.LogLevel {
        let levelStr = (json["level"] as? String ?? json["severity"] as? String ?? "info").uppercased()
        switch levelStr {
        case "ERROR", "ERR": return .error
        case "WARNING", "WARN": return .warning
        case "DEBUG": return .debug
        default: return .info
        }
    }
    
    private func parseTimestampedLog(_ line: String, lineNumber: Int) -> LogEntry? {
        // Common format: "2024-01-01 12:00:00.000 [LEVEL] message"
        let pattern = #"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?)(?:\s+\[(\w+)\])?\s+(.+)$"#
        
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) else {
            return nil
        }
        
        let timestampRange = Range(match.range(at: 1), in: line)!
        let levelRange = Range(match.range(at: 2), in: line)
        let messageRange = Range(match.range(at: 3), in: line)!
        
        let timestamp = dateFormatter.date(from: String(line[timestampRange])) ?? Date()
        let level = levelRange != nil ? parseLevelString(String(line[levelRange!])) : .info
        let message = String(line[messageRange])
        
        return LogEntry(timestamp: timestamp, level: level, message: message, 
                       source: nil, lineNumber: lineNumber)
    }
    
    private func parseLevelString(_ str: String) -> LogEntry.LogLevel {
        switch str.uppercased() {
        case "ERROR", "ERR": return .error
        case "WARNING", "WARN": return .warning
        case "DEBUG": return .debug
        default: return .info
        }
    }
}