// Virtual log store that provides on-demand access to log entries without loading entire file
// Based on HexFiend's lazy loading architecture

import Foundation
import SwiftUI

class VirtualLogStore: ObservableObject {
    private let lineIndex: LogLineIndex
    private let slice: LogSlice
    private var parsedEntries: [Int: LogEntry] = [:] // Cache parsed entries
    private let dateFormatter: DateFormatter
    private let isoFormatter: ISO8601DateFormatter
    
    @Published var selectedIndex: Int? = nil
    
    init?(url: URL) {
        guard let fileRef = LogFileReference(url: url) else { return nil }
        
        self.slice = LogFileSlice(fileRef: fileRef)
        self.lineIndex = LogLineIndex(slice: slice)
        
        // Initialize formatters
        self.dateFormatter = DateFormatter()
        self.dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        
        self.isoFormatter = ISO8601DateFormatter()
        self.isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    }
    
    /// Get total number of lines in the file
    var totalLines: Int {
        lineIndex.totalLines()
    }
    
    /// Get log entry for a specific line number
    func entry(at lineNumber: Int) -> LogEntry? {
        // Check cache first
        if let cached = parsedEntries[lineNumber] {
            return cached
        }
        
        // Get line info and parse
        guard let lineInfo = lineIndex.lineInfo(at: lineNumber),
              let lineText = slice.substring(in: lineInfo.byteRange) else {
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
        // Get first entry
        guard let firstEntry = entry(at: 0) else { return nil }
        
        // Get last entry
        let lastLineIndex = totalLines - 1
        guard lastLineIndex >= 0,
              let lastEntry = entry(at: lastLineIndex) else {
            return (firstEntry.timestamp, firstEntry.timestamp)
        }
        
        return (firstEntry.timestamp, lastEntry.timestamp)
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
    
    private func parseLogLine(_ line: String, lineNumber: Int) -> LogEntry {
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