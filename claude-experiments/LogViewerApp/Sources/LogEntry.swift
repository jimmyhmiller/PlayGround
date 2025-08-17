import Foundation
import SwiftUI

extension Notification.Name {
    static let jumpToLogLine = Notification.Name("jumpToLogLine")
}

struct LogEntry: Identifiable, Hashable {
    let id = UUID()
    let timestamp: Date
    let level: LogLevel
    let message: String
    let source: String?
    let lineNumber: Int
    
    enum LogLevel: String, CaseIterable {
        case info = "INFO"
        case warning = "WARN" 
        case error = "ERROR"
        case debug = "DEBUG"
        
        var color: Color {
            switch self {
            case .info: return .blue
            case .warning: return .orange
            case .error: return .red
            case .debug: return .gray
            }
        }
    }
}

class LogStore: ObservableObject {
    @Published var entries: [LogEntry] = []
    @Published var selectedIndex: Int? = nil
    
    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()
    
    private let isoFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
    
    private let syslogFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM dd HH:mm:ss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()
    
    func loadFromFile(at url: URL) {
        do {
            let content = try String(contentsOf: url, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines)
            
            var parsedEntries: [LogEntry] = []
            
            for line in lines where !line.isEmpty {
                if let entry = parseLogLine(line) {
                    parsedEntries.append(entry)
                }
            }
            
            self.entries = parsedEntries.sorted { $0.timestamp < $1.timestamp }
        } catch {
            print("Error loading file: \(error)")
            loadSampleData()
        }
    }
    
    private func parseLogLine(_ line: String) -> LogEntry? {
        // Try JSON format first
        if line.starts(with: "{") {
            return parseJSONLog(line)
        }
        
        // Try common log formats with timestamps
        if let entry = parseTimestampedLog(line) {
            return entry
        }
        
        // Fallback to simple parsing
        return parseSimpleLog(line)
    }
    
    private func parseJSONLog(_ line: String) -> LogEntry? {
        guard let data = line.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        
        let timestamp = parseJSONTimestamp(json) ?? Date()
        let level = parseJSONLevel(json)
        let message = json["message"] as? String ?? json["msg"] as? String ?? line
        let source = json["source"] as? String ?? json["logger"] as? String
        
        return LogEntry(timestamp: timestamp, level: level, message: message, source: source, lineNumber: -1)
    }
    
    private func parseJSONTimestamp(_ json: [String: Any]) -> Date? {
        if let timestampStr = json["timestamp"] as? String ?? json["time"] as? String ?? json["@timestamp"] as? String {
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
    
    private func parseTimestampedLog(_ line: String) -> LogEntry? {
        // Common format: "2024-01-01 12:00:00.000 [LEVEL] message"
        let pattern = #"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?)(?:\s+\[(\w+)\])?\s+(.+)$"#
        
        if let regex = try? NSRegularExpression(pattern: pattern),
           let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) {
            
            let timestampRange = Range(match.range(at: 1), in: line)!
            let levelRange = Range(match.range(at: 2), in: line)
            let messageRange = Range(match.range(at: 3), in: line)!
            
            let timestamp = dateFormatter.date(from: String(line[timestampRange])) ?? Date()
            let level = levelRange != nil ? parseLevelString(String(line[levelRange!])) : .info
            let message = String(line[messageRange])
            
            return LogEntry(timestamp: timestamp, level: level, message: message, source: nil, lineNumber: -1)
        }
        
        return nil
    }
    
    private func parseSimpleLog(_ line: String) -> LogEntry? {
        // Fallback: treat the whole line as a message with current timestamp
        return LogEntry(timestamp: Date(), level: .info, message: line, source: nil, lineNumber: -1)
    }
    
    private func parseLevelString(_ str: String) -> LogEntry.LogLevel {
        switch str.uppercased() {
        case "ERROR", "ERR": return .error
        case "WARNING", "WARN": return .warning
        case "DEBUG": return .debug
        default: return .info
        }
    }
    
    func loadSampleData() {
        let baseDate = Date()
        var sampleEntries: [LogEntry] = []
        
        for i in 0..<50 {
            let timestamp = baseDate.addingTimeInterval(TimeInterval(i * 2))
            let levels: [LogEntry.LogLevel] = [.info, .warning, .error, .debug]
            let level = levels.randomElement() ?? .info
            let messages = [
                "Task completed successfully",
                "User login attempt",
                "Connection establishment", 
                "Database query executed",
                "Cache invalidated",
                "Server startup",
                "User logout",
                "Authentication failed",
                "Application start start",
                "User login"
            ]
            let message = messages.randomElement() ?? "Log message"
            
            sampleEntries.append(LogEntry(
                timestamp: timestamp,
                level: level,
                message: message,
                source: "app.server",
                lineNumber: i
            ))
        }
        
        self.entries = sampleEntries.sorted { $0.timestamp < $1.timestamp }
    }
    
    func getTimeRange() -> (start: Date, end: Date)? {
        guard let first = entries.first?.timestamp,
              let last = entries.last?.timestamp else {
            return nil
        }
        return (start: first, end: last)
    }
    
    func getEntriesCount(in timeInterval: TimeInterval, from startTime: Date) -> Int {
        let endTime = startTime.addingTimeInterval(timeInterval)
        return entries.filter { entry in
            entry.timestamp >= startTime && entry.timestamp < endTime
        }.count
    }
    
    func jumpToTime(_ targetTime: Date) {
        // Find the entry closest to the target time
        var closestIndex = 0
        var smallestDifference = Double.infinity
        
        for (index, entry) in entries.enumerated() {
            let difference = abs(entry.timestamp.timeIntervalSince(targetTime))
            if difference < smallestDifference {
                smallestDifference = difference
                closestIndex = index
            }
        }
        
        selectedIndex = closestIndex
    }
}