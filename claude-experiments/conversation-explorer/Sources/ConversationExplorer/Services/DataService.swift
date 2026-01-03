import Foundation
import SwiftUI

@MainActor
class DataService: ObservableObject {
    @Published var historyEntries: [HistoryEntry] = []
    @Published var isLoading = false
    @Published var searchQuery = ""
    @Published var selectedProject: String?

    private var conversationCache: [String: Conversation] = [:]
    private let claudeDir: URL

    init() {
        claudeDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".claude")
    }

    var filteredEntries: [HistoryEntry] {
        var entries = historyEntries

        // Filter by project
        if let project = selectedProject {
            entries = entries.filter { $0.project == project }
        }

        // Filter by search query (quick filter on display + project)
        if !searchQuery.isEmpty {
            let query = searchQuery.lowercased()
            entries = entries.filter {
                $0.display.lowercased().contains(query) ||
                $0.project.lowercased().contains(query)
            }
        }

        return entries
    }

    var uniqueProjects: [String] {
        // Group by project and get the most recent timestamp for each
        var projectTimestamps: [String: Int64] = [:]
        for entry in historyEntries {
            if let existing = projectTimestamps[entry.project] {
                projectTimestamps[entry.project] = max(existing, entry.timestamp)
            } else {
                projectTimestamps[entry.project] = entry.timestamp
            }
        }
        // Sort by most recent first
        return projectTimestamps.keys.sorted { a, b in
            (projectTimestamps[a] ?? 0) > (projectTimestamps[b] ?? 0)
        }
    }

    func entriesForProject(_ project: String) -> [HistoryEntry] {
        historyEntries
            .filter { $0.project == project }
            .sorted { $0.timestamp > $1.timestamp }
    }

    func loadHistory() async {
        isLoading = true
        defer { isLoading = false }

        let historyFile = claudeDir.appendingPathComponent("history.jsonl")

        guard let data = try? Data(contentsOf: historyFile),
              let content = String(data: data, encoding: .utf8) else {
            return
        }

        let lines = content.split(separator: "\n", omittingEmptySubsequences: true)
        let decoder = JSONDecoder()

        var entries: [HistoryEntry] = []
        for line in lines {
            if let entry = try? decoder.decode(HistoryEntry.self, from: Data(line.utf8)) {
                entries.append(entry)
            }
        }

        // Deduplicate by sessionId, keeping the entry with the latest timestamp
        var seenSessions: [String: HistoryEntry] = [:]
        for entry in entries {
            guard let sessionId = entry.sessionId else { continue }
            if let existing = seenSessions[sessionId] {
                if entry.timestamp > existing.timestamp {
                    seenSessions[sessionId] = entry
                }
            } else {
                seenSessions[sessionId] = entry
            }
        }

        // Sort by timestamp descending (newest first)
        historyEntries = Array(seenSessions.values)
            .sorted { $0.timestamp > $1.timestamp }
    }

    func loadConversation(for entry: HistoryEntry) async -> Conversation? {
        guard let sessionId = entry.sessionId else { return nil }

        // Check cache
        if let cached = conversationCache[sessionId] {
            return cached
        }

        let projectDir = claudeDir
            .appendingPathComponent("projects")
            .appendingPathComponent(entry.encodedProjectPath)

        let conversationFile = projectDir.appendingPathComponent("\(sessionId).jsonl")

        guard FileManager.default.fileExists(atPath: conversationFile.path),
              let data = try? Data(contentsOf: conversationFile),
              let content = String(data: data, encoding: .utf8) else {
            return nil
        }

        let lines = content.split(separator: "\n", omittingEmptySubsequences: true)
        let decoder = JSONDecoder()

        var messages: [Message] = []
        var summary: String?

        for line in lines {
            guard let message = try? decoder.decode(Message.self, from: Data(line.utf8)) else {
                continue
            }

            if message.isSummary {
                summary = message.summary
            }

            if message.isUser || message.isAssistant {
                messages.append(message)
            }
        }

        let timestamps = messages.compactMap { $0.dateFromTimestamp }

        let conversation = Conversation(
            sessionId: sessionId,
            project: entry.project,
            messages: messages,
            summary: summary,
            firstTimestamp: timestamps.min(),
            lastTimestamp: timestamps.max()
        )

        conversationCache[sessionId] = conversation
        return conversation
    }

    func searchFullText(query: String) async -> [HistoryEntry] {
        guard !query.isEmpty else { return historyEntries }

        let lowercasedQuery = query.lowercased()
        var results: [HistoryEntry] = []

        // First pass: filter by display/project (fast)
        let quickMatches = Set(historyEntries.filter {
            $0.display.lowercased().contains(lowercasedQuery) ||
            $0.project.lowercased().contains(lowercasedQuery)
        }.map { $0.id })

        // Add quick matches
        results = historyEntries.filter { quickMatches.contains($0.id) }

        // Second pass: search full conversation content
        for entry in historyEntries where !quickMatches.contains(entry.id) {
            if let conversation = await loadConversation(for: entry),
               conversation.allTextContent.lowercased().contains(lowercasedQuery) {
                results.append(entry)
            }
        }

        return results.sorted { $0.timestamp > $1.timestamp }
    }
}
