import Foundation
import SwiftData
import ACPLib

/// Service for caching session history locally using SwiftData
@MainActor
class SessionCacheService {
    private let modelContainer: ModelContainer

    init(modelContainer: ModelContainer) {
        self.modelContainer = modelContainer
    }

    /// Load cached messages for a session
    func loadCachedHistory(sessionId: String) async -> [ACPHistoryMessage] {
        let context = modelContainer.mainContext

        let descriptor = FetchDescriptor<CachedSession>(
            predicate: #Predicate { $0.sessionId == sessionId }
        )

        guard let cachedSession = try? context.fetch(descriptor).first else {
            return []
        }

        // Sort messages by orderIndex and convert to ACPHistoryMessage
        let sortedMessages = cachedSession.messages.sorted { $0.orderIndex < $1.orderIndex }

        return sortedMessages.map { cached in
            ACPHistoryMessage(
                id: cached.messageId,
                role: cached.role == "user" ? .user : .assistant,
                content: cached.content,
                timestamp: cached.timestamp
            )
        }
    }

    /// Save messages to cache for a session
    func cacheHistory(
        sessionId: String,
        projectId: UUID,
        workingDirectory: String,
        messages: [ACPHistoryMessage]
    ) async {
        let context = modelContainer.mainContext

        // Find or create session
        let descriptor = FetchDescriptor<CachedSession>(
            predicate: #Predicate { $0.sessionId == sessionId }
        )

        let cachedSession: CachedSession
        if let existing = try? context.fetch(descriptor).first {
            cachedSession = existing
            // Clear existing messages
            for msg in cachedSession.messages {
                context.delete(msg)
            }
            cachedSession.messages = []
        } else {
            cachedSession = CachedSession(
                sessionId: sessionId,
                projectId: projectId,
                workingDirectory: workingDirectory
            )
            context.insert(cachedSession)
        }

        // Add messages
        for (index, msg) in messages.enumerated() {
            let cachedMsg = CachedMessage(
                messageId: msg.id,
                role: msg.role == .user ? "user" : "assistant",
                content: msg.content,
                timestamp: msg.timestamp,
                orderIndex: index
            )
            cachedMsg.session = cachedSession
            cachedSession.messages.append(cachedMsg)
        }

        cachedSession.lastUpdated = Date()

        do {
            try context.save()
            appLog("SessionCacheService: cached \(messages.count) messages for session \(sessionId)", category: "Cache")
        } catch {
            appLog("SessionCacheService: failed to save cache: \(error)", category: "Cache")
        }
    }

    /// Append a single message to the cache (for new messages during chat)
    func appendMessage(
        sessionId: String,
        projectId: UUID,
        workingDirectory: String,
        message: ACPHistoryMessage
    ) async {
        let context = modelContainer.mainContext

        // Find or create session
        let descriptor = FetchDescriptor<CachedSession>(
            predicate: #Predicate { $0.sessionId == sessionId }
        )

        let cachedSession: CachedSession
        if let existing = try? context.fetch(descriptor).first {
            cachedSession = existing
        } else {
            cachedSession = CachedSession(
                sessionId: sessionId,
                projectId: projectId,
                workingDirectory: workingDirectory
            )
            context.insert(cachedSession)
        }

        // Get next order index
        let nextIndex = cachedSession.messages.count

        let cachedMsg = CachedMessage(
            messageId: message.id,
            role: message.role == .user ? "user" : "assistant",
            content: message.content,
            timestamp: message.timestamp,
            orderIndex: nextIndex
        )
        cachedMsg.session = cachedSession
        cachedSession.messages.append(cachedMsg)
        cachedSession.lastUpdated = Date()

        do {
            try context.save()
        } catch {
            appLog("SessionCacheService: failed to append message: \(error)", category: "Cache")
        }
    }

    /// Update the last message in cache (for streaming updates)
    func updateLastMessage(sessionId: String, content: String) async {
        let context = modelContainer.mainContext

        let descriptor = FetchDescriptor<CachedSession>(
            predicate: #Predicate { $0.sessionId == sessionId }
        )

        guard let cachedSession = try? context.fetch(descriptor).first else {
            return
        }

        // Find last message
        if let lastMessage = cachedSession.messages.max(by: { $0.orderIndex < $1.orderIndex }) {
            lastMessage.content = content
            cachedSession.lastUpdated = Date()

            try? context.save()
        }
    }

    /// Delete cached session
    func deleteCachedSession(sessionId: String) async {
        let context = modelContainer.mainContext

        let descriptor = FetchDescriptor<CachedSession>(
            predicate: #Predicate { $0.sessionId == sessionId }
        )

        if let cachedSession = try? context.fetch(descriptor).first {
            context.delete(cachedSession)
            try? context.save()
            appLog("SessionCacheService: deleted cache for session \(sessionId)", category: "Cache")
        }
    }

    /// Get all cached sessions for a project
    func getCachedSessions(projectId: UUID) async -> [CachedSession] {
        let context = modelContainer.mainContext

        let descriptor = FetchDescriptor<CachedSession>(
            predicate: #Predicate { $0.projectId == projectId },
            sortBy: [SortDescriptor(\.lastUpdated, order: .reverse)]
        )

        return (try? context.fetch(descriptor)) ?? []
    }

    /// Check if a session is cached
    func isCached(sessionId: String) async -> Bool {
        let context = modelContainer.mainContext

        let descriptor = FetchDescriptor<CachedSession>(
            predicate: #Predicate { $0.sessionId == sessionId }
        )

        return (try? context.fetchCount(descriptor)) ?? 0 > 0
    }
}
