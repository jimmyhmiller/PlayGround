import Foundation
import SwiftData

/// Cached session for offline access
@Model
final class CachedSession {
    @Attribute(.unique) var sessionId: String
    var projectId: UUID
    var workingDirectory: String
    var lastUpdated: Date

    @Relationship(deleteRule: .cascade)
    var messages: [CachedMessage] = []

    init(sessionId: String, projectId: UUID, workingDirectory: String) {
        self.sessionId = sessionId
        self.projectId = projectId
        self.workingDirectory = workingDirectory
        self.lastUpdated = Date()
    }
}

/// Cached message within a session
@Model
final class CachedMessage {
    var messageId: String
    var role: String  // "user" or "assistant"
    var content: String
    var timestamp: Date
    var orderIndex: Int  // For maintaining message order

    var session: CachedSession?

    init(messageId: String, role: String, content: String, timestamp: Date, orderIndex: Int) {
        self.messageId = messageId
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.orderIndex = orderIndex
    }
}
