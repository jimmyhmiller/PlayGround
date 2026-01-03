import Foundation

struct Conversation: Identifiable {
    let sessionId: String
    let project: String
    let messages: [Message]
    let summary: String?
    let firstTimestamp: Date?
    let lastTimestamp: Date?

    var id: String { sessionId }

    var userMessages: [Message] {
        messages.filter { $0.isUser }
    }

    var assistantMessages: [Message] {
        messages.filter { $0.isAssistant }
    }

    var allTextContent: String {
        messages.compactMap { msg -> String? in
            if msg.isUser || msg.isAssistant {
                return msg.textContent
            } else if msg.isSummary {
                return msg.summary
            }
            return nil
        }.joined(separator: " ")
    }
}
