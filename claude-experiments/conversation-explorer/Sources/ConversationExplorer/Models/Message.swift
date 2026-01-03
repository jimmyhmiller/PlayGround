import Foundation

struct Message: Codable, Identifiable {
    let type: String
    let uuid: String?
    let parentUuid: String?
    let sessionId: String?
    let timestamp: String?
    let cwd: String?
    let gitBranch: String?
    let message: MessageContent?
    let summary: String?

    var id: String { uuid ?? UUID().uuidString }

    var dateFromTimestamp: Date? {
        guard let timestamp = timestamp else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: timestamp)
    }

    var isUser: Bool { type == "user" }
    var isAssistant: Bool { type == "assistant" }
    var isSummary: Bool { type == "summary" }

    var textContent: String {
        if let summary = summary {
            return summary
        }
        guard let content = message?.content else { return "" }
        switch content {
        case .string(let s):
            return s
        case .blocks(let blocks):
            return blocks.compactMap { block -> String? in
                switch block {
                case .text(let text):
                    return text
                case .thinking(let thinking, _):
                    return thinking
                case .toolUse(let name, _):
                    return "[Tool: \(name)]"
                case .toolResult(let content):
                    return content
                case .unknown:
                    return nil
                }
            }.joined(separator: "\n")
        }
    }
}

struct MessageContent: Codable {
    let role: String?
    let content: ContentValue?
    let model: String?
}

enum ContentValue: Codable {
    case string(String)
    case blocks([ContentBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let blocks = try? container.decode([ContentBlock].self) {
            self = .blocks(blocks)
        } else {
            self = .string("")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s):
            try container.encode(s)
        case .blocks(let b):
            try container.encode(b)
        }
    }
}

enum ContentBlock: Codable {
    case text(String)
    case thinking(String, signature: String?)
    case toolUse(name: String, id: String?)
    case toolResult(String)
    case unknown

    private enum CodingKeys: String, CodingKey {
        case type, text, thinking, signature, name, id, content
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let typeString = try container.decode(String.self, forKey: .type)

        switch typeString {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)
        case "thinking":
            let thinking = try container.decode(String.self, forKey: .thinking)
            let signature = try? container.decode(String.self, forKey: .signature)
            self = .thinking(thinking, signature: signature)
        case "tool_use":
            let name = try container.decode(String.self, forKey: .name)
            let id = try? container.decode(String.self, forKey: .id)
            self = .toolUse(name: name, id: id)
        case "tool_result":
            let content = (try? container.decode(String.self, forKey: .content)) ?? ""
            self = .toolResult(content)
        default:
            self = .unknown
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .thinking(let thinking, let signature):
            try container.encode("thinking", forKey: .type)
            try container.encode(thinking, forKey: .thinking)
            if let sig = signature {
                try container.encode(sig, forKey: .signature)
            }
        case .toolUse(let name, let id):
            try container.encode("tool_use", forKey: .type)
            try container.encode(name, forKey: .name)
            if let id = id {
                try container.encode(id, forKey: .id)
            }
        case .toolResult(let content):
            try container.encode("tool_result", forKey: .type)
            try container.encode(content, forKey: .content)
        case .unknown:
            try container.encode("unknown", forKey: .type)
        }
    }
}
