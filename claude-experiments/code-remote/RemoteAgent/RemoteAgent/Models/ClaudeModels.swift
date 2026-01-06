import Foundation

// MARK: - Claude Code Stream-JSON Protocol Models
// Based on the actual Claude Code CLI output format

/// Union type for all messages from Claude Code
enum ClaudeMessage: Decodable {
    case system(ClaudeSystemMessage)
    case assistant(ClaudeAssistantMessage)
    case user(ClaudeUserMessage)
    case result(ClaudeResultMessage)
    case streamEvent(ClaudeStreamEvent)

    enum CodingKeys: String, CodingKey {
        case type
        case subtype
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "system":
            self = .system(try ClaudeSystemMessage(from: decoder))
        case "assistant":
            self = .assistant(try ClaudeAssistantMessage(from: decoder))
        case "user":
            self = .user(try ClaudeUserMessage(from: decoder))
        case "result":
            self = .result(try ClaudeResultMessage(from: decoder))
        case "stream_event":
            self = .streamEvent(try ClaudeStreamEvent(from: decoder))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown message type: \(type)"
            )
        }
    }
}

// MARK: - System Message

struct ClaudeSystemMessage: Decodable {
    let type: String
    let subtype: String
    let sessionId: String
    let uuid: String
    let cwd: String
    let tools: [String]
    let mcpServers: [MCPServer]
    let model: String
    let permissionMode: String
    let slashCommands: [String]
    let apiKeySource: String
    let claudeCodeVersion: String?

    enum CodingKeys: String, CodingKey {
        case type, subtype, uuid, cwd, tools, model
        case sessionId = "session_id"
        case mcpServers = "mcp_servers"
        case permissionMode = "permissionMode"
        case slashCommands = "slash_commands"
        case apiKeySource = "apiKeySource"
        case claudeCodeVersion = "claude_code_version"
    }
}

struct MCPServer: Decodable {
    let name: String
    let status: String
}

// MARK: - Assistant Message

struct ClaudeAssistantMessage: Decodable {
    let type: String
    let uuid: String
    let sessionId: String
    let message: APIMessage
    let parentToolUseId: String?

    enum CodingKeys: String, CodingKey {
        case type, uuid, message
        case sessionId = "session_id"
        case parentToolUseId = "parent_tool_use_id"
    }
}

struct APIMessage: Decodable {
    let model: String?
    let id: String?
    let role: String?
    let content: [ContentBlock]
    let stopReason: String?
    let usage: APIUsage?

    enum CodingKeys: String, CodingKey {
        case model, id, role, content
        case stopReason = "stop_reason"
        case usage
    }
}

struct APIUsage: Decodable {
    let inputTokens: Int?
    let outputTokens: Int?
    let cacheCreationInputTokens: Int?
    let cacheReadInputTokens: Int?

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case cacheCreationInputTokens = "cache_creation_input_tokens"
        case cacheReadInputTokens = "cache_read_input_tokens"
    }
}

// MARK: - User Message

struct ClaudeUserMessage: Decodable {
    let type: String
    let uuid: String?
    let sessionId: String
    let message: APIMessage
    let parentToolUseId: String?

    enum CodingKeys: String, CodingKey {
        case type, uuid, message
        case sessionId = "session_id"
        case parentToolUseId = "parent_tool_use_id"
    }
}

// MARK: - Result Message

struct ClaudeResultMessage: Decodable {
    let type: String
    let subtype: String
    let uuid: String
    let sessionId: String
    let isError: Bool
    let durationMs: Int
    let durationApiMs: Int
    let numTurns: Int
    let result: String?
    let totalCostUsd: Double
    let usage: ResultUsage?
    let errors: [String]?

    enum CodingKeys: String, CodingKey {
        case type, subtype, uuid, result, errors, usage
        case sessionId = "session_id"
        case isError = "is_error"
        case durationMs = "duration_ms"
        case durationApiMs = "duration_api_ms"
        case numTurns = "num_turns"
        case totalCostUsd = "total_cost_usd"
    }
}

struct ResultUsage: Decodable {
    let inputTokens: Int?
    let outputTokens: Int?

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }
}

// MARK: - Stream Event (for partial messages)

struct ClaudeStreamEvent: Decodable {
    let type: String
    let event: StreamEventData
    let parentToolUseId: String?
    let uuid: String
    let sessionId: String

    enum CodingKeys: String, CodingKey {
        case type, event, uuid
        case parentToolUseId = "parent_tool_use_id"
        case sessionId = "session_id"
    }
}

struct StreamEventData: Decodable {
    let type: String
    let index: Int?
    let delta: StreamDelta?
    let contentBlock: ContentBlock?

    enum CodingKeys: String, CodingKey {
        case type, index, delta
        case contentBlock = "content_block"
    }
}

struct StreamDelta: Decodable {
    let type: String
    let text: String?
    let partialJson: String?

    enum CodingKeys: String, CodingKey {
        case type, text
        case partialJson = "partial_json"
    }
}

// MARK: - Content Blocks

enum ContentBlock: Decodable {
    case text(TextBlock)
    case toolUse(ToolUseBlock)
    case toolResult(ToolResultBlock)
    case thinking(ThinkingBlock)
    case unknown(String)

    enum CodingKeys: String, CodingKey {
        case type
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "text":
            self = .text(try TextBlock(from: decoder))
        case "tool_use":
            self = .toolUse(try ToolUseBlock(from: decoder))
        case "tool_result":
            self = .toolResult(try ToolResultBlock(from: decoder))
        case "thinking":
            self = .thinking(try ThinkingBlock(from: decoder))
        default:
            self = .unknown(type)
        }
    }

    var textValue: String? {
        switch self {
        case .text(let block): return block.text
        case .thinking(let block): return block.thinking
        default: return nil
        }
    }
}

struct TextBlock: Decodable {
    let type: String
    let text: String
}

struct ToolUseBlock: Decodable {
    let type: String
    let id: String
    let name: String
    let input: [String: AnyCodable]
}

struct ToolResultBlock: Decodable {
    let type: String
    let toolUseId: String
    let content: AnyCodable?
    let isError: Bool?

    enum CodingKeys: String, CodingKey {
        case type
        case toolUseId = "tool_use_id"
        case content
        case isError = "is_error"
    }
}

struct ThinkingBlock: Decodable {
    let type: String
    let thinking: String
}

// MARK: - AnyCodable for dynamic JSON

struct AnyCodable: Decodable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self.value = NSNull()
        } else if let bool = try? container.decode(Bool.self) {
            self.value = bool
        } else if let int = try? container.decode(Int.self) {
            self.value = int
        } else if let double = try? container.decode(Double.self) {
            self.value = double
        } else if let string = try? container.decode(String.self) {
            self.value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            self.value = array.map { $0.value }
        } else if let dictionary = try? container.decode([String: AnyCodable].self) {
            self.value = dictionary.mapValues { $0.value }
        } else {
            self.value = NSNull()
        }
    }

    func asString() -> String? {
        value as? String
    }

    func asDict() -> [String: Any]? {
        value as? [String: Any]
    }

    func prettyPrinted() -> String {
        if let dict = value as? [String: Any],
           let data = try? JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted),
           let string = String(data: data, encoding: .utf8) {
            return string
        }
        return String(describing: value)
    }
}

extension AnyCodable: Encodable {
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch value {
        case is NSNull:
            try container.encodeNil()
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dictionary as [String: Any]:
            try container.encode(dictionary.mapValues { AnyCodable($0) })
        default:
            try container.encodeNil()
        }
    }
}
