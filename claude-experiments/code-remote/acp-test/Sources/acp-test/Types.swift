import Foundation

// MARK: - ACP Protocol Constants

let acpProtocolVersion = 1

// MARK: - Client Info

struct ACPClientInfo: Codable, Sendable {
    let name: String
    let title: String
    let version: String
}

// MARK: - Agent Info

struct ACPAgentInfo: Codable, Sendable {
    let name: String
    let title: String
    let version: String
}

// MARK: - Capabilities

struct ACPClientCapabilities: Codable, Sendable {
    let fs: ACPFSCapabilities?
    let terminal: Bool?

    init(fs: ACPFSCapabilities? = nil, terminal: Bool? = nil) {
        self.fs = fs
        self.terminal = terminal
    }
}

struct ACPFSCapabilities: Codable, Sendable {
    let readTextFile: Bool?
    let writeTextFile: Bool?

    init(readTextFile: Bool? = nil, writeTextFile: Bool? = nil) {
        self.readTextFile = readTextFile
        self.writeTextFile = writeTextFile
    }
}

struct ACPPromptCapabilities: Codable, Sendable {
    let image: Bool?
    let audio: Bool?
    let embeddedContext: Bool?
}

struct ACPMCPCapabilities: Codable, Sendable {
    let http: Bool?
    let sse: Bool?
}

struct ACPAgentCapabilities: Codable, Sendable {
    let loadSession: Bool?
    let promptCapabilities: ACPPromptCapabilities?
    let mcp: ACPMCPCapabilities?
}

// MARK: - Initialize

struct ACPInitializeParams: Codable, Sendable {
    let protocolVersion: Int
    let clientCapabilities: ACPClientCapabilities
    let clientInfo: ACPClientInfo

    init(
        protocolVersion: Int = acpProtocolVersion,
        clientCapabilities: ACPClientCapabilities = ACPClientCapabilities(),
        clientInfo: ACPClientInfo
    ) {
        self.protocolVersion = protocolVersion
        self.clientCapabilities = clientCapabilities
        self.clientInfo = clientInfo
    }
}

struct ACPAuthMethod: Codable, Sendable {
    let type: String?
    let method: String?
    let url: String?

    // Handle various possible structures
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try container.decodeIfPresent(String.self, forKey: .type)
        self.method = try container.decodeIfPresent(String.self, forKey: .method)
        self.url = try container.decodeIfPresent(String.self, forKey: .url)
    }

    enum CodingKeys: String, CodingKey {
        case type, method, url
    }
}

struct ACPInitializeResult: Codable, Sendable {
    let protocolVersion: Int
    let agentCapabilities: ACPAgentCapabilities?
    let agentInfo: ACPAgentInfo
    let authMethods: AnyCodableValue? // Make flexible to handle any structure
}

// MARK: - Authenticate

struct ACPAuthenticateParams: Codable, Sendable {
    let method: String
    let token: String?
}

struct ACPAuthenticateResult: Codable, Sendable {
    let success: Bool
}

// MARK: - MCP Server Config

struct ACPMCPServer: Codable, Sendable {
    let name: String
    let command: String?
    let args: [String]?
    let env: [String: String]?
    let url: String?

    init(name: String, command: String? = nil, args: [String]? = nil, env: [String: String]? = nil, url: String? = nil) {
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.url = url
    }
}

// MARK: - Session New

struct ACPSessionNewParams: Codable, Sendable {
    let cwd: String
    let mcpServers: [ACPMCPServer]

    init(cwd: String, mcpServers: [ACPMCPServer] = []) {
        self.cwd = cwd
        self.mcpServers = mcpServers
    }
}

struct ACPSessionNewResult: Codable, Sendable {
    let sessionId: String
}

// MARK: - Session Load

struct ACPSessionLoadParams: Codable, Sendable {
    let sessionId: String
    let cwd: String?
    let mcpServers: [ACPMCPServer]?

    init(sessionId: String, cwd: String? = nil, mcpServers: [ACPMCPServer]? = nil) {
        self.sessionId = sessionId
        self.cwd = cwd
        self.mcpServers = mcpServers
    }
}

struct ACPSessionLoadResult: Codable, Sendable {
    let sessionId: String
}

// MARK: - Content Blocks

enum ACPContentBlock: Codable, Sendable {
    case text(ACPTextContent)
    case resource(ACPResourceContent)
    case image(ACPImageContent)

    enum CodingKeys: String, CodingKey {
        case type
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "text":
            self = .text(try ACPTextContent(from: decoder))
        case "resource":
            self = .resource(try ACPResourceContent(from: decoder))
        case "image":
            self = .image(try ACPImageContent(from: decoder))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown content block type: \(type)"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .text(let content):
            try content.encode(to: encoder)
        case .resource(let content):
            try content.encode(to: encoder)
        case .image(let content):
            try content.encode(to: encoder)
        }
    }
}

struct ACPTextContent: Codable, Sendable {
    let type: String
    let text: String

    init(text: String) {
        self.type = "text"
        self.text = text
    }
}

struct ACPResourceContent: Codable, Sendable {
    let type: String
    let resource: ACPResource

    init(resource: ACPResource) {
        self.type = "resource"
        self.resource = resource
    }
}

struct ACPResource: Codable, Sendable {
    let uri: String
    let mimeType: String?
    let text: String?
    let blob: String? // base64 encoded
}

struct ACPImageContent: Codable, Sendable {
    let type: String
    let source: ACPImageSource

    init(source: ACPImageSource) {
        self.type = "image"
        self.source = source
    }
}

struct ACPImageSource: Codable, Sendable {
    let type: String // "base64" or "url"
    let mediaType: String?
    let data: String? // base64 for type="base64"
    let url: String?  // url for type="url"
}

// MARK: - Session Prompt

struct ACPSessionPromptParams: Codable, Sendable {
    let sessionId: String
    let prompt: [ACPContentBlock]

    init(sessionId: String, prompt: [ACPContentBlock]) {
        self.sessionId = sessionId
        self.prompt = prompt
    }

    init(sessionId: String, text: String) {
        self.sessionId = sessionId
        self.prompt = [.text(ACPTextContent(text: text))]
    }
}

struct ACPSessionPromptResult: Codable, Sendable {
    let stopReason: String // "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"
}

// MARK: - Session Cancel

struct ACPSessionCancelParams: Codable, Sendable {
    let sessionId: String
}

// MARK: - Session Update (Notification from Agent)

struct ACPSessionUpdateParams: Codable, Sendable {
    let sessionId: String
    let update: ACPSessionUpdate
}

enum ACPSessionUpdate: Codable, Sendable {
    case agentMessageChunk(ACPAgentMessageChunk)
    case agentMessageComplete(ACPAgentMessageComplete)
    case toolCall(ACPToolCallUpdate)
    case planStep(ACPPlanStepUpdate)
    case modeChange(ACPModeChangeUpdate)
    case unknown(String) // Handle unknown update types gracefully

    enum CodingKeys: String, CodingKey {
        case sessionUpdate
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let updateType = try container.decode(String.self, forKey: .sessionUpdate)

        switch updateType {
        case "agent_message_chunk":
            self = .agentMessageChunk(try ACPAgentMessageChunk(from: decoder))
        case "agent_message_complete":
            self = .agentMessageComplete(try ACPAgentMessageComplete(from: decoder))
        case "tool_call":
            self = .toolCall(try ACPToolCallUpdate(from: decoder))
        case "plan_step":
            self = .planStep(try ACPPlanStepUpdate(from: decoder))
        case "mode_change":
            self = .modeChange(try ACPModeChangeUpdate(from: decoder))
        default:
            // Handle unknown types gracefully instead of throwing
            self = .unknown(updateType)
        }
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .agentMessageChunk(let chunk):
            try chunk.encode(to: encoder)
        case .agentMessageComplete(let complete):
            try complete.encode(to: encoder)
        case .toolCall(let toolCall):
            try toolCall.encode(to: encoder)
        case .planStep(let planStep):
            try planStep.encode(to: encoder)
        case .modeChange(let modeChange):
            try modeChange.encode(to: encoder)
        case .unknown:
            // Can't encode unknown types
            break
        }
    }
}

struct ACPAgentMessageChunk: Codable, Sendable {
    let sessionUpdate: String
    let content: ACPAgentContentBlock

    init(content: ACPAgentContentBlock) {
        self.sessionUpdate = "agent_message_chunk"
        self.content = content
    }
}

struct ACPAgentMessageComplete: Codable, Sendable {
    let sessionUpdate: String

    init() {
        self.sessionUpdate = "agent_message_complete"
    }
}

enum ACPAgentContentBlock: Codable, Sendable {
    case text(ACPAgentTextContent)
    case thinking(ACPAgentThinkingContent)
    case toolUse(ACPAgentToolUseContent)

    enum CodingKeys: String, CodingKey {
        case type
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "text":
            self = .text(try ACPAgentTextContent(from: decoder))
        case "thinking":
            self = .thinking(try ACPAgentThinkingContent(from: decoder))
        case "tool_use":
            self = .toolUse(try ACPAgentToolUseContent(from: decoder))
        default:
            // Default to text for unknown types
            self = .text(ACPAgentTextContent(text: ""))
        }
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .text(let content):
            try content.encode(to: encoder)
        case .thinking(let content):
            try content.encode(to: encoder)
        case .toolUse(let content):
            try content.encode(to: encoder)
        }
    }

    var textValue: String? {
        switch self {
        case .text(let content): return content.text
        case .thinking(let content): return content.thinking
        case .toolUse: return nil
        }
    }
}

struct ACPAgentTextContent: Codable, Sendable {
    let type: String
    let text: String

    init(text: String) {
        self.type = "text"
        self.text = text
    }
}

struct ACPAgentThinkingContent: Codable, Sendable {
    let type: String
    let thinking: String

    init(thinking: String) {
        self.type = "thinking"
        self.thinking = thinking
    }
}

struct ACPAgentToolUseContent: Codable, Sendable {
    let type: String
    let id: String
    let name: String
    let input: AnyCodableValue?

    init(id: String, name: String, input: AnyCodableValue? = nil) {
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
    }
}

struct ACPToolCallUpdate: Codable, Sendable {
    let sessionUpdate: String
    let toolCallId: String
    let title: String?
    let kind: String? // "bash", "file_read", "file_write", "file_edit", "mcp", "other"
    let status: String // "pending", "running", "complete", "error", "cancelled"
    let output: String?
    let error: String?

    init(
        toolCallId: String,
        title: String? = nil,
        kind: String? = nil,
        status: String,
        output: String? = nil,
        error: String? = nil
    ) {
        self.sessionUpdate = "tool_call"
        self.toolCallId = toolCallId
        self.title = title
        self.kind = kind
        self.status = status
        self.output = output
        self.error = error
    }
}

struct ACPPlanStepUpdate: Codable, Sendable {
    let sessionUpdate: String
    let stepId: String
    let title: String
    let status: String // "pending", "in_progress", "complete", "error"

    init(stepId: String, title: String, status: String) {
        self.sessionUpdate = "plan_step"
        self.stepId = stepId
        self.title = title
        self.status = status
    }
}

struct ACPModeChangeUpdate: Codable, Sendable {
    let sessionUpdate: String
    let mode: String // "agent", "plan", "user_approval"

    init(mode: String) {
        self.sessionUpdate = "mode_change"
        self.mode = mode
    }
}

// MARK: - Permission Request (Agent -> Client)

struct ACPRequestPermissionParams: Codable, Sendable {
    let toolCallId: String
    let toolName: String
    let input: AnyCodableValue?
    let prompt: String?
}

struct ACPRequestPermissionResult: Codable, Sendable {
    let granted: Bool
    let additionalContext: String?
}

// MARK: - File System Operations (Agent -> Client)

struct ACPReadTextFileParams: Codable, Sendable {
    let path: String
    let startLine: Int?
    let endLine: Int?
}

struct ACPReadTextFileResult: Codable, Sendable {
    let content: String
    let lineCount: Int?
}

struct ACPWriteTextFileParams: Codable, Sendable {
    let path: String
    let content: String
    let createDirectories: Bool?
}

struct ACPWriteTextFileResult: Codable, Sendable {
    let bytesWritten: Int
}

// MARK: - Terminal Operations (Agent -> Client)

struct ACPTerminalCreateParams: Codable, Sendable {
    let command: String
    let args: [String]?
    let cwd: String?
    let env: [String: String]?
    let timeout: Int?
}

struct ACPTerminalCreateResult: Codable, Sendable {
    let terminalId: String
}

struct ACPTerminalOutputParams: Codable, Sendable {
    let terminalId: String
}

struct ACPTerminalOutputResult: Codable, Sendable {
    let output: String
    let exitCode: Int?
}

struct ACPTerminalReleaseParams: Codable, Sendable {
    let terminalId: String
}

struct ACPTerminalKillParams: Codable, Sendable {
    let terminalId: String
}

// MARK: - ACP Error Codes

enum ACPErrorCode: Int {
    case parseError = -32700
    case invalidRequest = -32600
    case methodNotFound = -32601
    case invalidParams = -32602
    case internalError = -32603

    // ACP-specific errors
    case sessionNotFound = -32001
    case authenticationRequired = -32002
    case permissionDenied = -32003
    case resourceNotFound = -32004
}
