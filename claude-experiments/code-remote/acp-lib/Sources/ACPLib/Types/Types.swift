import Foundation

// MARK: - Client Info

public struct ACPClientInfo: Codable, Sendable {
    public let name: String
    public let title: String
    public let version: String

    public init(name: String, title: String, version: String) {
        self.name = name
        self.title = title
        self.version = version
    }
}

// MARK: - Agent Info

public struct ACPAgentInfo: Codable, Sendable {
    public let name: String
    public let title: String
    public let version: String

    public init(name: String, title: String, version: String) {
        self.name = name
        self.title = title
        self.version = version
    }
}

// MARK: - Capabilities

public struct ACPClientCapabilities: Codable, Sendable {
    public let fs: ACPFSCapabilities?
    public let terminal: Bool?

    public init(fs: ACPFSCapabilities? = nil, terminal: Bool? = nil) {
        self.fs = fs
        self.terminal = terminal
    }
}

public struct ACPFSCapabilities: Codable, Sendable {
    public let readTextFile: Bool?
    public let writeTextFile: Bool?

    public init(readTextFile: Bool? = nil, writeTextFile: Bool? = nil) {
        self.readTextFile = readTextFile
        self.writeTextFile = writeTextFile
    }
}

public struct ACPPromptCapabilities: Codable, Sendable {
    public let image: Bool?
    public let audio: Bool?
    public let embeddedContext: Bool?

    public init(image: Bool? = nil, audio: Bool? = nil, embeddedContext: Bool? = nil) {
        self.image = image
        self.audio = audio
        self.embeddedContext = embeddedContext
    }
}

public struct ACPMCPCapabilities: Codable, Sendable {
    public let http: Bool?
    public let sse: Bool?

    public init(http: Bool? = nil, sse: Bool? = nil) {
        self.http = http
        self.sse = sse
    }
}

public struct ACPAgentCapabilities: Codable, Sendable {
    public let loadSession: Bool?
    public let promptCapabilities: ACPPromptCapabilities?
    public let mcp: ACPMCPCapabilities?

    public init(loadSession: Bool? = nil, promptCapabilities: ACPPromptCapabilities? = nil, mcp: ACPMCPCapabilities? = nil) {
        self.loadSession = loadSession
        self.promptCapabilities = promptCapabilities
        self.mcp = mcp
    }
}

// MARK: - Initialize

public struct ACPInitializeParams: Codable, Sendable {
    public let protocolVersion: Int
    public let clientCapabilities: ACPClientCapabilities
    public let clientInfo: ACPClientInfo

    public init(
        protocolVersion: Int = acpProtocolVersion,
        clientCapabilities: ACPClientCapabilities = ACPClientCapabilities(),
        clientInfo: ACPClientInfo
    ) {
        self.protocolVersion = protocolVersion
        self.clientCapabilities = clientCapabilities
        self.clientInfo = clientInfo
    }
}

public struct ACPAuthMethod: Codable, Sendable {
    public let type: String?
    public let method: String?
    public let url: String?

    // Handle various possible structures
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try container.decodeIfPresent(String.self, forKey: .type)
        self.method = try container.decodeIfPresent(String.self, forKey: .method)
        self.url = try container.decodeIfPresent(String.self, forKey: .url)
    }

    enum CodingKeys: String, CodingKey {
        case type, method, url
    }
}

public struct ACPInitializeResult: Codable, Sendable {
    public let protocolVersion: Int
    public let agentCapabilities: ACPAgentCapabilities?
    public let agentInfo: ACPAgentInfo
    public let authMethods: AnyCodableValue? // Make flexible to handle any structure

    public init(protocolVersion: Int, agentCapabilities: ACPAgentCapabilities?, agentInfo: ACPAgentInfo, authMethods: AnyCodableValue? = nil) {
        self.protocolVersion = protocolVersion
        self.agentCapabilities = agentCapabilities
        self.agentInfo = agentInfo
        self.authMethods = authMethods
    }
}

// MARK: - Authenticate

public struct ACPAuthenticateParams: Codable, Sendable {
    public let method: String
    public let token: String?

    public init(method: String, token: String? = nil) {
        self.method = method
        self.token = token
    }
}

public struct ACPAuthenticateResult: Codable, Sendable {
    public let success: Bool

    public init(success: Bool) {
        self.success = success
    }
}

// MARK: - MCP Server Config

public struct ACPMCPServer: Codable, Sendable {
    public let name: String
    public let command: String?
    public let args: [String]?
    public let env: [String: String]?
    public let url: String?

    public init(name: String, command: String? = nil, args: [String]? = nil, env: [String: String]? = nil, url: String? = nil) {
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.url = url
    }
}

// MARK: - Session New

public struct ACPSessionNewParams: Codable, Sendable {
    public let cwd: String
    public let mcpServers: [ACPMCPServer]

    public init(cwd: String, mcpServers: [ACPMCPServer] = []) {
        self.cwd = cwd
        self.mcpServers = mcpServers
    }
}

public struct ACPSessionNewResult: Codable, Sendable {
    public let sessionId: String
    public let modes: ACPModeInfo?

    public init(sessionId: String, modes: ACPModeInfo? = nil) {
        self.sessionId = sessionId
        self.modes = modes
    }
}

// MARK: - Session Load

public struct ACPSessionLoadParams: Codable, Sendable {
    public let sessionId: String
    public let cwd: String?
    public let mcpServers: [ACPMCPServer]

    public init(sessionId: String, cwd: String? = nil, mcpServers: [ACPMCPServer] = []) {
        self.sessionId = sessionId
        self.cwd = cwd
        self.mcpServers = mcpServers
    }
}

public struct ACPSessionLoadResult: Codable, Sendable {
    public let sessionId: String
    public let modes: ACPModeInfo?

    public init(sessionId: String, modes: ACPModeInfo? = nil) {
        self.sessionId = sessionId
        self.modes = modes
    }
}

// MARK: - Session Resume

public struct ACPSessionResumeParams: Codable, Sendable {
    public let sessionId: String
    public let cwd: String
    public let mcpServers: [ACPMCPServer]

    public init(sessionId: String, cwd: String, mcpServers: [ACPMCPServer] = []) {
        self.sessionId = sessionId
        self.cwd = cwd
        self.mcpServers = mcpServers
    }
}

public struct ACPSessionResumeResult: Codable, Sendable {
    public let sessionId: String
    public let modes: ACPModeInfo?

    public init(sessionId: String, modes: ACPModeInfo? = nil) {
        self.sessionId = sessionId
        self.modes = modes
    }
}

// MARK: - Mode Types

public struct ACPMode: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let name: String

    public init(id: String, name: String) {
        self.id = id
        self.name = name
    }
}

public struct ACPModeInfo: Codable, Sendable {
    public let availableModes: [ACPMode]
    public let currentModeId: String

    public init(availableModes: [ACPMode], currentModeId: String) {
        self.availableModes = availableModes
        self.currentModeId = currentModeId
    }

    public var currentMode: ACPMode? {
        availableModes.first { $0.id == currentModeId }
    }
}

// MARK: - Set Session Mode

public struct ACPSetSessionModeParams: Codable, Sendable {
    public let sessionId: String
    public let modeId: String

    public init(sessionId: String, modeId: String) {
        self.sessionId = sessionId
        self.modeId = modeId
    }
}

public struct ACPSetSessionModeResult: Codable, Sendable {
    public let success: Bool

    public init(success: Bool) {
        self.success = success
    }
}

// MARK: - Content Blocks

public enum ACPContentBlock: Codable, Sendable {
    case text(ACPTextContent)
    case resource(ACPResourceContent)
    case image(ACPImageContent)

    enum CodingKeys: String, CodingKey {
        case type
    }

    public init(from decoder: Decoder) throws {
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

    public func encode(to encoder: Encoder) throws {
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

public struct ACPTextContent: Codable, Sendable {
    public let type: String
    public let text: String

    public init(text: String) {
        self.type = "text"
        self.text = text
    }
}

public struct ACPResourceContent: Codable, Sendable {
    public let type: String
    public let resource: ACPResource

    public init(resource: ACPResource) {
        self.type = "resource"
        self.resource = resource
    }
}

public struct ACPResource: Codable, Sendable {
    public let uri: String
    public let mimeType: String?
    public let text: String?
    public let blob: String? // base64 encoded

    public init(uri: String, mimeType: String? = nil, text: String? = nil, blob: String? = nil) {
        self.uri = uri
        self.mimeType = mimeType
        self.text = text
        self.blob = blob
    }
}

public struct ACPImageContent: Codable, Sendable {
    public let type: String
    public let source: ACPImageSource

    public init(source: ACPImageSource) {
        self.type = "image"
        self.source = source
    }
}

public struct ACPImageSource: Codable, Sendable {
    public let type: String // "base64" or "url"
    public let mediaType: String?
    public let data: String? // base64 for type="base64"
    public let url: String?  // url for type="url"

    public init(type: String, mediaType: String? = nil, data: String? = nil, url: String? = nil) {
        self.type = type
        self.mediaType = mediaType
        self.data = data
        self.url = url
    }
}

// MARK: - Session Prompt

public struct ACPSessionPromptParams: Codable, Sendable {
    public let sessionId: String
    public let prompt: [ACPContentBlock]

    public init(sessionId: String, prompt: [ACPContentBlock]) {
        self.sessionId = sessionId
        self.prompt = prompt
    }

    public init(sessionId: String, text: String) {
        self.sessionId = sessionId
        self.prompt = [.text(ACPTextContent(text: text))]
    }
}

public struct ACPSessionPromptResult: Codable, Sendable {
    public let stopReason: String // "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"

    public init(stopReason: String) {
        self.stopReason = stopReason
    }
}

// MARK: - Session Cancel

public struct ACPSessionCancelParams: Codable, Sendable {
    public let sessionId: String

    public init(sessionId: String) {
        self.sessionId = sessionId
    }
}

// MARK: - Session Update (Notification from Agent)

public struct ACPSessionUpdateParams: Codable, Sendable {
    public let sessionId: String
    public let update: ACPSessionUpdate

    public init(sessionId: String, update: ACPSessionUpdate) {
        self.sessionId = sessionId
        self.update = update
    }
}

public enum ACPSessionUpdate: Codable, Sendable {
    case agentMessageChunk(ACPAgentMessageChunk)
    case agentMessageComplete(ACPAgentMessageComplete)
    case toolCall(ACPToolCallUpdate)
    case planStep(ACPPlanStepUpdate)
    case modeChange(ACPModeChangeUpdate)
    case currentModeUpdate(ACPCurrentModeUpdate)
    case unknown(String) // Handle unknown update types gracefully

    enum CodingKeys: String, CodingKey {
        case sessionUpdate
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let updateType = try container.decode(String.self, forKey: .sessionUpdate)

        switch updateType {
        case "agent_message_chunk":
            self = .agentMessageChunk(try ACPAgentMessageChunk(from: decoder))
        case "agent_message_complete":
            self = .agentMessageComplete(try ACPAgentMessageComplete(from: decoder))
        case "tool_call", "tool_call_update":
            self = .toolCall(try ACPToolCallUpdate(from: decoder))
        case "plan_step":
            self = .planStep(try ACPPlanStepUpdate(from: decoder))
        case "mode_change":
            self = .modeChange(try ACPModeChangeUpdate(from: decoder))
        case "current_mode_update":
            self = .currentModeUpdate(try ACPCurrentModeUpdate(from: decoder))
        default:
            // Handle unknown types gracefully instead of throwing
            self = .unknown(updateType)
        }
    }

    public func encode(to encoder: Encoder) throws {
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
        case .currentModeUpdate(let update):
            try update.encode(to: encoder)
        case .unknown:
            // Can't encode unknown types
            break
        }
    }
}

public struct ACPAgentMessageChunk: Codable, Sendable {
    public let sessionUpdate: String
    public let content: ACPAgentContentBlock

    public init(content: ACPAgentContentBlock) {
        self.sessionUpdate = "agent_message_chunk"
        self.content = content
    }
}

public struct ACPAgentMessageComplete: Codable, Sendable {
    public let sessionUpdate: String

    public init() {
        self.sessionUpdate = "agent_message_complete"
    }
}

public enum ACPAgentContentBlock: Codable, Sendable {
    case text(ACPAgentTextContent)
    case thinking(ACPAgentThinkingContent)
    case toolUse(ACPAgentToolUseContent)

    enum CodingKeys: String, CodingKey {
        case type
    }

    public init(from decoder: Decoder) throws {
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

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .text(let content):
            try content.encode(to: encoder)
        case .thinking(let content):
            try content.encode(to: encoder)
        case .toolUse(let content):
            try content.encode(to: encoder)
        }
    }

    public var textValue: String? {
        switch self {
        case .text(let content): return content.text
        case .thinking(let content): return content.thinking
        case .toolUse: return nil
        }
    }
}

public struct ACPAgentTextContent: Codable, Sendable {
    public let type: String
    public let text: String

    public init(text: String) {
        self.type = "text"
        self.text = text
    }
}

public struct ACPAgentThinkingContent: Codable, Sendable {
    public let type: String
    public let thinking: String

    public init(thinking: String) {
        self.type = "thinking"
        self.thinking = thinking
    }
}

public struct ACPAgentToolUseContent: Codable, Sendable {
    public let type: String
    public let id: String
    public let name: String
    public let input: AnyCodableValue?

    public init(id: String, name: String, input: AnyCodableValue? = nil) {
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
    }
}

public struct ACPToolCallUpdate: Codable, Sendable {
    public let sessionUpdate: String
    public let toolCallId: String
    public let title: String?
    public let kind: String? // "bash", "file_read", "file_write", "file_edit", "mcp", "other"
    public let status: String // "pending", "running", "complete", "error", "cancelled"
    public let rawInput: AnyCodableValue?
    public let output: String?
    public let error: String?

    /// Get rawInput as JSON string
    public var rawInputString: String? {
        guard let rawInput = rawInput,
              let data = try? JSONEncoder().encode(rawInput),
              let str = String(data: data, encoding: .utf8) else {
            return nil
        }
        return str
    }

    public init(
        toolCallId: String,
        title: String? = nil,
        kind: String? = nil,
        status: String,
        rawInput: AnyCodableValue? = nil,
        output: String? = nil,
        error: String? = nil
    ) {
        self.sessionUpdate = "tool_call"
        self.toolCallId = toolCallId
        self.title = title
        self.kind = kind
        self.status = status
        self.rawInput = rawInput
        self.output = output
        self.error = error
    }
}

public struct ACPPlanStepUpdate: Codable, Sendable {
    public let sessionUpdate: String
    public let stepId: String
    public let title: String
    public let status: String // "pending", "in_progress", "complete", "error"

    public init(stepId: String, title: String, status: String) {
        self.sessionUpdate = "plan_step"
        self.stepId = stepId
        self.title = title
        self.status = status
    }
}

public struct ACPModeChangeUpdate: Codable, Sendable {
    public let sessionUpdate: String
    public let mode: String // "agent", "plan", "user_approval"

    public init(mode: String) {
        self.sessionUpdate = "mode_change"
        self.mode = mode
    }
}

public struct ACPCurrentModeUpdate: Codable, Sendable {
    public let sessionUpdate: String
    public let modeId: String

    public init(modeId: String) {
        self.sessionUpdate = "current_mode_update"
        self.modeId = modeId
    }
}

// MARK: - Permission Request (Agent -> Client)

/// Permission option kind
public struct ACPPermissionOption: Codable, Sendable {
    public let kind: String       // "allow_always", "allow_once", "reject_once"
    public let name: String       // Display name
    public let optionId: String   // ID to send back

    public init(kind: String, name: String, optionId: String) {
        self.kind = kind
        self.name = name
        self.optionId = optionId
    }
}

/// Tool call info in permission request
public struct ACPPermissionToolCall: Codable, Sendable {
    public let toolCallId: String?
    public let rawInput: AnyCodableValue?
    public let title: String?

    public init(toolCallId: String? = nil, rawInput: AnyCodableValue? = nil, title: String? = nil) {
        self.toolCallId = toolCallId
        self.rawInput = rawInput
        self.title = title
    }
}

/// Permission request params from agent
public struct ACPRequestPermissionParams: Codable, Sendable {
    public let options: [ACPPermissionOption]
    public let sessionId: String
    public let toolName: String?
    public let toolCallId: String?
    public let input: AnyCodableValue?
    public let prompt: String?
    public let title: String?
    public let description: String?
    public let toolCall: ACPPermissionToolCall?

    public init(
        options: [ACPPermissionOption],
        sessionId: String,
        toolName: String? = nil,
        toolCallId: String? = nil,
        input: AnyCodableValue? = nil,
        prompt: String? = nil,
        title: String? = nil,
        description: String? = nil,
        toolCall: ACPPermissionToolCall? = nil
    ) {
        self.options = options
        self.sessionId = sessionId
        self.toolName = toolName
        self.toolCallId = toolCallId
        self.input = input
        self.prompt = prompt
        self.title = title
        self.description = description
        self.toolCall = toolCall
    }

    /// Get the display title for this permission request
    public var displayTitle: String? {
        toolCall?.title ?? title ?? toolName
    }

    /// Get the raw input as string
    public var rawInputString: String? {
        if let rawInput = toolCall?.rawInput,
           let data = try? JSONEncoder().encode(rawInput),
           let str = String(data: data, encoding: .utf8) {
            return str
        }
        return nil
    }
}

/// Permission response outcome
public struct ACPPermissionOutcome: Codable, Sendable {
    public let outcome: String  // "selected" or "cancelled"
    public let optionId: String?

    public init(outcome: String, optionId: String? = nil) {
        self.outcome = outcome
        self.optionId = optionId
    }

    /// Create a "selected" outcome with the given optionId
    public static func selected(_ optionId: String) -> ACPPermissionOutcome {
        ACPPermissionOutcome(outcome: "selected", optionId: optionId)
    }

    /// Create a "cancelled" outcome
    public static func cancelled() -> ACPPermissionOutcome {
        ACPPermissionOutcome(outcome: "cancelled", optionId: nil)
    }
}

/// Permission response to agent
public struct ACPRequestPermissionResult: Codable, Sendable {
    public let outcome: ACPPermissionOutcome

    public init(outcome: ACPPermissionOutcome) {
        self.outcome = outcome
    }

    /// Create a response granting permission with the given optionId
    public static func allow(_ optionId: String) -> ACPRequestPermissionResult {
        ACPRequestPermissionResult(outcome: .selected(optionId))
    }

    /// Create a response cancelling the permission request
    public static func cancel() -> ACPRequestPermissionResult {
        ACPRequestPermissionResult(outcome: .cancelled())
    }
}

// MARK: - File System Operations (Agent -> Client)

public struct ACPReadTextFileParams: Codable, Sendable {
    public let path: String
    public let startLine: Int?
    public let endLine: Int?

    public init(path: String, startLine: Int? = nil, endLine: Int? = nil) {
        self.path = path
        self.startLine = startLine
        self.endLine = endLine
    }
}

public struct ACPReadTextFileResult: Codable, Sendable {
    public let content: String
    public let lineCount: Int?

    public init(content: String, lineCount: Int? = nil) {
        self.content = content
        self.lineCount = lineCount
    }
}

public struct ACPWriteTextFileParams: Codable, Sendable {
    public let path: String
    public let content: String
    public let createDirectories: Bool?

    public init(path: String, content: String, createDirectories: Bool? = nil) {
        self.path = path
        self.content = content
        self.createDirectories = createDirectories
    }
}

public struct ACPWriteTextFileResult: Codable, Sendable {
    public let bytesWritten: Int

    public init(bytesWritten: Int) {
        self.bytesWritten = bytesWritten
    }
}

// MARK: - Terminal Operations (Agent -> Client)

/// Environment variable as sent by agent
public struct ACPEnvVar: Codable, Sendable {
    public let name: String
    public let value: String

    public init(name: String, value: String) {
        self.name = name
        self.value = value
    }
}

public struct ACPTerminalCreateParams: Codable, Sendable {
    public let command: String
    public let args: [String]?
    public let cwd: String?
    public let env: [ACPEnvVar]?  // Agent sends array of {name, value}
    public let timeout: Int?
    public let sessionId: String?
    public let outputByteLimit: Int?

    public init(command: String, args: [String]? = nil, cwd: String? = nil, env: [ACPEnvVar]? = nil, timeout: Int? = nil, sessionId: String? = nil, outputByteLimit: Int? = nil) {
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env = env
        self.timeout = timeout
        self.sessionId = sessionId
        self.outputByteLimit = outputByteLimit
    }

    /// Convert env array to dictionary
    public var envDict: [String: String]? {
        guard let env = env else { return nil }
        return Dictionary(uniqueKeysWithValues: env.map { ($0.name, $0.value) })
    }
}

public struct ACPTerminalCreateResult: Codable, Sendable {
    public let terminalId: String

    public init(terminalId: String) {
        self.terminalId = terminalId
    }
}

public struct ACPTerminalOutputParams: Codable, Sendable {
    public let terminalId: String

    public init(terminalId: String) {
        self.terminalId = terminalId
    }
}

public struct ACPTerminalOutputResult: Codable, Sendable {
    public let output: String
    public let exitCode: Int?

    public init(output: String, exitCode: Int? = nil) {
        self.output = output
        self.exitCode = exitCode
    }
}

public struct ACPTerminalReleaseParams: Codable, Sendable {
    public let terminalId: String

    public init(terminalId: String) {
        self.terminalId = terminalId
    }
}

public struct ACPTerminalKillParams: Codable, Sendable {
    public let terminalId: String

    public init(terminalId: String) {
        self.terminalId = terminalId
    }
}
