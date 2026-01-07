import Foundation

// MARK: - ACP Client Events

enum ACPEvent: Sendable {
    case connected(agentInfo: ACPAgentInfo)
    case disconnected
    case sessionCreated(sessionId: String)
    case sessionLoaded(sessionId: String)
    case textChunk(String)
    case thinkingChunk(String)
    case toolCallStarted(id: String, name: String, input: String?)
    case toolCallUpdate(id: String, status: String, output: String?, error: String?)
    case planStep(id: String, title: String, status: String)
    case modeChange(mode: String)
    case promptComplete(stopReason: String)
    case error(String)
}

// MARK: - ACP Client Delegate

protocol ACPClientDelegate: Actor {
    func acpClient(_ client: ACPClient, didReceive event: ACPEvent) async
    func acpClient(_ client: ACPClient, requestPermissionFor toolName: String, input: String?, prompt: String?) async -> (granted: Bool, context: String?)
    func acpClient(_ client: ACPClient, readFile path: String, startLine: Int?, endLine: Int?) async throws -> (content: String, lineCount: Int?)
    func acpClient(_ client: ACPClient, writeFile path: String, content: String, createDirectories: Bool) async throws -> Int
}

// Default implementations for optional delegate methods
extension ACPClientDelegate {
    func acpClient(_ client: ACPClient, requestPermissionFor toolName: String, input: String?, prompt: String?) async -> (granted: Bool, context: String?) {
        // Default: grant all permissions (for demo purposes)
        return (true, nil)
    }

    func acpClient(_ client: ACPClient, readFile path: String, startLine: Int?, endLine: Int?) async throws -> (content: String, lineCount: Int?) {
        // Default: read from local filesystem
        let url = URL(fileURLWithPath: path)
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: "\n")

        if let start = startLine, let end = endLine {
            let sliced = lines[max(0, start - 1)..<min(lines.count, end)]
            return (sliced.joined(separator: "\n"), lines.count)
        }

        return (content, lines.count)
    }

    func acpClient(_ client: ACPClient, writeFile path: String, content: String, createDirectories: Bool) async throws -> Int {
        // Default: write to local filesystem
        let url = URL(fileURLWithPath: path)

        if createDirectories {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        }

        try content.write(to: url, atomically: true, encoding: .utf8)
        return content.utf8.count
    }
}

// MARK: - ACP Client

actor ACPClient {
    // MARK: - Properties

    private var connection: (any ACPConnectionProtocol)?
    private weak var delegate: (any ACPClientDelegate)?

    private(set) var isConnected = false
    private(set) var agentInfo: ACPAgentInfo?
    private(set) var agentCapabilities: ACPAgentCapabilities?
    private(set) var currentSessionId: String?

    private let clientInfo: ACPClientInfo
    private let clientCapabilities: ACPClientCapabilities

    private var eventContinuation: AsyncStream<ACPEvent>.Continuation?

    // MARK: - Init

    init(
        clientInfo: ACPClientInfo,
        clientCapabilities: ACPClientCapabilities = ACPClientCapabilities(
            fs: ACPFSCapabilities(readTextFile: true, writeTextFile: true),
            terminal: true
        )
    ) {
        self.clientInfo = clientInfo
        self.clientCapabilities = clientCapabilities
    }

    // MARK: - Event Stream

    var events: AsyncStream<ACPEvent> {
        AsyncStream { continuation in
            self.eventContinuation = continuation
        }
    }

    func setDelegate(_ delegate: any ACPClientDelegate) {
        self.delegate = delegate
    }

    // MARK: - Connection

    #if os(macOS)
    /// Connect to an ACP agent via subprocess
    func connect(
        command: String,
        arguments: [String] = [],
        environment: [String: String]? = nil,
        currentDirectory: String? = nil
    ) async throws {
        // Close existing connection
        await disconnect()

        // Create subprocess connection
        let conn = try ACPSubprocessConnection(
            command: command,
            arguments: arguments,
            environment: environment,
            currentDirectory: currentDirectory
        )

        // Set up notification handler
        await conn.setNotificationHandler { [weak self] notification in
            await self?.handleNotification(notification)
        }

        // Start reading
        await conn.startReading()

        self.connection = conn

        // Give the remote process time to initialize (especially over SSH)
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        print("[ACPClient] Process ready, sending initialize request...")
        let result: ACPInitializeResult = try await conn.sendRequest(
            method: "initialize",
            params: ACPInitializeParams(
                clientCapabilities: clientCapabilities,
                clientInfo: clientInfo
            )
        )
        print("[ACPClient] Initialize response received: \(result.agentInfo.title)")

        isConnected = true
        agentInfo = result.agentInfo
        agentCapabilities = result.agentCapabilities

        emitEvent(.connected(agentInfo: result.agentInfo))
    }
    #endif

    /// Connect using an existing connection (for testing or custom transports)
    func connect(using connection: any ACPConnectionProtocol) async throws {
        await disconnect()

        self.connection = connection

        // Initialize the connection
        let result: ACPInitializeResult = try await connection.sendRequest(
            method: "initialize",
            params: ACPInitializeParams(
                clientCapabilities: clientCapabilities,
                clientInfo: clientInfo
            )
        )

        isConnected = true
        agentInfo = result.agentInfo
        agentCapabilities = result.agentCapabilities

        emitEvent(.connected(agentInfo: result.agentInfo))
    }

    func disconnect() async {
        if let conn = connection {
            await conn.close()
        }
        connection = nil
        isConnected = false
        currentSessionId = nil
        agentInfo = nil
        agentCapabilities = nil
        emitEvent(.disconnected)
    }

    // MARK: - Session Management

    /// Create a new session
    func newSession(cwd: String, mcpServers: [ACPMCPServer] = []) async throws -> String {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let result: ACPSessionNewResult = try await conn.sendRequest(
            method: "session/new",
            params: ACPSessionNewParams(cwd: cwd, mcpServers: mcpServers)
        )

        currentSessionId = result.sessionId
        emitEvent(.sessionCreated(sessionId: result.sessionId))
        return result.sessionId
    }

    /// Load an existing session
    func loadSession(sessionId: String, cwd: String? = nil, mcpServers: [ACPMCPServer] = []) async throws -> String {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let result: ACPSessionLoadResult = try await conn.sendRequest(
            method: "session/load",
            params: ACPSessionLoadParams(sessionId: sessionId, cwd: cwd, mcpServers: mcpServers)
        )

        currentSessionId = result.sessionId
        emitEvent(.sessionLoaded(sessionId: result.sessionId))
        return result.sessionId
    }

    // MARK: - Prompting

    /// Send a prompt to the agent
    func prompt(text: String, sessionId: String? = nil) async throws -> ACPSessionPromptResult {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let sid = sessionId ?? currentSessionId
        guard let sid = sid else {
            throw ACPConnectionError.invalidResponse("No active session")
        }

        let result: ACPSessionPromptResult = try await conn.sendRequest(
            method: "session/prompt",
            params: ACPSessionPromptParams(sessionId: sid, text: text)
        )

        emitEvent(.promptComplete(stopReason: result.stopReason))
        return result
    }

    /// Send a prompt with rich content
    func prompt(content: [ACPContentBlock], sessionId: String? = nil) async throws -> ACPSessionPromptResult {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let sid = sessionId ?? currentSessionId
        guard let sid = sid else {
            throw ACPConnectionError.invalidResponse("No active session")
        }

        let result: ACPSessionPromptResult = try await conn.sendRequest(
            method: "session/prompt",
            params: ACPSessionPromptParams(sessionId: sid, prompt: content)
        )

        emitEvent(.promptComplete(stopReason: result.stopReason))
        return result
    }

    /// Cancel the current prompt
    func cancel(sessionId: String? = nil) async throws {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let sid = sessionId ?? currentSessionId
        guard let sid = sid else { return }

        try await conn.sendNotification(
            method: "session/cancel",
            params: ACPSessionCancelParams(sessionId: sid)
        )
    }

    // MARK: - Notification Handling

    private func handleNotification(_ notification: JSONRPCNotification) async {
        switch notification.method {
        case "session/update":
            await handleSessionUpdate(notification.params)
        default:
            print("[ACPClient] Unknown notification: \(notification.method)")
        }
    }

    private func handleSessionUpdate(_ params: AnyCodableValue?) async {
        guard let params = params else { return }

        do {
            let updateParams = try params.decode(ACPSessionUpdateParams.self)

            switch updateParams.update {
            case .agentMessageChunk(let chunk):
                switch chunk.content {
                case .text(let textContent):
                    emitEvent(.textChunk(textContent.text))
                case .thinking(let thinkingContent):
                    emitEvent(.thinkingChunk(thinkingContent.thinking))
                case .toolUse(let toolUseContent):
                    let inputStr = toolUseContent.input.flatMap { input -> String? in
                        if let data = try? JSONEncoder().encode(input),
                           let str = String(data: data, encoding: .utf8) {
                            return str
                        }
                        return nil
                    }
                    emitEvent(.toolCallStarted(id: toolUseContent.id, name: toolUseContent.name, input: inputStr))
                }

            case .agentMessageComplete:
                // Message complete, no specific event needed
                break

            case .toolCall(let toolCall):
                emitEvent(.toolCallUpdate(
                    id: toolCall.toolCallId,
                    status: toolCall.status,
                    output: toolCall.output,
                    error: toolCall.error
                ))

            case .planStep(let planStep):
                emitEvent(.planStep(id: planStep.stepId, title: planStep.title, status: planStep.status))

            case .modeChange(let modeChange):
                emitEvent(.modeChange(mode: modeChange.mode))

            case .unknown(let type):
                // Silently ignore unknown update types
                acpLog("Unknown session update type: \(type)")
            }

        } catch {
            print("[ACPClient] Failed to decode session update: \(error)")
        }
    }

    // MARK: - Helpers

    private func emitEvent(_ event: ACPEvent) {
        eventContinuation?.yield(event)
        Task {
            await delegate?.acpClient(self, didReceive: event)
        }
    }
}

// MARK: - Convenience Factory

extension ACPClient {
    /// Create a client configured for Claude Code ACP
    static func forClaudeCode(
        name: String = "RemoteAgent",
        version: String = "1.0.0"
    ) -> ACPClient {
        ACPClient(
            clientInfo: ACPClientInfo(
                name: name.lowercased().replacingOccurrences(of: " ", with: "-"),
                title: name,
                version: version
            ),
            clientCapabilities: ACPClientCapabilities(
                fs: ACPFSCapabilities(readTextFile: true, writeTextFile: true),
                terminal: true
            )
        )
    }

    #if os(macOS)
    /// Connect to claude-code-acp subprocess (macOS only)
    /// Requires: npm install -g @zed-industries/claude-code-acp
    func connectToClaudeCodeACP(
        acpPath: String = "/usr/local/bin/claude-code-acp",
        currentDirectory: String? = nil,
        environment: [String: String]? = nil
    ) async throws {
        // claude-code-acp is the npm package @zed-industries/claude-code-acp
        // It wraps Claude Code and exposes it via ACP protocol on stdin/stdout
        try await connect(
            command: acpPath,
            arguments: [],
            environment: environment,
            currentDirectory: currentDirectory
        )
    }
    #endif
}
