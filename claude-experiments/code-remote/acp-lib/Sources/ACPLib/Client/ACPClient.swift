import Foundation

// MARK: - ACP Client

/// High-level ACP client with queueing, interruption, and session management
public actor ACPClient {
    // MARK: - Properties

    private var connection: (any ACPConnectionProtocol)?
    private weak var delegate: (any ACPClientDelegate)?

    private(set) public var isConnected = false
    private(set) public var agentInfo: ACPAgentInfo?
    private(set) public var agentCapabilities: ACPAgentCapabilities?
    private(set) public var currentSessionId: String?

    private let clientInfo: ACPClientInfo
    private let clientCapabilities: ACPClientCapabilities

    private var eventContinuation: AsyncStream<ACPEvent>.Continuation?

    // Feature components
    private let messageQueue = MessageQueue()
    private let promptTracker = PromptTracker()
    private let modeManager = ModeManager()
    private let terminalManager = TerminalManager()

    // MARK: - Init

    public init(
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

    public var events: AsyncStream<ACPEvent> {
        AsyncStream { continuation in
            self.eventContinuation = continuation
        }
    }

    public func setDelegate(_ delegate: any ACPClientDelegate) {
        self.delegate = delegate
    }

    /// Configure SSH for remote terminal execution (macOS only)
    /// When set, terminal commands will be executed via SSH on the remote host
    public func setSSHConfiguration(_ config: SSHConfiguration?) async {
        await terminalManager.setSSHConfiguration(config)
    }

    #if !os(macOS)
    /// Set the remote command executor for iOS terminal execution
    /// This allows terminal commands to be executed via SSH on a remote server
    public func setRemoteExecutor(_ executor: any RemoteCommandExecutor) async {
        await terminalManager.setRemoteExecutor(executor)
    }
    #endif

    // MARK: - State Accessors

    /// Whether a prompt is currently in flight
    public var isPromptInFlight: Bool {
        get async { await messageQueue.isPromptActive }
    }

    /// Available modes for the current session
    public var availableModes: [ACPMode] {
        get async { await modeManager.availableModes }
    }

    /// Current mode
    public var currentMode: ACPMode? {
        get async { await modeManager.currentMode }
    }

    /// Current mode ID
    public var currentModeId: String? {
        get async { await modeManager.currentModeId }
    }

    // MARK: - Connection

    #if os(macOS)
    /// Connect to an ACP agent via subprocess (macOS only)
    public func connect(
        command: String,
        arguments: [String] = [],
        environment: [String: String]? = nil,
        currentDirectory: String? = nil
    ) async throws {
        acpLog("ACPClient.connect: starting, command=\(command), args=\(arguments)")
        // Close existing connection
        await disconnect()

        // Create subprocess connection
        acpLog("ACPClient.connect: creating subprocess connection")
        let conn = try ACPSubprocessConnection(
            command: command,
            arguments: arguments,
            environment: environment,
            currentDirectory: currentDirectory
        )

        // Set up notification handler
        await conn.setNotificationHandler { [weak self] (notification: JSONRPCNotification) in
            acpLog("ACPClient: received notification \(notification.method)")
            await self?.handleNotification(notification)
        }

        // Set up request handler for permission requests
        await conn.setRequestHandler { [weak self] (request: JSONRPCRequest) in
            acpLog("ACPClient: received request \(request.method), id=\(request.id)")
            return await self?.handleRequest(request)
        }

        // Start reading
        await conn.startReading()
        acpLog("ACPClient.connect: reading started")

        self.connection = conn

        // Give the remote process time to initialize (especially over SSH)
        acpLog("ACPClient.connect: waiting 1 second for process to initialize...")
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        acpLog("ACPClient.connect: sending initialize request...")
        let result: ACPInitializeResult = try await conn.sendRequest(
            method: "initialize",
            params: ACPInitializeParams(
                clientCapabilities: clientCapabilities,
                clientInfo: clientInfo
            )
        )
        acpLog("ACPClient.connect: initialize response received, agent=\(result.agentInfo.title)")

        isConnected = true
        agentInfo = result.agentInfo
        agentCapabilities = result.agentCapabilities

        emitEvent(.connected(agentInfo: result.agentInfo))
        acpLog("ACPClient.connect: connected successfully")
    }
    #endif

    /// Connect using an existing connection (for testing or custom transports)
    public func connect(using connection: any ACPConnectionProtocol) async throws {
        await disconnect()

        // Set up notification handler for streaming events
        await connection.setNotificationHandler { [weak self] (notification: JSONRPCNotification) in
            acpLog("ACPClient (custom connection): received notification \(notification.method)")
            await self?.handleNotification(notification)
        }

        // Set up request handler for permission requests
        await connection.setRequestHandler { [weak self] (request: JSONRPCRequest) in
            acpLog("ACPClient (custom connection): received request \(request.method), id=\(request.id)")
            return await self?.handleRequest(request)
        }

        self.connection = connection

        // Initialize the connection
        acpLog("ACPClient.connect(using:): sending initialize request...")
        let result: ACPInitializeResult = try await connection.sendRequest(
            method: "initialize",
            params: ACPInitializeParams(
                clientCapabilities: clientCapabilities,
                clientInfo: clientInfo
            )
        )
        acpLog("ACPClient.connect(using:): initialize complete, agent=\(result.agentInfo.name)")

        isConnected = true
        agentInfo = result.agentInfo
        agentCapabilities = result.agentCapabilities

        emitEvent(.connected(agentInfo: result.agentInfo))
    }

    public func disconnect() async {
        if let conn = connection {
            await conn.close()
        }
        connection = nil
        isConnected = false
        currentSessionId = nil
        agentInfo = nil
        agentCapabilities = nil
        await messageQueue.clearActivePrompt()
        await promptTracker.clearState()
        await modeManager.clear()
        await terminalManager.clearAll()
        emitEvent(.disconnected)
    }

    // MARK: - Session Management

    /// Create a new session
    public func newSession(cwd: String, mcpServers: [ACPMCPServer] = []) async throws -> String {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let result: ACPSessionNewResult = try await conn.sendRequest(
            method: "session/new",
            params: ACPSessionNewParams(cwd: cwd, mcpServers: mcpServers)
        )

        currentSessionId = result.sessionId
        await modeManager.updateModes(from: result.modes)
        emitEvent(.sessionCreated(sessionId: result.sessionId, modes: result.modes))
        return result.sessionId
    }

    /// Load an existing session
    public func loadSession(sessionId: String, cwd: String? = nil, mcpServers: [ACPMCPServer] = []) async throws -> String {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let result: ACPSessionLoadResult = try await conn.sendRequest(
            method: "session/load",
            params: ACPSessionLoadParams(sessionId: sessionId, cwd: cwd, mcpServers: mcpServers)
        )

        currentSessionId = result.sessionId
        await modeManager.updateModes(from: result.modes)
        emitEvent(.sessionLoaded(sessionId: result.sessionId, modes: result.modes))
        return result.sessionId
    }

    /// Resume an existing session (unstable API)
    public func resumeSession(sessionId: String, cwd: String, mcpServers: [ACPMCPServer] = []) async throws -> (sessionId: String, modes: ACPModeInfo?, history: [ACPHistoryMessage]) {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let result: ACPSessionResumeResult = try await conn.sendRequest(
            method: "session/resume",
            params: ACPSessionResumeParams(sessionId: sessionId, cwd: cwd, mcpServers: mcpServers)
        )

        currentSessionId = result.sessionId
        await modeManager.updateModes(from: result.modes)

        // Load history - only available on macOS (files are local)
        // On iOS, history files are on the remote server and not accessible
        var history: [ACPHistoryMessage] = []
        #if os(macOS)
        do {
            history = try await SessionHistoryLoader.loadHistory(sessionId: sessionId, cwd: cwd)
        } catch {
            acpLog("Failed to load session history: \(error)")
        }
        #endif

        emitEvent(.sessionResumed(sessionId: result.sessionId, modes: result.modes))
        return (result.sessionId, result.modes, history)
    }

    /// Load session history from file
    public func loadSessionHistory(sessionId: String, cwd: String) async throws -> [ACPHistoryMessage] {
        try await SessionHistoryLoader.loadHistory(sessionId: sessionId, cwd: cwd)
    }

    // MARK: - Prompting

    /// Send a text prompt to the agent
    public func prompt(text: String, sessionId: String? = nil) async throws -> ACPPromptResultInfo {
        try await prompt(content: [.text(ACPTextContent(text: text))], sessionId: sessionId)
    }

    /// Send a prompt with rich content
    public func prompt(content: [ACPContentBlock], sessionId: String? = nil) async throws -> ACPPromptResultInfo {
        acpLog("ACPClient.prompt: starting")
        guard let conn = connection else {
            acpLogError("ACPClient.prompt: not connected")
            throw ACPConnectionError.notConnected
        }

        let sid = sessionId ?? currentSessionId
        guard let sid = sid else {
            acpLogError("ACPClient.prompt: no active session")
            throw ACPError.noActiveSession
        }

        // Generate new prompt ID
        let promptId = await messageQueue.generatePromptId()
        acpLog("ACPClient.prompt: generated promptId=\(promptId), sessionId=\(sid)")
        await messageQueue.setActivePrompt(promptId)
        await promptTracker.setPromptId(promptId)
        await promptTracker.clearState()

        acpLog("ACPClient.prompt: sending session/prompt request...")
        let result: ACPSessionPromptResult = try await conn.sendRequest(
            method: "session/prompt",
            params: ACPSessionPromptParams(sessionId: sid, prompt: content)
        )
        acpLog("ACPClient.prompt: received response, stopReason=\(result.stopReason)")

        await messageQueue.clearActivePrompt()
        emitEvent(.promptComplete(stopReason: result.stopReason, promptId: promptId))

        acpLog("ACPClient.prompt: completed")
        return ACPPromptResultInfo(
            stopReason: result.stopReason,
            promptId: promptId,
            wasInterrupted: result.stopReason == "cancelled"
        )
    }

    /// Interrupt any current prompt and send a new one
    public func interruptAndPrompt(text: String, sessionId: String? = nil) async throws -> ACPPromptResultInfo {
        try await interruptAndPrompt(content: [.text(ACPTextContent(text: text))], sessionId: sessionId)
    }

    /// Interrupt any current prompt and send new content
    public func interruptAndPrompt(content: [ACPContentBlock], sessionId: String? = nil) async throws -> ACPPromptResultInfo {
        let sid = sessionId ?? currentSessionId
        guard let sid = sid else {
            throw ACPError.noActiveSession
        }

        // 1. Generate new prompt ID (invalidates old one)
        let newPromptId = await messageQueue.generatePromptId()
        let oldPromptId = await messageQueue.activePromptId

        // 2. Invalidate old prompt immediately
        if let oldId = oldPromptId {
            _ = await messageQueue.invalidatePrompt(oldId)
        }

        // 3. Send cancel signal if there was an active prompt
        if oldPromptId != nil {
            try? await cancel(sessionId: sid)
        }

        // 4. Capture interrupted state for event emission
        if await promptTracker.hasContent {
            let interruptedState = await promptTracker.captureInterruptedState()
            emitEvent(.promptInterrupted(
                partialText: interruptedState.formattedText,
                promptId: oldPromptId ?? ""
            ))
        }

        // 5. Set new prompt as active
        await messageQueue.setActivePrompt(newPromptId)
        await promptTracker.setPromptId(newPromptId)

        // 6. Send new prompt
        return try await sendPromptInternal(content: content, sessionId: sid, promptId: newPromptId)
    }

    private func sendPromptInternal(content: [ACPContentBlock], sessionId: String, promptId: String) async throws -> ACPPromptResultInfo {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let result: ACPSessionPromptResult = try await conn.sendRequest(
            method: "session/prompt",
            params: ACPSessionPromptParams(sessionId: sessionId, prompt: content)
        )

        await messageQueue.clearActivePrompt()
        emitEvent(.promptComplete(stopReason: result.stopReason, promptId: promptId))

        return ACPPromptResultInfo(
            stopReason: result.stopReason,
            promptId: promptId,
            wasInterrupted: result.stopReason == "cancelled"
        )
    }

    /// Cancel the current prompt
    public func cancel(sessionId: String? = nil) async throws {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let sid = sessionId ?? currentSessionId
        guard let sid = sid else { return }

        // Capture interrupted state before clearing
        let oldPromptId = await messageQueue.activePromptId
        if await promptTracker.hasContent {
            let interruptedState = await promptTracker.captureInterruptedState()
            emitEvent(.promptInterrupted(
                partialText: interruptedState.formattedText,
                promptId: oldPromptId ?? ""
            ))
        }

        await messageQueue.clearActivePrompt()

        try await conn.sendNotification(
            method: "session/cancel",
            params: ACPSessionCancelParams(sessionId: sid)
        )
    }

    // MARK: - Mode Management

    /// Set the session mode
    public func setMode(_ modeId: String, sessionId: String? = nil) async throws {
        guard let conn = connection else {
            throw ACPConnectionError.notConnected
        }

        let sid = sessionId ?? currentSessionId
        guard let sid = sid else {
            throw ACPError.noActiveSession
        }

        let _: ACPSetSessionModeResult = try await conn.sendRequest(
            method: "session/set_mode",
            params: ACPSetSessionModeParams(sessionId: sid, modeId: modeId)
        )

        await modeManager.setCurrentMode(modeId)
        emitEvent(.modeChanged(modeId: modeId))
    }

    /// Cycle to the next mode
    public func cycleMode(sessionId: String? = nil) async throws -> ACPMode {
        guard let nextMode = await modeManager.nextMode() else {
            throw ACPError.invalidState("No modes available")
        }

        try await setMode(nextMode.id, sessionId: sessionId)
        return nextMode
    }

    // MARK: - Request Handling

    private func handleRequest(_ request: JSONRPCRequest) async -> JSONRPCResponse? {
        switch request.method {
        case "session/request_permission":
            return await handlePermissionRequest(request)
        case "terminal/create":
            return await handleTerminalCreate(request)
        case "terminal/output":
            return await handleTerminalOutput(request)
        case "terminal/wait_for_exit":
            return await handleTerminalWaitForExit(request)
        case "terminal/kill":
            return await handleTerminalKill(request)
        case "terminal/release":
            return await handleTerminalRelease(request)
        case "fs/read_text_file":
            return await handleFsReadTextFile(request)
        default:
            acpLog("Unknown incoming request: \(request.method)")
            return nil
        }
    }

    private func handlePermissionRequest(_ request: JSONRPCRequest) async -> JSONRPCResponse {
        acpLog("handlePermissionRequest: processing id=\(request.id)")

        do {
            guard let params = request.params else {
                acpLogError("handlePermissionRequest: missing params")
                return .error(JSONRPCErrorResponse(
                    jsonrpc: jsonrpcVersion,
                    id: request.id,
                    error: JSONRPCError(code: -32602, message: "Missing params", data: nil)
                ))
            }

            let permissionParams = try params.decode(ACPRequestPermissionParams.self)
            let displayTitle = permissionParams.displayTitle ?? "unknown"
            acpLog("handlePermissionRequest: tool=\(displayTitle), options=\(permissionParams.options.map { $0.optionId })")

            // Get input as string for delegate - prefer rawInput from toolCall
            let inputStr = permissionParams.rawInputString ?? {
                if let input = permissionParams.input,
                   let data = try? JSONEncoder().encode(input),
                   let str = String(data: data, encoding: .utf8) {
                    return str
                }
                return nil
            }()

            // Ask delegate for permission, or DENY if no delegate
            let (granted, _): (Bool, String?)
            if let delegate = delegate {
                acpLog("handlePermissionRequest: asking delegate for permission")
                (granted, _) = await delegate.acpClient(
                    self,
                    requestPermissionFor: displayTitle,
                    input: inputStr,
                    prompt: permissionParams.prompt
                )
            } else {
                // DENY if no delegate - permissions should be explicit
                acpLog("handlePermissionRequest: no delegate, DENYING permission")
                granted = false
            }

            // Find the appropriate optionId based on granted
            let optionId: String
            if granted {
                // Prefer "allow_once" if available, otherwise first allow option
                optionId = permissionParams.options.first { $0.kind == "allow_once" }?.optionId
                    ?? permissionParams.options.first { $0.kind.starts(with: "allow") }?.optionId
                    ?? "allow"
            } else {
                // Use reject option
                optionId = permissionParams.options.first { $0.kind.starts(with: "reject") }?.optionId
                    ?? "reject"
            }

            acpLog("handlePermissionRequest: granted=\(granted), optionId=\(optionId)")

            let result = ACPRequestPermissionResult.allow(optionId)
            return .success(JSONRPCSuccessResponse(
                id: request.id,
                result: AnyCodableValue(result)
            ))
        } catch {
            acpLogError("handlePermissionRequest: failed to decode params: \(error)")
            return .error(JSONRPCErrorResponse(
                jsonrpc: jsonrpcVersion,
                id: request.id,
                error: JSONRPCError(code: -32602, message: "Invalid params: \(error.localizedDescription)", data: nil)
            ))
        }
    }

    // MARK: - Terminal Handlers

    private func handleTerminalCreate(_ request: JSONRPCRequest) async -> JSONRPCResponse {
        acpLog("handleTerminalCreate: processing id=\(request.id)")

        do {
            guard let params = request.params else {
                return makeErrorResponse(request.id, code: -32602, message: "Missing params")
            }

            let createParams = try params.decode(ACPTerminalCreateParams.self)
            acpLog("handleTerminalCreate: command=\(createParams.command)")

            let terminalId = try await terminalManager.createTerminal(
                command: createParams.command,
                args: createParams.args,
                cwd: createParams.cwd,
                env: createParams.envDict
            )

            acpLog("handleTerminalCreate: created terminal \(terminalId)")
            let result = ACPTerminalCreateResult(terminalId: terminalId)
            return .success(JSONRPCSuccessResponse(
                id: request.id,
                result: AnyCodableValue(result)
            ))
        } catch {
            acpLogError("handleTerminalCreate: error: \(error)")
            return makeErrorResponse(request.id, code: -32000, message: "Failed to create terminal: \(error.localizedDescription)")
        }
    }

    private func handleTerminalOutput(_ request: JSONRPCRequest) async -> JSONRPCResponse {
        acpLog("handleTerminalOutput: processing id=\(request.id)")

        do {
            guard let params = request.params else {
                return makeErrorResponse(request.id, code: -32602, message: "Missing params")
            }

            let outputParams = try params.decode(ACPTerminalOutputParams.self)
            let (output, exitStatus) = await terminalManager.getOutput(terminalId: outputParams.terminalId)

            acpLog("handleTerminalOutput: terminal=\(outputParams.terminalId), outputLen=\(output.count), exited=\(exitStatus != nil)")

            var result = ACPTerminalOutputResult(output: output, exitCode: exitStatus.flatMap { Int($0.exitCode ?? 0) })
            return .success(JSONRPCSuccessResponse(
                id: request.id,
                result: AnyCodableValue(result)
            ))
        } catch {
            acpLogError("handleTerminalOutput: error: \(error)")
            return makeErrorResponse(request.id, code: -32000, message: "Failed to get output: \(error.localizedDescription)")
        }
    }

    private func handleTerminalWaitForExit(_ request: JSONRPCRequest) async -> JSONRPCResponse {
        acpLog("handleTerminalWaitForExit: processing id=\(request.id)")

        do {
            guard let params = request.params else {
                return makeErrorResponse(request.id, code: -32602, message: "Missing params")
            }

            let waitParams = try params.decode(ACPTerminalOutputParams.self)
            let (exitCode, signal) = await terminalManager.waitForExit(terminalId: waitParams.terminalId)

            acpLog("handleTerminalWaitForExit: terminal=\(waitParams.terminalId), exitCode=\(exitCode ?? -1)")

            struct WaitResult: Codable {
                let exitCode: Int32?
                let signal: String?
            }
            let result = WaitResult(exitCode: exitCode, signal: signal)
            return .success(JSONRPCSuccessResponse(
                id: request.id,
                result: AnyCodableValue(result)
            ))
        } catch {
            acpLogError("handleTerminalWaitForExit: error: \(error)")
            return makeErrorResponse(request.id, code: -32000, message: "Failed to wait: \(error.localizedDescription)")
        }
    }

    private func handleTerminalKill(_ request: JSONRPCRequest) async -> JSONRPCResponse {
        acpLog("handleTerminalKill: processing id=\(request.id)")

        do {
            guard let params = request.params else {
                return makeErrorResponse(request.id, code: -32602, message: "Missing params")
            }

            let killParams = try params.decode(ACPTerminalKillParams.self)
            await terminalManager.killTerminal(terminalId: killParams.terminalId)

            acpLog("handleTerminalKill: killed terminal \(killParams.terminalId)")

            struct EmptyResult: Codable {}
            return .success(JSONRPCSuccessResponse(
                id: request.id,
                result: AnyCodableValue(EmptyResult())
            ))
        } catch {
            acpLogError("handleTerminalKill: error: \(error)")
            return makeErrorResponse(request.id, code: -32000, message: "Failed to kill: \(error.localizedDescription)")
        }
    }

    private func handleTerminalRelease(_ request: JSONRPCRequest) async -> JSONRPCResponse {
        acpLog("handleTerminalRelease: processing id=\(request.id)")

        do {
            guard let params = request.params else {
                return makeErrorResponse(request.id, code: -32602, message: "Missing params")
            }

            let releaseParams = try params.decode(ACPTerminalReleaseParams.self)
            await terminalManager.releaseTerminal(terminalId: releaseParams.terminalId)

            acpLog("handleTerminalRelease: released terminal \(releaseParams.terminalId)")

            struct EmptyResult: Codable {}
            return .success(JSONRPCSuccessResponse(
                id: request.id,
                result: AnyCodableValue(EmptyResult())
            ))
        } catch {
            acpLogError("handleTerminalRelease: error: \(error)")
            return makeErrorResponse(request.id, code: -32000, message: "Failed to release: \(error.localizedDescription)")
        }
    }

    // MARK: - File System Handlers

    private func handleFsReadTextFile(_ request: JSONRPCRequest) async -> JSONRPCResponse {
        acpLog("handleFsReadTextFile: processing id=\(request.id)")

        do {
            guard let params = request.params else {
                return makeErrorResponse(request.id, code: -32602, message: "Missing params")
            }

            struct FsReadParams: Codable {
                let sessionId: String
                let path: String
                let line: Int?
                let limit: Int?
            }

            let readParams = try params.decode(FsReadParams.self)
            acpLog("handleFsReadTextFile: path=\(readParams.path), line=\(readParams.line ?? 1), limit=\(readParams.limit ?? 2000)")

            // Read file via remote executor (SSH) or locally
            let content: String
            let lineCount: Int

            #if os(iOS)
            // On iOS, use remote executor to read file via SSH
            if let remoteContent = await terminalManager.readRemoteFile(readParams.path) {
                content = remoteContent
            } else {
                return makeErrorResponse(request.id, code: -32000, message: "Failed to read file: \(readParams.path)")
            }
            #else
            // On macOS, read locally or via SSH if configured
            if await terminalManager.hasRemoteExecutor {
                if let remoteContent = await terminalManager.readRemoteFile(readParams.path) {
                    content = remoteContent
                } else {
                    return makeErrorResponse(request.id, code: -32000, message: "Failed to read file: \(readParams.path)")
                }
            } else {
                // Read locally
                let url = URL(fileURLWithPath: readParams.path)
                content = try String(contentsOf: url, encoding: .utf8)
            }
            #endif

            // Apply line offset and limit
            let lines = content.components(separatedBy: "\n")
            lineCount = lines.count
            let startLine = max(0, (readParams.line ?? 1) - 1)
            let endLine = min(lines.count, startLine + (readParams.limit ?? 2000))
            let slicedContent = lines[startLine..<endLine].joined(separator: "\n")

            acpLog("handleFsReadTextFile: read \(lineCount) lines, returning \(endLine - startLine) lines")

            struct FsReadResult: Codable {
                let content: String
                let lineCount: Int
            }

            let result = FsReadResult(content: slicedContent, lineCount: lineCount)
            return .success(JSONRPCSuccessResponse(
                id: request.id,
                result: AnyCodableValue(result)
            ))
        } catch {
            acpLogError("handleFsReadTextFile: error: \(error)")
            return makeErrorResponse(request.id, code: -32000, message: "Failed to read file: \(error.localizedDescription)")
        }
    }

    private func makeErrorResponse(_ id: JSONRPCId, code: Int, message: String) -> JSONRPCResponse {
        .error(JSONRPCErrorResponse(
            jsonrpc: jsonrpcVersion,
            id: id,
            error: JSONRPCError(code: code, message: message, data: nil)
        ))
    }

    // MARK: - Notification Handling

    private func handleNotification(_ notification: JSONRPCNotification) async {
        switch notification.method {
        case "session/update":
            await handleSessionUpdate(notification.params)
        default:
            acpLog("Unknown notification: \(notification.method)")
        }
    }

    private func handleSessionUpdate(_ params: AnyCodableValue?) async {
        guard let params = params else {
            acpLog("handleSessionUpdate: params is nil")
            return
        }

        // Get current prompt ID for event filtering
        let currentPromptId = await messageQueue.activePromptId ?? ""
        let isActive = await messageQueue.isPromptActive

        // Check if we should process this event
        guard isActive else {
            acpLog("handleSessionUpdate: ignoring - no active prompt (promptId=\(currentPromptId))")
            return
        }

        acpLog("handleSessionUpdate: processing with promptId=\(currentPromptId)")

        do {
            let updateParams = try params.decode(ACPSessionUpdateParams.self)

            switch updateParams.update {
            case .agentMessageChunk(let chunk):
                switch chunk.content {
                case .text(let textContent):
                    acpLogDebug("handleSessionUpdate: textChunk, length=\(textContent.text.count)")
                    await promptTracker.appendText(textContent.text)
                    emitEvent(.textChunk(textContent.text, promptId: currentPromptId))
                case .thinking(let thinkingContent):
                    acpLogDebug("handleSessionUpdate: thinkingChunk, length=\(thinkingContent.thinking.count)")
                    emitEvent(.thinkingChunk(thinkingContent.thinking, promptId: currentPromptId))
                case .toolUse(let toolUseContent):
                    let inputStr = toolUseContent.input.flatMap { input -> String? in
                        if let data = try? JSONEncoder().encode(input),
                           let str = String(data: data, encoding: .utf8) {
                            return str
                        }
                        return nil
                    }
                    acpLog("handleSessionUpdate: toolUse STARTED, id=\(toolUseContent.id), name=\(toolUseContent.name)")
                    await promptTracker.startToolCall(id: toolUseContent.id, name: toolUseContent.name, input: inputStr)
                    emitEvent(.toolCallStarted(id: toolUseContent.id, name: toolUseContent.name, input: inputStr, promptId: currentPromptId))
                }

            case .agentMessageComplete:
                acpLog("handleSessionUpdate: agentMessageComplete")
                // Message complete, no specific event needed
                break

            case .toolCall(let toolCall):
                let status = ACPToolCallStatus(rawValue: toolCall.status) ?? .pending
                acpLog("handleSessionUpdate: toolCall UPDATE, id=\(toolCall.toolCallId), status=\(toolCall.status), title=\(toolCall.title ?? "nil"), hasOutput=\(toolCall.output != nil), hasError=\(toolCall.error != nil)")
                if let output = toolCall.output {
                    acpLogDebug("handleSessionUpdate: toolCall output preview: \(String(output.prefix(200)))")
                }
                if let error = toolCall.error {
                    acpLogError("handleSessionUpdate: toolCall error: \(error)")
                }
                await promptTracker.completeToolCall(id: toolCall.toolCallId, status: status, output: toolCall.output, error: toolCall.error)
                emitEvent(.toolCallUpdate(
                    id: toolCall.toolCallId,
                    status: toolCall.status,
                    title: toolCall.title,
                    input: toolCall.rawInputString,
                    output: toolCall.output,
                    error: toolCall.error,
                    promptId: currentPromptId
                ))
                acpLog("handleSessionUpdate: toolCall UPDATE emitted for id=\(toolCall.toolCallId)")

            case .planStep(let planStep):
                acpLog("handleSessionUpdate: planStep, id=\(planStep.stepId), title=\(planStep.title), status=\(planStep.status)")
                emitEvent(.planStep(id: planStep.stepId, title: planStep.title, status: planStep.status, promptId: currentPromptId))

            case .modeChange(let modeChange):
                acpLog("handleSessionUpdate: modeChange, mode=\(modeChange.mode)")
                await modeManager.setCurrentMode(modeChange.mode)
                emitEvent(.modeChanged(modeId: modeChange.mode))

            case .currentModeUpdate(let update):
                acpLog("handleSessionUpdate: currentModeUpdate, modeId=\(update.modeId)")
                await modeManager.setCurrentMode(update.modeId)
                emitEvent(.modeChanged(modeId: update.modeId))

            case .unknown(let type):
                acpLog("handleSessionUpdate: unknown update type: \(type)")
            }

        } catch {
            acpLogError("handleSessionUpdate: failed to decode: \(error)")
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

public extension ACPClient {
    /// Create a client configured for Claude Code ACP
    static func forClaudeCode(
        name: String = "ACPLib",
        version: String = "1.0.0"
    ) -> ACPClient {
        ACPClient(
            clientInfo: ACPClientInfo(
                name: name.lowercased().replacingOccurrences(of: " ", with: "-"),
                title: name,
                version: version
            ),
            clientCapabilities: ACPClientCapabilities(
                // Don't advertise fs capabilities - agent should use its own tools
                // to read/write files on the server where it's running
                fs: nil,
                terminal: true
            )
        )
    }

    #if os(macOS)
    /// Connect to claude-code-acp subprocess (macOS only)
    /// Requires: npm install -g @anthropics/claude-code-acp
    func connectToClaudeCodeACP(
        acpPath: String = "/usr/local/bin/claude-code-acp",
        currentDirectory: String? = nil,
        environment: [String: String]? = nil
    ) async throws {
        // claude-code-acp is the npm package
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
