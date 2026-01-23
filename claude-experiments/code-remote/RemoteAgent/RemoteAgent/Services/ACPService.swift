import Foundation
import ACPLib

private func log(_ message: String) {
    appLog(message, category: "ACPService")
}

// MARK: - ACP Service

/// Service for interacting with Claude Code via ACP protocol
/// This replaces the SSH-based approach with the standardized ACP protocol
@MainActor
class ACPService: ObservableObject {
    // Static tracker to prevent duplicate connections from SwiftUI view recreation
    private static var activeConnections: Set<String> = []

    private var client: ACPClient?
    #if !os(macOS)
    private var sshConnection: ACPSSHConnection?
    #endif
    private var eventTask: Task<Void, Never>?
    private var permissionDelegate: ACPPermissionDelegate?
    private var connectionKey: String?  // Tracks which connection this instance owns

    @Published var isConnected = false
    @Published var sessionId: String?
    @Published var agentInfo: ACPAgentInfo?
    @Published var availableModes: [ACPMode] = []
    @Published var currentMode: ACPMode?
    @Published var pendingPermissionRequest: PendingPermissionRequest?

    private var eventContinuation: AsyncStream<ACPServiceEvent>.Continuation?

    private var permissionContinuation: CheckedContinuation<Bool, Never>?

    init() {}

    var events: AsyncStream<ACPServiceEvent> {
        AsyncStream { continuation in
            self.eventContinuation = continuation
        }
    }

    // MARK: - Permission Response

    /// Respond to a pending permission request
    func respondToPermission(granted: Bool) {
        log("respondToPermission: granted=\(granted)")
        pendingPermissionRequest = nil
        permissionContinuation?.resume(returning: granted)
        permissionContinuation = nil
    }

    // MARK: - Connection

    /// Check if claude-code-acp is installed locally
    static func isClaudeCodeACPInstalled() -> Bool {
        #if os(macOS)
        let possiblePaths = [
            "/usr/local/bin/claude-code-acp",
            "/opt/homebrew/bin/claude-code-acp",
            "\(FileManager.default.homeDirectoryForCurrentUser.path)/.npm-global/bin/claude-code-acp"
        ]

        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                return true
            }
        }

        // Check if in PATH
        let result = Process()
        result.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        result.arguments = ["claude-code-acp"]
        let pipe = Pipe()
        result.standardOutput = pipe
        result.standardError = pipe
        try? result.run()
        result.waitUntilExit()
        return result.terminationStatus == 0
        #else
        // Local ACP not supported on iOS
        return false
        #endif
    }

    /// Check if claude-code-acp is installed on a remote server via SSH
    static func isClaudeCodeACPInstalledRemote(
        server: Server
    ) async -> Bool {
        do {
            log("isClaudeCodeACPInstalledRemote: checking \(server.username)@\(server.host)")
            log("isClaudeCodeACPInstalledRemote: authMethod=\(server.authMethod), keyPath=\(server.privateKeyPath ?? "nil")")

            let sshService = SSHService()

            // Get password from keychain if using password auth
            var password: String? = nil
            if server.authMethod == .password {
                password = try? await KeychainService.shared.getPassword(for: server.id)
                log("isClaudeCodeACPInstalledRemote: using password auth, hasPassword=\(password != nil)")
            }

            try await sshService.connect(to: server, password: password)

            let output = try await sshService.executeCommand("which claude-code-acp || command -v claude-code-acp")
            log("isClaudeCodeACPInstalledRemote: command output='\(output.trimmingCharacters(in: .whitespacesAndNewlines))'")
            await sshService.disconnect()

            let isInstalled = !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            log("isClaudeCodeACPInstalledRemote: isInstalled=\(isInstalled)")
            return isInstalled
        } catch {
            log("isClaudeCodeACPInstalledRemote: ERROR - \(error)")
            return false
        }
    }

    /// Install claude-code-acp locally
    static func installClaudeCodeACPLocal(
        onOutput: @escaping @Sendable (String) -> Void
    ) async -> Result<Void, Error> {
        #if os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["npm", "install", "-g", "@zed-industries/claude-code-acp"]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        // Set up async reading for stdout
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    onOutput(output)
                }
            }
        }

        // Set up async reading for stderr
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    onOutput(output)
                }
            }
        }

        do {
            try process.run()
        } catch {
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil
            return .failure(error)
        }

        // Wait in background
        return await Task.detached {
            process.waitUntilExit()

            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil

            if process.terminationStatus == 0 {
                return .success(())
            } else {
                return .failure(ACPServiceError.connectionFailed("Installation failed with exit code \(process.terminationStatus)"))
            }
        }.value
        #else
        return .failure(ACPServiceError.connectionFailed("Local installation not supported on iOS"))
        #endif
    }

    /// Install claude-code-acp on a remote server via SSH
    static func installClaudeCodeACPRemote(
        server: Server,
        onOutput: @escaping @Sendable (String) -> Void
    ) async -> Result<Void, Error> {
        do {
            log("installClaudeCodeACPRemote: installing on \(server.username)@\(server.host)")

            let sshService = SSHService()

            // Get password from keychain if using password auth
            var password: String? = nil
            if server.authMethod == .password {
                password = try? await KeychainService.shared.getPassword(for: server.id)
            }

            try await sshService.connect(to: server, password: password)

            // Install command with PATH setup for common node locations
            let installCommand = "export PATH=$PATH:$HOME/.nvm/versions/node/*/bin:$HOME/.npm-global/bin:/usr/local/bin && npm install -g @zed-industries/claude-code-acp"

            await MainActor.run { onOutput("Connecting to \(server.host)...\n") }
            await MainActor.run { onOutput("Running: npm install -g @zed-industries/claude-code-acp\n") }

            let output = try await sshService.executeCommand(installCommand)
            await MainActor.run { onOutput(output) }

            await sshService.disconnect()
            return .success(())
        } catch {
            log("installClaudeCodeACPRemote: ERROR - \(error)")
            return .failure(error)
        }
    }

    /// Connect to a local Claude Code ACP agent
    /// Requires: npm install -g @zed-industries/claude-code-acp
    func connectLocal(
        acpPath: String? = nil,
        workingDirectory: String
    ) async throws {
        #if os(macOS)
        log("connectLocal: starting, workingDirectory=\(workingDirectory)")
        await disconnect()

        // Find claude-code-acp binary
        let binaryPath = acpPath ?? findClaudeCodeACP()
        log("connectLocal: using binary at \(binaryPath)")

        // Check if binary exists
        if binaryPath == "claude-code-acp" && !ACPService.isClaudeCodeACPInstalled() {
            log("connectLocal: claude-code-acp not installed")
            throw ACPServiceError.claudeCodeACPNotInstalled
        }

        let client = ACPClient.forClaudeCode(name: "RemoteAgent", version: "1.0.0")
        self.client = client

        // Set up permission delegate with handler
        let delegate = ACPPermissionDelegate()
        self.permissionDelegate = delegate
        await setupPermissionHandler(delegate)
        await client.setDelegate(delegate)
        log("connectLocal: permission delegate set")

        // Start listening for events before connecting
        startEventListener()
        log("connectLocal: event listener started")

        // Connect to Claude Code ACP
        // claude-code-acp is installed via: npm install -g @zed-industries/claude-code-acp
        try await client.connect(
            command: binaryPath,
            arguments: [],
            currentDirectory: workingDirectory
        )

        isConnected = true
        agentInfo = await client.agentInfo
        log("connectLocal: connected, agentInfo=\(String(describing: self.agentInfo))")
        eventContinuation?.yield(.connected)
        #else
        throw ACPServiceError.connectionFailed("Local connections are not supported on iOS. Please use a remote server.")
        #endif
    }

    /// Connect to a remote Claude Code ACP agent via SSH tunnel
    /// This creates an SSH tunnel and then connects via ACP over the tunnel
    /// Requires claude-code-acp to be installed on the remote machine
    func connectRemote(
        sshHost: String,
        sshPort: Int = 22,
        sshUsername: String,
        sshKeyPath: String?,
        sshPassword: String? = nil,
        acpPath: String = "claude-code-acp",
        workingDirectory: String
    ) async throws {
        #if os(macOS)
        await disconnect()

        // For remote connections, we use SSH to run claude-code-acp remotely
        // The ACP messages are piped through SSH stdin/stdout
        let client = ACPClient.forClaudeCode(name: "RemoteAgent", version: "1.0.0")
        self.client = client

        // Configure SSH for terminal execution - this ensures terminal commands
        // run on the remote server, not locally
        let sshConfig = SSHConfiguration(
            host: sshHost,
            username: sshUsername,
            keyPath: sshKeyPath,
            remoteWorkingDirectory: workingDirectory
        )
        await client.setSSHConfiguration(sshConfig)
        log("connectRemote: SSH terminal configuration set for \(sshUsername)@\(sshHost)")

        // Set up permission delegate with handler
        let delegate = ACPPermissionDelegate()
        self.permissionDelegate = delegate
        await setupPermissionHandler(delegate)
        await client.setDelegate(delegate)
        log("connectRemote: permission delegate set")

        startEventListener()

        // Build SSH command to run claude-code-acp remotely
        var sshArgs = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]

        if sshPort != 22 {
            sshArgs.append(contentsOf: ["-p", String(sshPort)])
        }

        if let keyPath = sshKeyPath, !keyPath.isEmpty {
            sshArgs.append(contentsOf: ["-i", (keyPath as NSString).expandingTildeInPath])
        }

        sshArgs.append("\(sshUsername)@\(sshHost)")
        sshArgs.append("cd \(workingDirectory) && \(acpPath)")

        try await client.connect(
            command: "/usr/bin/ssh",
            arguments: sshArgs
        )

        isConnected = true
        agentInfo = await client.agentInfo
        eventContinuation?.yield(.connected)
        #else
        // iOS: Use ACPSSHConnection with the real ACPClient
        let totalStart = Date()
        log("connectRemote: iOS using ACPSSHConnection + ACPClient")
        log("connectRemote: host=\(sshHost), port=\(sshPort), username=\(sshUsername)")
        log("connectRemote: workingDirectory=\(workingDirectory), acpPath=\(acpPath)")
        log("connectRemote: hasKeyPath=\(sshKeyPath != nil), hasPassword=\(sshPassword != nil)")

        // Prevent duplicate connections from SwiftUI view recreation
        let key = "\(sshHost):\(workingDirectory)"
        if Self.activeConnections.contains(key) {
            log("connectRemote: SKIPPING - connection already active for \(key)")
            return
        }

        // Disconnect any existing connection FIRST (before registering new key)
        await disconnect()

        // Now register our connection key
        Self.activeConnections.insert(key)
        self.connectionKey = key
        log("connectRemote: registered connection key \(key), active=\(Self.activeConnections.count)")

        // Create SSH connection
        var stepStart = Date()
        log("connectRemote: creating ACPSSHConnection")
        let connection = ACPSSHConnection()
        self.sshConnection = connection

        // Connect via SSH first (this starts the output reader)
        log("connectRemote: connecting via SSH...")
        try await connection.connect(
            host: sshHost,
            port: sshPort,
            username: sshUsername,
            privateKeyPath: sshKeyPath,
            password: sshPassword,
            workingDirectory: workingDirectory,
            acpPath: acpPath
        )
        log("connectRemote: SSH connection established in \(Date().timeIntervalSince(stepStart))s")

        // Create ACPClient
        stepStart = Date()
        log("connectRemote: creating ACPClient")
        let acpClient = ACPClient.forClaudeCode(name: "RemoteAgent", version: "1.0.0")
        self.client = acpClient
        log("connectRemote: ACPClient created in \(Date().timeIntervalSince(stepStart))s")

        // Set up permission delegate
        stepStart = Date()
        log("connectRemote: creating permission delegate")
        let delegate = ACPPermissionDelegate()
        log("connectRemote: delegate created in \(Date().timeIntervalSince(stepStart))s")

        stepStart = Date()
        self.permissionDelegate = delegate
        log("connectRemote: delegate stored in \(Date().timeIntervalSince(stepStart))s")

        stepStart = Date()
        await acpClient.setDelegate(delegate)
        log("connectRemote: setDelegate completed in \(Date().timeIntervalSince(stepStart))s")

        stepStart = Date()
        await setupPermissionHandler(delegate)
        log("connectRemote: setupPermissionHandler completed in \(Date().timeIntervalSince(stepStart))s")

        // Connect ACPClient using our SSH connection
        // This will set up notification/request handlers and send initialize
        stepStart = Date()
        log("connectRemote: calling acpClient.connect(using: connection)")
        try await acpClient.connect(using: connection)
        log("connectRemote: ACPClient connected (initialize) in \(Date().timeIntervalSince(stepStart))s")

        // Set up remote executor for terminal commands on iOS
        // This allows Claude to run bash commands via SSH
        log("connectRemote: setting remote executor for terminals")
        await acpClient.setRemoteExecutor(connection)

        // Start listening for events from ACPClient
        log("connectRemote: starting event listener")
        startEventListener()

        isConnected = true
        agentInfo = await acpClient.agentInfo
        log("connectRemote: iOS connected in \(Date().timeIntervalSince(totalStart))s total, agent=\(String(describing: self.agentInfo))")
        eventContinuation?.yield(.connected)
        #endif
    }

    /// Connect to an existing ACP session (iOS only)
    func connectToExistingSession(
        sessionId: String,
        host: String,
        port: Int = 22,
        username: String,
        privateKeyPath: String?,
        password: String?
    ) async throws {
        #if os(iOS)
        log("connectToExistingSession: reconnecting to session \(sessionId)")

        await disconnect()

        // Create SSH connection with existing session ID
        let connection = ACPSSHConnection(existingSessionId: sessionId)
        self.sshConnection = connection

        // Connect via SSH (this will reconnect to the existing session)
        try await connection.connect(
            host: host,
            port: port,
            username: username,
            privateKeyPath: privateKeyPath,
            password: password,
            workingDirectory: "/tmp", // Not used for reconnection
            acpPath: "claude-code-acp" // Not used for reconnection
        )
        log("connectToExistingSession: SSH connection established")

        // Create ACPClient
        let acpClient = ACPClient.forClaudeCode(name: "RemoteAgent", version: "1.0.0")
        self.client = acpClient

        // Set up permission delegate
        let delegate = ACPPermissionDelegate()
        self.permissionDelegate = delegate
        await acpClient.setDelegate(delegate)
        await setupPermissionHandler(delegate)

        // Connect ACPClient using our SSH connection
        try await acpClient.connect(using: connection)
        log("connectToExistingSession: ACPClient connected")

        // Set up remote executor
        await acpClient.setRemoteExecutor(connection)

        startEventListener()

        isConnected = true
        agentInfo = await acpClient.agentInfo
        self.sessionId = sessionId
        log("connectToExistingSession: connected to existing session")
        eventContinuation?.yield(.connected)
        #else
        throw ACPServiceError.connectionFailed("connectToExistingSession is only supported on iOS")
        #endif
    }

    /// Set up permission handler on delegate
    private func setupPermissionHandler(_ delegate: ACPPermissionDelegate) async {
        await delegate.setPermissionHandler { [weak self] toolName, input, prompt in
            guard let self = self else { return (false, "Service deallocated") }

            // Create permission request and wait for UI response
            let granted = await withCheckedContinuation { continuation in
                Task { @MainActor in
                    let requestId = UUID().uuidString
                    self.permissionContinuation = continuation
                    self.pendingPermissionRequest = PendingPermissionRequest(
                        id: requestId,
                        toolName: toolName,
                        input: input,
                        title: prompt,
                        continuation: continuation
                    )
                    // Emit event for UI
                    self.eventContinuation?.yield(.permissionRequest(
                        id: requestId,
                        toolName: toolName,
                        input: input,
                        title: prompt
                    ))
                }
            }

            return (granted, nil)
        }
    }

    /// Find the claude-code-acp binary
    private func findClaudeCodeACP() -> String {
        #if os(macOS)
        // Common locations for npm global binaries
        let possiblePaths = [
            "/usr/local/bin/claude-code-acp",
            "/opt/homebrew/bin/claude-code-acp",
            "\(FileManager.default.homeDirectoryForCurrentUser.path)/.npm-global/bin/claude-code-acp",
            "\(FileManager.default.homeDirectoryForCurrentUser.path)/.nvm/versions/node/*/bin/claude-code-acp"
        ]

        for path in possiblePaths {
            if path.contains("*") {
                // Handle glob pattern for nvm
                let basePath = (path as NSString).deletingLastPathComponent
                let parentPath = (basePath as NSString).deletingLastPathComponent
                if let contents = try? FileManager.default.contentsOfDirectory(atPath: parentPath) {
                    for dir in contents.sorted().reversed() {
                        let fullPath = "\(parentPath)/\(dir)/bin/claude-code-acp"
                        if FileManager.default.fileExists(atPath: fullPath) {
                            return fullPath
                        }
                    }
                }
            } else if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        #endif

        // Fallback - assume it's in PATH
        return "claude-code-acp"
    }

    func disconnect() async {
        // Log call stack to understand what's triggering disconnect
        let callStack = Thread.callStackSymbols.prefix(10).joined(separator: "\n")
        log("disconnect: starting, call stack:\n\(callStack)")

        // Clean up connection tracking
        if let key = connectionKey {
            Self.activeConnections.remove(key)
            log("disconnect: removed connection key \(key), active=\(Self.activeConnections.count)")
            connectionKey = nil
        }

        eventTask?.cancel()
        eventTask = nil

        #if os(macOS)
        if let client = client {
            // Clear SSH configuration
            await client.setSSHConfiguration(nil)
            await client.disconnect()
        }
        client = nil
        #else
        if let client = client {
            await client.disconnect()
        }
        client = nil
        if let sshConnection = sshConnection {
            // Use disconnect() to keep the ACP session alive for reconnection
            await sshConnection.disconnect()
        }
        sshConnection = nil
        #endif

        isConnected = false
        sessionId = nil
        agentInfo = nil
        eventContinuation?.yield(.disconnected)
        log("disconnect: completed")
    }

    // MARK: - Session Management

    func newSession(workingDirectory: String) async throws -> String {
        log("newSession: starting, workingDirectory=\(workingDirectory)")

        guard let client = client else {
            log("newSession: not connected")
            throw ACPServiceError.notConnected
        }

        let id = try await client.newSession(cwd: workingDirectory)
        sessionId = id
        log("newSession: created session \(id)")
        eventContinuation?.yield(.sessionCreated(sessionId: id))
        return id
    }

    func loadSession(sessionId: String, workingDirectory: String) async throws -> (sessionId: String, history: [ACPHistoryMessage]) {
        log("loadSession: starting, sessionId=\(sessionId), cwd=\(workingDirectory)")
        guard let client = client else {
            log("loadSession: not connected")
            throw ACPServiceError.notConnected
        }

        // Use resumeSession instead of loadSession - server only supports unstable_resumeSession
        let result = try await client.resumeSession(sessionId: sessionId, cwd: workingDirectory)
        self.sessionId = result.sessionId
        log("loadSession: resumed session \(result.sessionId), historyCount=\(result.history.count)")

        // Update modes from resume result
        if let modeInfo = result.modes {
            availableModes = modeInfo.availableModes
            currentMode = availableModes.first { $0.id == modeInfo.currentModeId }
        }

        eventContinuation?.yield(.sessionLoaded(sessionId: result.sessionId))
        return (result.sessionId, result.history)
    }

    // MARK: - Prompting

    func sendPrompt(_ text: String) async throws {
        log("sendPrompt: starting, text length=\(text.count)")

        guard let client = client else {
            log("sendPrompt: not connected")
            throw ACPServiceError.notConnected
        }

        log("sendPrompt: calling client.prompt")
        _ = try await client.prompt(text: text)
        log("sendPrompt: prompt sent successfully")
    }

    func cancel() async throws {
        log("cancel: starting")
        guard let client = client else {
            log("cancel: no client")
            return
        }
        try await client.cancel()
        log("cancel: completed")
    }

    /// Interrupt current operation and send a new prompt
    func interruptAndPrompt(_ text: String) async throws {
        log("interruptAndPrompt: starting, text length=\(text.count)")
        guard let client = client else {
            log("interruptAndPrompt: not connected")
            throw ACPServiceError.notConnected
        }

        _ = try await client.interruptAndPrompt(text: text)
        log("interruptAndPrompt: completed")
    }

    // MARK: - Mode Management

    func setMode(_ modeId: String) async throws {
        log("setMode: starting, modeId=\(modeId)")
        guard let client = client else {
            log("setMode: not connected")
            throw ACPServiceError.notConnected
        }

        try await client.setMode(modeId)
        log("setMode: completed")
    }

    func cycleMode() async throws -> ACPMode? {
        log("cycleMode: starting")
        guard let client = client else {
            log("cycleMode: not connected")
            throw ACPServiceError.notConnected
        }

        let mode = try await client.cycleMode()
        log("cycleMode: completed, newMode=\(String(describing: mode))")
        return mode
    }

    // MARK: - Session History

    func loadSessionHistory(cwd: String) async throws -> [ACPHistoryMessage] {
        log("loadSessionHistory: starting, cwd=\(cwd), sessionId=\(String(describing: self.sessionId))")
        guard let sessionId = sessionId else {
            log("loadSessionHistory: no active session")
            throw ACPServiceError.noActiveSession
        }
        return try await loadSessionHistory(cwd: cwd, sessionId: sessionId)
    }

    func loadSessionHistory(cwd: String, sessionId: String) async throws -> [ACPHistoryMessage] {
        log("loadSessionHistory: starting, cwd=\(cwd), sessionId=\(sessionId)")

        #if os(macOS)
        let history = try await SessionHistoryLoader.loadHistory(sessionId: sessionId, cwd: cwd)
        log("loadSessionHistory: loaded \(history.count) messages")
        return history
        #else
        // On iOS, load history from remote server via SSH
        guard let sshConnection = sshConnection else {
            log("loadSessionHistory: no SSH connection")
            throw ACPServiceError.notConnected
        }

        let history = try await loadRemoteSessionHistory(sessionId: sessionId, cwd: cwd, via: sshConnection)
        log("loadSessionHistory: loaded \(history.count) messages from remote")
        return history
        #endif
    }

    #if !os(macOS)
    /// Load session history from remote server via SSH
    private func loadRemoteSessionHistory(sessionId: String, cwd: String, via connection: ACPSSHConnection) async throws -> [ACPHistoryMessage] {
        // Path pattern: ~/.claude/projects/{encoded-cwd}/{sessionId}.jsonl
        let encodedCwd = cwd.replacingOccurrences(of: "/", with: "-")
        let remotePath = "~/.claude/projects/\(encodedCwd)/\(sessionId).jsonl"

        log("loadRemoteSessionHistory: reading \(remotePath)")

        // Check if file exists
        let exists = try await connection.remoteFileExists(remotePath)
        if !exists {
            log("loadRemoteSessionHistory: file not found")
            return []
        }

        // Read file content
        let content = try await connection.readRemoteFile(remotePath)
        if content.isEmpty {
            return []
        }

        // Parse JSONL
        return parseSessionHistory(content)
    }

    /// Parse session history from JSONL content
    private func parseSessionHistory(_ content: String) -> [ACPHistoryMessage] {
        let lines = content.components(separatedBy: "\n").filter { !$0.isEmpty }
        var messages: [ACPHistoryMessage] = []

        for line in lines {
            guard let data = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }

            guard let type = json["type"] as? String,
                  (type == "user" || type == "assistant") else {
                continue
            }

            let role: ACPMessageRole = type == "user" ? .user : .assistant
            let uuid = json["uuid"] as? String ?? UUID().uuidString

            // Parse timestamp
            var timestamp = Date()
            if let timestampStr = json["timestamp"] as? String {
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                timestamp = formatter.date(from: timestampStr) ?? Date()
            }

            // Extract content from message.content array
            guard let message = json["message"] as? [String: Any],
                  let contentBlocks = message["content"] as? [[String: Any]] else {
                continue
            }

            var textParts: [String] = []
            var toolCalls: [ACPHistoryToolCall] = []
            var hasToolResult = false

            for block in contentBlocks {
                guard let blockType = block["type"] as? String else { continue }

                switch blockType {
                case "text":
                    if let text = block["text"] as? String, !text.isEmpty {
                        textParts.append(text)
                    }
                case "tool_use":
                    // Parse tool use blocks properly
                    if let toolId = block["id"] as? String,
                       let toolName = block["name"] as? String {
                        var inputStr: String? = nil
                        if let input = block["input"] as? [String: Any],
                           let inputData = try? JSONSerialization.data(withJSONObject: input, options: []),
                           let str = String(data: inputData, encoding: .utf8) {
                            inputStr = str
                        }
                        toolCalls.append(ACPHistoryToolCall(id: toolId, name: toolName, input: inputStr, output: nil))
                    }
                case "tool_result":
                    hasToolResult = true
                    // Skip tool results - these are system responses, not user content
                default:
                    break
                }
            }

            // For user messages: only show if there's actual text (not just tool results)
            if role == .user {
                guard !textParts.isEmpty else { continue }
            }

            let textContent = textParts.joined(separator: "\n")

            // For assistant messages: include even if only tool calls (no text)
            if role == .assistant && textContent.isEmpty && toolCalls.isEmpty {
                continue
            }

            messages.append(ACPHistoryMessage(
                id: uuid,
                role: role,
                content: textContent,
                timestamp: timestamp,
                toolCalls: toolCalls.isEmpty ? nil : toolCalls
            ))
        }

        return messages
    }

    /// Check if a session exists on the remote server
    func sessionExistsOnServer(sessionId: String, cwd: String) async -> Bool {
        guard let connection = sshConnection else {
            log("sessionExistsOnServer: no SSH connection")
            return false
        }

        let encodedCwd = cwd.replacingOccurrences(of: "/", with: "-")
        let remotePath = "~/.claude/projects/\(encodedCwd)/\(sessionId).jsonl"
        log("sessionExistsOnServer: checking \(remotePath)")

        do {
            let exists = try await connection.remoteFileExists(remotePath)
            log("sessionExistsOnServer: exists=\(exists)")
            return exists
        } catch {
            log("sessionExistsOnServer: error - \(error)")
            return false
        }
    }

    /// List sessions from the remote server by reading .jsonl files
    /// Returns sessions sorted by modification time (newest first)
    func listRemoteSessions(cwd: String, projectId: UUID) async -> [Session] {
        guard let connection = sshConnection else {
            log("listRemoteSessions: no SSH connection")
            return []
        }

        let encodedCwd = cwd.replacingOccurrences(of: "/", with: "-")
        let remotePath = "~/.claude/projects/\(encodedCwd)"
        log("listRemoteSessions: listing \(remotePath)")

        do {
            // Use stat to get modification time in epoch seconds, along with size and filename
            // Format: epoch_seconds size filename
            // Sort by epoch descending (newest first), filter out empty/tiny files, limit to 10 most recent
            let command = """
            bash -c 'for f in \(remotePath)/*.jsonl; do [ -f "$f" ] && stat --format="%Y %s %n" "$f" 2>/dev/null; done | sort -rn | head -20'
            """
            let output = try await connection.executeCommand(command)

            var sessions: [Session] = []
            let lines = output.components(separatedBy: "\n").filter { !$0.isEmpty }

            for line in lines {
                // Parse: epoch_seconds size filename
                let parts = line.components(separatedBy: " ")
                guard parts.count >= 3 else { continue }

                guard let epoch = Double(parts[0]) else { continue }
                guard let size = Int(parts[1]), size > 500 else { continue } // Skip empty/tiny files

                let fullPath = parts.dropFirst(2).joined(separator: " ")
                guard fullPath.hasSuffix(".jsonl") else { continue }

                // Extract session ID from path
                let filename = (fullPath as NSString).lastPathComponent
                let sessionId = filename.replacingOccurrences(of: ".jsonl", with: "")

                let createdAt = Date(timeIntervalSince1970: epoch)
                let session = Session(id: sessionId, projectId: projectId, createdAt: createdAt)
                sessions.append(session)
            }

            log("listRemoteSessions: found \(sessions.count) sessions")
            return sessions
        } catch {
            log("listRemoteSessions: error - \(error)")
            return []
        }
    }
    #endif

    // MARK: - Event Handling

    private func startEventListener() {
        guard let client = client else {
            log("startEventListener: no client")
            return
        }

        log("startEventListener: starting event loop")
        eventTask = Task { [weak self] in
            for await event in await client.events {
                await self?.handleACPEvent(event)
            }
            log("startEventListener: event loop ended")
        }
    }

    private func handleACPEvent(_ event: ACPEvent) {
        switch event {
        case .connected(let info):
            log("EVENT: connected, agent=\(info.name)")
            agentInfo = info
            eventContinuation?.yield(.connected)

        case .disconnected:
            log("EVENT: disconnected")
            isConnected = false
            sessionId = nil
            eventContinuation?.yield(.disconnected)

        case .sessionCreated(let id, let modeInfo):
            log("EVENT: sessionCreated, id=\(id), modes=\(modeInfo?.availableModes.map { $0.id }.joined(separator: ",") ?? "none")")
            sessionId = id
            updateModes(from: modeInfo)
            eventContinuation?.yield(.sessionCreated(sessionId: id))

        case .sessionLoaded(let id, let modeInfo):
            log("EVENT: sessionLoaded, id=\(id)")
            sessionId = id
            updateModes(from: modeInfo)
            eventContinuation?.yield(.sessionLoaded(sessionId: id))

        case .sessionResumed(let id, let modeInfo):
            log("EVENT: sessionResumed, id=\(id)")
            sessionId = id
            updateModes(from: modeInfo)
            eventContinuation?.yield(.sessionLoaded(sessionId: id))

        case .textChunk(let text, let promptId):
            log("EVENT: textChunk, length=\(text.count), promptId=\(promptId)")
            eventContinuation?.yield(.textDelta(text))

        case .thinkingChunk(let text, let promptId):
            log("EVENT: thinkingChunk, length=\(text.count), promptId=\(promptId)")
            eventContinuation?.yield(.thinking(text))

        case .toolCallStarted(let id, let name, let input, let promptId):
            log("EVENT: toolCallStarted, id=\(id), name=\(name), inputLength=\(input?.count ?? 0), promptId=\(promptId)")
            eventContinuation?.yield(.toolUseStarted(id: id, name: name, input: input ?? ""))

        case .toolCallUpdate(let id, let status, let title, let input, let output, let error, let promptId):
            let isError = error != nil
            let result = error ?? output
            log("EVENT: toolCallUpdate, id=\(id), status=\(status), title=\(title ?? "nil"), hasInput=\(input != nil), hasOutput=\(output != nil), hasError=\(error != nil), promptId=\(promptId)")
            if let output = output {
                log("EVENT: toolCallUpdate output preview: \(String(output.prefix(200)))")
            }
            if let error = error {
                log("EVENT: toolCallUpdate error: \(error)")
            }
            if status == "complete" || status == "error" {
                log("EVENT: toolCallUpdate -> COMPLETED, id=\(id), isError=\(isError)")
                eventContinuation?.yield(.toolUseCompleted(id: id, result: result, isError: isError))
            } else {
                log("EVENT: toolCallUpdate -> PROGRESS, id=\(id), status=\(status), title=\(title ?? "nil")")
                eventContinuation?.yield(.toolUseProgress(id: id, status: status, title: title, input: input))
            }

        case .planStep(let id, let title, let status, let promptId):
            log("EVENT: planStep, id=\(id), title=\(title), status=\(status), promptId=\(promptId)")
            eventContinuation?.yield(.planStep(id: id, title: title, status: status))

        case .modeChanged(let modeId):
            log("EVENT: modeChanged, modeId=\(modeId)")
            // Update current mode from available modes
            if let mode = availableModes.first(where: { $0.id == modeId }) {
                currentMode = mode
            }
            eventContinuation?.yield(.modeChanged(mode: modeId))

        case .promptComplete(let reason, let promptId):
            log("EVENT: promptComplete, reason=\(reason), promptId=\(promptId)")
            eventContinuation?.yield(.turnComplete(stopReason: reason))

        case .promptInterrupted(let text, let promptId):
            log("EVENT: promptInterrupted, textLength=\(text.count), promptId=\(promptId)")
            eventContinuation?.yield(.textDelta(text))
            eventContinuation?.yield(.turnComplete(stopReason: "cancelled"))

        case .error(let err):
            log("EVENT: error, message=\(err.localizedDescription)")
            eventContinuation?.yield(.error(err.localizedDescription))
        }
    }

    private func updateModes(from modeInfo: ACPModeInfo?) {
        guard let modeInfo = modeInfo else { return }
        availableModes = modeInfo.availableModes
        currentMode = availableModes.first { $0.id == modeInfo.currentModeId }
    }
}

// MARK: - ACP Service Events

enum ACPServiceEvent: Sendable {
    case connected
    case disconnected
    case sessionCreated(sessionId: String)
    case sessionLoaded(sessionId: String)
    case textDelta(String)
    case thinking(String)
    case toolUseStarted(id: String, name: String, input: String)
    case toolUseProgress(id: String, status: String, title: String?, input: String?)
    case toolUseCompleted(id: String, result: String?, isError: Bool)
    case planStep(id: String, title: String, status: String)
    case modeChanged(mode: String)
    case turnComplete(stopReason: String)
    case error(String)
    case permissionRequest(id: String, toolName: String, input: String?, title: String?)
}

// MARK: - Pending Permission Request

struct PendingPermissionRequest: Identifiable {
    let id: String
    let toolName: String
    let input: String?
    let title: String?
    let continuation: CheckedContinuation<Bool, Never>
}

// MARK: - ACP Service Errors

enum ACPServiceError: Error, LocalizedError {
    case notConnected
    case noActiveSession
    case connectionFailed(String)
    case claudeCodeACPNotInstalled

    var errorDescription: String? {
        switch self {
        case .notConnected:
            return "Not connected to ACP agent"
        case .noActiveSession:
            return "No active session"
        case .connectionFailed(let message):
            return "Connection failed: \(message)"
        case .claudeCodeACPNotInstalled:
            return "claude-code-acp is not installed. Install it with: npm install -g @zed-industries/claude-code-acp"
        }
    }

    /// Install command for claude-code-acp
    static let installCommand = "npm install -g @zed-industries/claude-code-acp"
}

// MARK: - ACP Permission Delegate

/// Actor that handles permission requests for ACP
/// Emits permission requests to the service for UI handling
actor ACPPermissionDelegate: ACPClientDelegate {
    // Callback to handle permission requests - returns (granted, context)
    var permissionHandler: (@Sendable (String, String?, String?) async -> (Bool, String?))?

    func setPermissionHandler(_ handler: @escaping @Sendable (String, String?, String?) async -> (Bool, String?)) {
        self.permissionHandler = handler
    }

    func acpClient(_ client: ACPClient, didReceive event: ACPEvent) async {
        // Events are handled by ACPService's event listener
    }

    func acpClient(_ client: ACPClient, requestPermissionFor toolName: String, input: String?, prompt: String?) async -> (granted: Bool, context: String?) {
        appLog("Permission request: tool=\(toolName)", category: "Permission")
        if let input = input {
            appLog("Input: \(String(input.prefix(200)))", category: "Permission")
        }
        if let prompt = prompt {
            appLog("Prompt: \(prompt)", category: "Permission")
        }

        // Use handler to show UI
        if let handler = permissionHandler {
            let result = await handler(toolName, input, prompt)
            appLog("Handler result: granted=\(result.0)", category: "Permission")
            return result
        }

        // No handler - deny by default for safety
        appLog("No handler, denying permission", category: "Permission")
        return (false, "No permission handler configured")
    }
}
