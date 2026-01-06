import Foundation
import ACPLib

// Simple stderr logging for terminal visibility
private func log(_ message: String) {
    let timestamp = ISO8601DateFormatter().string(from: Date())
    fputs("[\(timestamp)] [ACPService] \(message)\n", stderr)
}

// MARK: - ACP Service

/// Service for interacting with Claude Code via ACP protocol
/// This replaces the SSH-based approach with the standardized ACP protocol
@MainActor
class ACPService: ObservableObject {
    private var client: ACPClient?
    private var eventTask: Task<Void, Never>?
    private var permissionDelegate: ACPPermissionDelegate?

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
    }

    /// Check if claude-code-acp is installed on a remote server via SSH
    static func isClaudeCodeACPInstalledRemote(
        host: String,
        username: String,
        keyPath: String?
    ) async -> Bool {
        await Task.detached {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")

            var args = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
            if let keyPath = keyPath, !keyPath.isEmpty {
                args.append(contentsOf: ["-i", (keyPath as NSString).expandingTildeInPath])
            }
            args.append("\(username)@\(host)")
            args.append("which claude-code-acp || command -v claude-code-acp")

            process.arguments = args

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe

            do {
                try process.run()
                process.waitUntilExit()
                return process.terminationStatus == 0
            } catch {
                return false
            }
        }.value
    }

    /// Install claude-code-acp locally
    static func installClaudeCodeACPLocal(
        onOutput: @escaping @Sendable (String) -> Void
    ) async -> Result<Void, Error> {
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
    }

    /// Install claude-code-acp on a remote server via SSH
    static func installClaudeCodeACPRemote(
        host: String,
        username: String,
        keyPath: String?,
        onOutput: @escaping @Sendable (String) -> Void
    ) async -> Result<Void, Error> {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")

        var args = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-t", "-t"]
        if let keyPath = keyPath, !keyPath.isEmpty {
            args.append(contentsOf: ["-i", (keyPath as NSString).expandingTildeInPath])
        }
        args.append("\(username)@\(host)")
        // Use npm to install globally, trying common node paths
        args.append("export PATH=$PATH:$HOME/.nvm/versions/node/*/bin:$HOME/.npm-global/bin:/usr/local/bin && npm install -g @zed-industries/claude-code-acp")

        process.arguments = args

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
    }

    /// Connect to a local Claude Code ACP agent
    /// Requires: npm install -g @zed-industries/claude-code-acp
    func connectLocal(
        acpPath: String? = nil,
        workingDirectory: String
    ) async throws {
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
    }

    /// Connect to a remote Claude Code ACP agent via SSH tunnel
    /// This creates an SSH tunnel and then connects via ACP over the tunnel
    /// Requires claude-code-acp to be installed on the remote machine
    func connectRemote(
        sshHost: String,
        sshUsername: String,
        sshKeyPath: String?,
        acpPath: String = "claude-code-acp",
        workingDirectory: String
    ) async throws {
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

        // Fallback - assume it's in PATH
        return "claude-code-acp"
    }

    func disconnect() async {
        log("disconnect: starting")
        eventTask?.cancel()
        eventTask = nil

        if let client = client {
            // Clear SSH configuration
            await client.setSSHConfiguration(nil)
            await client.disconnect()
        }
        client = nil
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

    func loadSession(sessionId: String, workingDirectory: String? = nil) async throws -> String {
        log("loadSession: starting, sessionId=\(sessionId)")
        guard let client = client else {
            log("loadSession: not connected")
            throw ACPServiceError.notConnected
        }

        let id = try await client.loadSession(sessionId: sessionId, cwd: workingDirectory)
        self.sessionId = id
        log("loadSession: loaded session \(id)")
        eventContinuation?.yield(.sessionLoaded(sessionId: id))
        return id
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

        let history = try await SessionHistoryLoader.loadHistory(sessionId: sessionId, cwd: cwd)
        log("loadSessionHistory: loaded \(history.count) messages")
        return history
    }

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
        let timestamp = ISO8601DateFormatter().string(from: Date())
        fputs("[\(timestamp)] [ACPPermissionDelegate] Permission request: tool=\(toolName)\n", stderr)
        if let input = input {
            fputs("[\(timestamp)] [ACPPermissionDelegate] Input: \(String(input.prefix(200)))\n", stderr)
        }
        if let prompt = prompt {
            fputs("[\(timestamp)] [ACPPermissionDelegate] Prompt: \(prompt)\n", stderr)
        }

        // Use handler to show UI
        if let handler = permissionHandler {
            let result = await handler(toolName, input, prompt)
            fputs("[\(timestamp)] [ACPPermissionDelegate] Handler result: granted=\(result.0)\n", stderr)
            return result
        }

        // No handler - deny by default for safety
        fputs("[\(timestamp)] [ACPPermissionDelegate] No handler, denying permission\n", stderr)
        return (false, "No permission handler configured")
    }
}
