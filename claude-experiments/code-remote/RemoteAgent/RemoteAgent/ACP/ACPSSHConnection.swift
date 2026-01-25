import Foundation
import Citadel
import NIO
import NIOSSH
import Crypto
import _CryptoExtras
import ACPLib

private func log(_ message: String) {
    appLog(message, category: "ACPSSHConnection")
}

// MARK: - ACP SSH Connection

/// SSH-based ACP connection that implements ACPConnectionProtocol
/// This allows using the real ACPClient on iOS via SSH transport
actor ACPSSHConnection: ACPConnectionProtocol, RemoteCommandExecutor {
    private var client: SSHClient?
    private var outputTask: Task<Void, Never>?
    private var heartbeatTask: Task<Void, Never>?
    private var notificationHandler: (@Sendable (JSONRPCNotification) async -> Void)?
    private var requestHandler: (@Sendable (JSONRPCRequest) async -> JSONRPCResponse?)?

    private var nextRequestId: Int = 0
    private var pendingRequests: [String: CheckedContinuation<JSONRPCResponse, Error>] = [:]
    private var _isClosed = false

    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    private let sessionId: String
    private var inputFifo: String { "/tmp/acp_in_\(sessionId)" }
    private var outputFile: String { "/tmp/acp_out_\(sessionId)" }  // Regular file, not FIFO - prevents blocking
    private var pidFile: String { "/tmp/acp_pid_\(sessionId)" }
    private var logFile: String { "/tmp/acp_log_\(sessionId)" }
    private var cwdFile: String { "/tmp/acp_cwd_\(sessionId)" }  // Stores working directory for reconnection

    var isClosed: Bool { _isClosed }

    /// Returns the session ID for potential reconnection
    var currentSessionId: String { sessionId }

    /// Create a new connection with a fresh session ID
    init() {
        self.sessionId = UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "").prefix(16).description
        log("init: created with sessionId=\(self.sessionId)")
    }

    /// Create a connection that will reconnect to an existing session
    init(existingSessionId: String) {
        self.sessionId = existingSessionId
        log("init: reconnecting to existing sessionId=\(self.sessionId)")
    }

    /// Check if a session with the given ID is still running on the remote server.
    /// This is a static helper to determine whether to reconnect or start fresh.
    static func isSessionRunning(
        sessionId: String,
        host: String,
        port: Int = 22,
        username: String,
        privateKeyPath: String?,
        password: String?
    ) async throws -> Bool {
        // Build authentication method
        let authMethod: SSHAuthenticationMethod
        if let keyPath = privateKeyPath, !keyPath.isEmpty {
            let expandedPath = (keyPath as NSString).expandingTildeInPath
            let keyURL = URL(fileURLWithPath: expandedPath)
            let keyData = try Data(contentsOf: keyURL)
            guard let keyString = String(data: keyData, encoding: .utf8) else {
                throw ACPConnectionError.encodingError("Could not read private key")
            }

            let keyType = try SSHKeyDetection.detectPrivateKeyType(from: keyString)
            if keyType == .rsa {
                let rsaKey = try Insecure.RSA.PrivateKey(sshRsa: keyString)
                authMethod = .rsa(username: username, privateKey: rsaKey)
            } else if keyType == .ed25519 {
                let ed25519Key = try Curve25519.Signing.PrivateKey(sshEd25519: keyString)
                authMethod = .ed25519(username: username, privateKey: ed25519Key)
            } else {
                throw ACPConnectionError.encodingError("Unsupported key type: \(keyType)")
            }
        } else if let password = password, !password.isEmpty {
            authMethod = .passwordBased(username: username, password: password)
        } else {
            throw ACPConnectionError.encodingError("Either SSH key or password must be provided")
        }

        let client = try await SSHClient.connect(
            host: host,
            port: port,
            authenticationMethod: authMethod,
            hostKeyValidator: .acceptAnything(),
            reconnect: .never
        )

        defer { Task { try? await client.close() } }

        let pidFile = "/tmp/acp_pid_\(sessionId)"
        let result = try await client.executeCommand("test -f \(pidFile) && kill -0 $(cat \(pidFile)) 2>/dev/null && echo 'running' || echo 'not_running'")
        let output = String(buffer: result).trimmingCharacters(in: .whitespacesAndNewlines)
        return output == "running"
    }

    /// Info about a running ACP session
    struct RunningSession: Identifiable {
        let id: String  // session ID
        let workingDirectory: String?
    }

    /// List all running ACP sessions on the remote server
    static func listRunningSessions(
        host: String,
        port: Int = 22,
        username: String,
        privateKeyPath: String?,
        password: String?
    ) async throws -> [RunningSession] {
        // Build authentication method
        let authMethod: SSHAuthenticationMethod
        if let keyPath = privateKeyPath, !keyPath.isEmpty {
            let expandedPath = (keyPath as NSString).expandingTildeInPath
            let keyURL = URL(fileURLWithPath: expandedPath)
            let keyData = try Data(contentsOf: keyURL)
            guard let keyString = String(data: keyData, encoding: .utf8) else {
                throw ACPConnectionError.encodingError("Could not read private key")
            }

            let keyType = try SSHKeyDetection.detectPrivateKeyType(from: keyString)
            if keyType == .rsa {
                let rsaKey = try Insecure.RSA.PrivateKey(sshRsa: keyString)
                authMethod = .rsa(username: username, privateKey: rsaKey)
            } else if keyType == .ed25519 {
                let ed25519Key = try Curve25519.Signing.PrivateKey(sshEd25519: keyString)
                authMethod = .ed25519(username: username, privateKey: ed25519Key)
            } else {
                throw ACPConnectionError.encodingError("Unsupported key type: \(keyType)")
            }
        } else if let password = password, !password.isEmpty {
            authMethod = .passwordBased(username: username, password: password)
        } else {
            throw ACPConnectionError.encodingError("Either SSH key or password must be provided")
        }

        let client = try await SSHClient.connect(
            host: host,
            port: port,
            authenticationMethod: authMethod,
            hostKeyValidator: .acceptAnything(),
            reconnect: .never
        )

        defer { Task { try? await client.close() } }

        // Find all PID files and check which ones have running processes
        // Output format: sessionId|workingDirectory (one per line)
        let result = try await client.executeCommand("""
            for f in /tmp/acp_pid_*; do
                if [ -f "$f" ]; then
                    sid=$(echo "$f" | sed 's/.*acp_pid_//')
                    if kill -0 $(cat "$f") 2>/dev/null; then
                        cwd=""
                        cwdfile="/tmp/acp_cwd_$sid"
                        if [ -f "$cwdfile" ]; then
                            cwd=$(cat "$cwdfile")
                        fi
                        echo "$sid|$cwd"
                    fi
                fi
            done
            """)
        let output = String(buffer: result)
        return output.components(separatedBy: "\n")
            .filter { !$0.isEmpty }
            .map { line in
                let parts = line.components(separatedBy: "|")
                let sessionId = parts[0]
                let cwd = parts.count > 1 && !parts[1].isEmpty ? parts[1] : nil
                return RunningSession(id: sessionId, workingDirectory: cwd)
            }
    }

    func setNotificationHandler(_ handler: @escaping @Sendable (JSONRPCNotification) async -> Void) {
        log("setNotificationHandler: handler set")
        self.notificationHandler = handler
    }

    func setRequestHandler(_ handler: @escaping @Sendable (JSONRPCRequest) async -> JSONRPCResponse?) {
        log("setRequestHandler: handler set")
        self.requestHandler = handler
    }

    // MARK: - Connection

    func connect(
        host: String,
        port: Int = 22,
        username: String,
        privateKeyPath: String?,
        password: String?,
        workingDirectory: String,
        acpPath: String = "claude-code-acp"
    ) async throws {
        log("connect: starting connection to \(username)@\(host):\(port)")
        log("connect: workingDirectory=\(workingDirectory), acpPath=\(acpPath)")
        log("connect: privateKeyPath=\(privateKeyPath ?? "nil"), hasPassword=\(password != nil)")

        // Build authentication method
        let authMethod: SSHAuthenticationMethod
        if let keyPath = privateKeyPath, !keyPath.isEmpty {
            log("connect: using private key authentication from \(keyPath)")
            let expandedPath = (keyPath as NSString).expandingTildeInPath
            let keyURL = URL(fileURLWithPath: expandedPath)
            let keyData = try Data(contentsOf: keyURL)
            guard let keyString = String(data: keyData, encoding: .utf8) else {
                throw ACPConnectionError.encodingError("Could not read private key")
            }

            let keyType = try SSHKeyDetection.detectPrivateKeyType(from: keyString)
            log("connect: detected key type: \(keyType)")
            if keyType == .rsa {
                let rsaKey = try Insecure.RSA.PrivateKey(sshRsa: keyString)
                authMethod = .rsa(username: username, privateKey: rsaKey)
                log("connect: using RSA key authentication")
            } else if keyType == .ed25519 {
                let ed25519Key = try Curve25519.Signing.PrivateKey(sshEd25519: keyString)
                authMethod = .ed25519(username: username, privateKey: ed25519Key)
                log("connect: using Ed25519 key authentication")
            } else {
                throw ACPConnectionError.encodingError("Unsupported key type: \(keyType)")
            }
        } else if let password = password, !password.isEmpty {
            log("connect: using password authentication")
            authMethod = .passwordBased(username: username, password: password)
        } else {
            log("connect: ERROR - no authentication method provided")
            throw ACPConnectionError.encodingError("Either SSH key or password must be provided for authentication")
        }

        // Connect via SSH
        client = try await SSHClient.connect(
            host: host,
            port: port,
            authenticationMethod: authMethod,
            hostKeyValidator: .acceptAnything(),
            reconnect: .never
        )
        log("connect: SSH connected")

        // Check if session already exists (reconnection case)
        let existsResult = try await client?.executeCommand("test -f \(pidFile) && kill -0 $(cat \(pidFile)) 2>/dev/null && echo 'running' || echo 'not_running'")
        let existsStr = String(buffer: existsResult ?? ByteBuffer()).trimmingCharacters(in: .whitespacesAndNewlines)
        let sessionExists = existsStr == "running"

        if sessionExists {
            log("connect: found existing ACP session, reconnecting...")
            // Start reading from the existing output file (tail -f)
            startOutputReader()
            log("connect: reconnected to existing session")
        } else {
            log("connect: starting new ACP session")

            // Clean up any stale files and create input FIFO
            let setupCmd = "rm -f \(inputFifo) \(outputFile) \(pidFile) \(logFile) \(cwdFile); mkfifo \(inputFifo) && echo '\(workingDirectory)' > \(cwdFile) && echo 'ok'"
            let mkfifoResult = try await client?.executeCommand(setupCmd)
            let mkfifoStr = String(buffer: mkfifoResult ?? ByteBuffer()).trimmingCharacters(in: .whitespacesAndNewlines)
            if mkfifoStr != "ok" {
                throw ACPConnectionError.connectionClosed
            }
            log("connect: created input FIFO at \(inputFifo), saved cwd to \(cwdFile)")

            // Start claude-code-acp using nohup so it survives SSH disconnection
            // Output goes to a regular file so writes never block
            let startCmd = "nohup bash -c 'cd \(workingDirectory) && tail -f \(inputFifo) | \(acpPath)' >> \(outputFile) 2>> \(logFile) & echo $!"
            log("connect: starting detached ACP process")
            let pidResult = try await client?.executeCommand(startCmd)
            let pidStr = String(buffer: pidResult ?? ByteBuffer()).trimmingCharacters(in: .whitespacesAndNewlines)

            // Save PID for later process management
            _ = try await client?.executeCommand("echo '\(pidStr)' > \(pidFile)")
            log("connect: ACP process started with PID \(pidStr)")

            // Wait for the process to initialize
            try await Task.sleep(nanoseconds: 1_500_000_000)

            // Verify the process is running
            let checkCmd = "kill -0 \(pidStr) 2>/dev/null && echo 'ok' || echo 'failed'"
            let checkResult = try await client?.executeCommand(checkCmd)
            let checkStr = String(buffer: checkResult ?? ByteBuffer()).trimmingCharacters(in: .whitespacesAndNewlines)
            if checkStr != "ok" {
                let logsResult = try await client?.executeCommand("cat \(logFile) 2>/dev/null | tail -20")
                let logs = String(buffer: logsResult ?? ByteBuffer())
                log("connect: ACP process died. Logs: \(logs)")
                throw ACPConnectionError.encodingError("ACP process failed to start: \(logs)")
            }

            log("connect: ACP process running")

            // Start reading from output file
            startOutputReader()
        }

        log("connect: ready to send requests")
    }

    private func startOutputReader() {
        // Start heartbeat task to monitor if output reader is still alive
        heartbeatTask = Task { [weak self] in
            var heartbeatCount = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 10_000_000_000) // 10 seconds
                heartbeatCount += 1
                let isClosed = await self?.isClosed ?? true
                log("SSH heartbeat #\(heartbeatCount) (isClosed=\(isClosed))")
            }
            log("SSH heartbeat task ended")
        }

        outputTask = Task { [weak self] in
            guard let self = self, let client = await self.client else {
                log("startOutputReader: no client")
                return
            }

            let outputFile = await self.outputFile
            log("startOutputReader: tailing output file \(outputFile)")

            do {
                // Use tail -f to read from the output file - this follows new data as it's appended
                // Using -n +1 to read from the beginning on fresh sessions, or just new data on reconnect
                let stream = try await client.executeCommandStream("tail -f -n +1 \(outputFile)")
                log("startOutputReader: stream obtained, waiting for output")

                var buffer = Data()

                for try await chunk in stream {
                    switch chunk {
                    case .stdout(var stdoutBuffer):
                        if let bytes = stdoutBuffer.readBytes(length: stdoutBuffer.readableBytes) {
                            buffer.append(contentsOf: bytes)
                        }
                    case .stderr(var stderrBuffer):
                        if let bytes = stderrBuffer.readBytes(length: stderrBuffer.readableBytes) {
                            let stderr = String(data: Data(bytes), encoding: .utf8) ?? ""
                            log("stderr: \(stderr)")
                        }
                    }

                    // Process complete lines (NDJSON)
                    while let newlineIndex = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                        let lineData = buffer.prefix(upTo: newlineIndex)
                        buffer = Data(buffer.suffix(from: buffer.index(after: newlineIndex)))

                        guard !lineData.isEmpty else { continue }

                        if let lineStr = String(data: Data(lineData), encoding: .utf8) {
                            log("received: \(lineStr.prefix(200))")
                        }

                        await self.handleReceivedData(Data(lineData))
                    }
                }

                if !buffer.isEmpty {
                    await self.handleReceivedData(buffer)
                }

                log("output stream ended")
            } catch {
                log("output reader error: \(error)")
            }

            await self.markClosed()
        }
    }

    private func handleReceivedData(_ data: Data) async {
        do {
            let message = try decoder.decode(JSONRPCMessage.self, from: data)

            switch message {
            case .response(let response):
                log("handleReceivedData: response for id=\(response.id)")
                if let cont = pendingRequests.removeValue(forKey: response.id.description) {
                    cont.resume(returning: response)
                } else {
                    log("handleReceivedData: WARNING - no pending request for id=\(response.id)")
                }

            case .notification(let notification):
                log("handleReceivedData: notification method=\(notification.method)")
                if let handler = notificationHandler {
                    await handler(notification)
                } else {
                    log("handleReceivedData: WARNING - no notification handler for \(notification.method)")
                }

            case .request(let request):
                log("handleReceivedData: request from agent method=\(request.method), id=\(request.id)")
                if let handler = requestHandler {
                    if let response = await handler(request) {
                        log("handleReceivedData: sending response for request id=\(request.id)")
                        // Send response back to agent
                        await sendResponseToAgent(response)
                    } else {
                        log("handleReceivedData: no response from handler for request id=\(request.id)")
                    }
                } else {
                    log("handleReceivedData: WARNING - no request handler for \(request.method)")
                }
            }
        } catch {
            log("handleReceivedData: decode error: \(error)")
            if let dataStr = String(data: data, encoding: .utf8) {
                log("handleReceivedData: raw data: \(dataStr.prefix(500))")
            }
        }
    }

    /// Maximum size for inline base64 data in shell commands (64KB to be safe with shell limits)
    private static let maxInlineDataSize = 64 * 1024

    /// Helper to send data to the FIFO, handling large payloads by chunking
    private func sendDataToFifo(_ data: Data, label: String) async throws {
        guard let client = client else {
            throw ACPConnectionError.notConnected
        }

        log("\(label): data size=\(data.count) bytes")

        if data.count <= Self.maxInlineDataSize {
            // Small data: send inline via echo
            let base64Data = data.base64EncodedString()
            let writeCmd = "echo '\(base64Data)' | base64 -d > \(inputFifo)"
            _ = try await client.executeCommand(writeCmd)
        } else {
            // Large data: write to temp file first, then cat to FIFO
            // This avoids shell argument limits (ARG_MAX)
            let tempFile = "/tmp/acp_chunk_\(sessionId)_\(Int.random(in: 0..<Int.max))"
            log("\(label): using temp file \(tempFile) for large payload")

            // Write data in chunks to the temp file
            let chunkSize = Self.maxInlineDataSize
            var offset = 0
            var isFirst = true

            while offset < data.count {
                let end = min(offset + chunkSize, data.count)
                let chunk = data[offset..<end]
                let base64Chunk = chunk.base64EncodedString()

                let redirectOp = isFirst ? ">" : ">>"
                let writeCmd = "echo '\(base64Chunk)' | base64 -d \(redirectOp) \(tempFile)"
                _ = try await client.executeCommand(writeCmd)

                isFirst = false
                offset = end
            }

            // Now cat the temp file to the FIFO and clean up
            let catCmd = "cat \(tempFile) > \(inputFifo) && rm -f \(tempFile)"
            _ = try await client.executeCommand(catCmd)
        }
    }

    private func sendResponseToAgent(_ response: JSONRPCResponse) async {
        do {
            var data = try encoder.encode(response)
            data.append(UInt8(ascii: "\n"))
            try await sendDataToFifo(data, label: "sendResponseToAgent[\(response.id)]")
            log("sendResponseToAgent: sent response for id=\(response.id)")
        } catch {
            log("sendResponseToAgent: error sending response: \(error)")
        }
    }

    private func markClosed() {
        _isClosed = true
        for (_, cont) in pendingRequests {
            cont.resume(throwing: ACPConnectionError.connectionClosed)
        }
        pendingRequests.removeAll()
    }

    // MARK: - ACPConnectionProtocol

    func sendRequest<P: Encodable, R: Decodable>(method: String, params: P?) async throws -> R {
        log("sendRequest: method=\(method), isClosed=\(_isClosed), hasClient=\(client != nil)")
        guard !_isClosed else {
            log("sendRequest: ERROR - connection is closed")
            throw ACPConnectionError.notConnected
        }
        guard client != nil else {
            log("sendRequest: ERROR - no SSH client")
            throw ACPConnectionError.notConnected
        }

        let id = "\(nextRequestId)"
        nextRequestId += 1
        log("sendRequest: id=\(id), method=\(method)")

        let request = JSONRPCRequest(id: id, method: method, params: params)

        var data = try encoder.encode(request)
        data.append(UInt8(ascii: "\n"))

        if let jsonStr = String(data: data, encoding: .utf8) {
            log("sendRequest: sending JSON (first 300 chars): \(jsonStr.prefix(300))")
        }

        let response = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<JSONRPCResponse, Error>) in
            pendingRequests[id] = cont
            log("sendRequest: registered pending request id=\(id), total pending=\(pendingRequests.count)")

            Task {
                do {
                    try await self.sendDataToFifo(data, label: "sendRequest[\(id)]")
                    log("sendRequest: write completed for id=\(id)")
                } catch {
                    log("sendRequest: ERROR writing request id=\(id): \(error)")
                    if let removed = self.pendingRequests.removeValue(forKey: id) {
                        removed.resume(throwing: error)
                    }
                }
            }
        }

        log("sendRequest: received response for id=\(id)")
        switch response {
        case .success(let successResponse):
            log("sendRequest: success response for id=\(id)")
            return try successResponse.result.decode(R.self)
        case .error(let errorResponse):
            log("sendRequest: JSON-RPC error for id=\(id): code=\(errorResponse.error.code), message=\(errorResponse.error.message)")
            throw errorResponse.error
        }
    }

    func sendNotification<P: Encodable>(method: String, params: P?) async throws {
        guard !_isClosed else {
            throw ACPConnectionError.notConnected
        }
        guard client != nil else {
            throw ACPConnectionError.notConnected
        }

        let notification = JSONRPCNotification(method: method, params: params)

        var data = try encoder.encode(notification)
        data.append(UInt8(ascii: "\n"))

        log("sendNotification: method=\(method)")
        try await sendDataToFifo(data, label: "sendNotification[\(method)]")
    }

    // MARK: - Remote Command Execution (for terminals and file reading)

    /// Execute a command on the remote server and return the output
    func executeCommand(_ command: String) async throws -> String {
        guard let client = client else {
            throw ACPConnectionError.notConnected
        }

        log("executeCommand: \(command.prefix(100))")
        let result = try await client.executeCommand(command)
        let output = String(buffer: result)
        log("executeCommand: output length=\(output.count)")
        return output
    }

    /// Read a file from the remote server
    func readRemoteFile(_ path: String) async throws -> String {
        guard let client = client else {
            throw ACPConnectionError.notConnected
        }

        log("readRemoteFile: \(path)")
        // Use bash -c to ensure tilde expansion works
        let result = try await client.executeCommand("bash -c 'cat \(path) 2>/dev/null'")
        let content = String(buffer: result)
        log("readRemoteFile: read \(content.count) bytes")
        return content
    }

    /// Check if a file exists on the remote server
    func remoteFileExists(_ path: String) async throws -> Bool {
        guard let client = client else {
            throw ACPConnectionError.notConnected
        }

        // Use bash -c to ensure tilde expansion works (single quotes prevent expansion)
        let result = try await client.executeCommand("bash -c 'test -f \(path) && echo exists || echo notfound'")
        let output = String(buffer: result).trimmingCharacters(in: .whitespacesAndNewlines)
        return output == "exists"
    }

    /// List files in a remote directory
    func listRemoteDirectory(_ path: String, pattern: String? = nil) async throws -> [String] {
        guard let client = client else {
            throw ACPConnectionError.notConnected
        }

        let escapedPath = path.replacingOccurrences(of: "'", with: "'\"'\"'")
        let command: String
        if let pattern = pattern {
            command = "ls -1 '\(escapedPath)'/\(pattern) 2>/dev/null || true"
        } else {
            command = "ls -1 '\(escapedPath)' 2>/dev/null || true"
        }

        let result = try await client.executeCommand(command)
        let output = String(buffer: result)
        return output.components(separatedBy: "\n").filter { !$0.isEmpty }
    }

    /// Close the connection and terminate the ACP session (protocol conformance)
    func close() async {
        await closeConnection(keepSessionAlive: false)
    }

    /// Close the connection with option to keep session alive.
    /// - Parameter keepSessionAlive: If true, the ACP process keeps running on the server for later reconnection.
    ///                               If false (default), the ACP process is terminated.
    private func closeConnection(keepSessionAlive: Bool) async {
        guard !_isClosed else { return }
        _isClosed = true

        heartbeatTask?.cancel()
        heartbeatTask = nil
        outputTask?.cancel()
        outputTask = nil

        if let client = client {
            if keepSessionAlive {
                // Just disconnect SSH, leave ACP running for reconnection
                log("close: keeping session alive for reconnection (sessionId=\(sessionId))")
            } else {
                // Terminate ACP process and clean up
                log("close: terminating ACP session")
                _ = try? await client.executeCommand("""
                    if [ -f \(pidFile) ]; then
                        kill $(cat \(pidFile)) 2>/dev/null
                    fi
                    pkill -f 'tail.*\(inputFifo)' 2>/dev/null
                    rm -f \(inputFifo) \(outputFile) \(pidFile) \(logFile)
                    """)
            }
            try? await client.close()
        }
        client = nil

        for (_, cont) in pendingRequests {
            cont.resume(throwing: ACPConnectionError.connectionClosed)
        }
        pendingRequests.removeAll()

        log("connection closed (keepSessionAlive=\(keepSessionAlive))")
    }

    /// Disconnect from SSH but keep the ACP session running for later reconnection
    func disconnect() async {
        await closeConnection(keepSessionAlive: true)
    }

    /// Terminate the ACP session completely
    func terminate() async {
        await closeConnection(keepSessionAlive: false)
    }

    deinit {
        heartbeatTask?.cancel()
        outputTask?.cancel()
    }
}
