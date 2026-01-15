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
    private var pidFile: String { "/tmp/acp_pid_\(sessionId)" }

    var isClosed: Bool { _isClosed }

    init() {
        self.sessionId = UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "").prefix(16).description
        log("init: created with sessionId=\(self.sessionId)")
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

        // Create FIFO for input
        let mkfifoResult = try await client?.executeCommand("rm -f \(inputFifo) \(pidFile); mkfifo \(inputFifo) && echo 'ok'")
        let mkfifoStr = String(buffer: mkfifoResult ?? ByteBuffer()).trimmingCharacters(in: .whitespacesAndNewlines)
        if mkfifoStr != "ok" {
            throw ACPConnectionError.connectionClosed
        }
        log("connect: created input FIFO at \(inputFifo)")

        // Start claude-code-acp reading from FIFO
        let startCmd = "cd \(workingDirectory) && echo \\$\\$ > \(pidFile) && exec tail -f \(inputFifo) | \(acpPath)"
        log("connect: starting ACP process with command: \(startCmd)")

        // Start reading output in background
        startOutputReader(command: startCmd)

        // Wait for the process to start
        // We can't easily poll the PID file while executeCommandStream is running on the same SSH client
        // So we just wait a bit and let the initialize request verify the connection works
        log("connect: waiting 2 seconds for ACP process to initialize...")
        try await Task.sleep(nanoseconds: 2_000_000_000)

        log("connect: ready to send requests (process assumed started, initialize will verify)")
    }

    private func startOutputReader(command: String) {
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

            log("startOutputReader: starting command stream")

            do {
                let stream = try await client.executeCommandStream(command)
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

    private func sendResponseToAgent(_ response: JSONRPCResponse) async {
        guard let client = client else {
            log("sendResponseToAgent: no client")
            return
        }

        do {
            var data = try encoder.encode(response)
            data.append(UInt8(ascii: "\n"))

            let base64Json = data.base64EncodedString()
            let writeCmd = "echo '\(base64Json)' | base64 -d > \(inputFifo)"
            _ = try await client.executeCommand(writeCmd)
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
        guard let client = client else {
            log("sendRequest: ERROR - no SSH client")
            throw ACPConnectionError.notConnected
        }

        let id = "\(nextRequestId)"
        nextRequestId += 1
        log("sendRequest: id=\(id), method=\(method)")

        let request = JSONRPCRequest(id: id, method: method, params: params)

        var data = try encoder.encode(request)
        data.append(UInt8(ascii: "\n"))

        guard let jsonStr = String(data: data, encoding: .utf8) else {
            log("sendRequest: ERROR - failed to encode request")
            throw ACPConnectionError.encodingError("Failed to encode request")
        }

        log("sendRequest: sending JSON (first 300 chars): \(jsonStr.prefix(300))")

        // Base64 encode to avoid shell escaping issues
        let base64Json = data.base64EncodedString()
        log("sendRequest: base64 length=\(base64Json.count)")

        let response = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<JSONRPCResponse, Error>) in
            pendingRequests[id] = cont
            log("sendRequest: registered pending request id=\(id), total pending=\(pendingRequests.count)")

            Task {
                do {
                    let writeCmd = "echo '\(base64Json)' | base64 -d > \(self.inputFifo)"
                    log("sendRequest: executing write command for id=\(id)")
                    _ = try await client.executeCommand(writeCmd)
                    log("sendRequest: write command completed for id=\(id)")
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
        guard let client = client else {
            throw ACPConnectionError.notConnected
        }

        let notification = JSONRPCNotification(method: method, params: params)

        var data = try encoder.encode(notification)
        data.append(UInt8(ascii: "\n"))

        log("sending notification: \(method)")

        let base64Json = data.base64EncodedString()
        let writeCmd = "echo '\(base64Json)' | base64 -d > \(inputFifo)"
        _ = try await client.executeCommand(writeCmd)
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

    func close() async {
        guard !_isClosed else { return }
        _isClosed = true

        heartbeatTask?.cancel()
        heartbeatTask = nil
        outputTask?.cancel()
        outputTask = nil

        // Clean up remote resources
        if let client = client {
            _ = try? await client.executeCommand("pkill -f 'tail.*\(inputFifo)' 2>/dev/null; rm -f \(inputFifo) \(pidFile)")
            try? await client.close()
        }
        client = nil

        for (_, cont) in pendingRequests {
            cont.resume(throwing: ACPConnectionError.connectionClosed)
        }
        pendingRequests.removeAll()

        log("connection closed")
    }

    deinit {
        heartbeatTask?.cancel()
        outputTask?.cancel()
    }
}
