import Foundation

// MARK: - ACP Connection Errors

enum ACPConnectionError: Error, LocalizedError {
    case notConnected
    case processTerminated(exitCode: Int32)
    case encodingError(String)
    case decodingError(String)
    case timeout
    case connectionClosed
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .notConnected:
            return "Not connected to ACP agent"
        case .processTerminated(let code):
            return "Agent process terminated with code \(code)"
        case .encodingError(let msg):
            return "Encoding error: \(msg)"
        case .decodingError(let msg):
            return "Decoding error: \(msg)"
        case .timeout:
            return "Request timed out"
        case .connectionClosed:
            return "Connection closed"
        case .invalidResponse(let msg):
            return "Invalid response: \(msg)"
        }
    }
}

// MARK: - ACP Connection Protocol

protocol ACPConnectionProtocol: Actor {
    func sendRequest<P: Encodable, R: Decodable>(method: String, params: P?) async throws -> R
    func sendNotification<P: Encodable>(method: String, params: P?) async throws
    func close() async
    var isClosed: Bool { get async }
}

// MARK: - Subprocess ACP Connection

/// Manages a connection to an ACP agent via subprocess (stdin/stdout)
actor ACPSubprocessConnection: ACPConnectionProtocol {
    private let process: Process
    private let stdin: FileHandle
    private let stdout: FileHandle
    private let stderr: FileHandle

    private var nextRequestId: Int = 0
    private var pendingRequests: [String: CheckedContinuation<JSONRPCResponse, Error>] = [:]
    private var _isClosed = false
    private var readTask: Task<Void, Never>?
    private var notificationHandler: (@Sendable (JSONRPCNotification) async -> Void)?

    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    var isClosed: Bool { _isClosed }

    init(
        command: String,
        arguments: [String] = [],
        environment: [String: String]? = nil,
        currentDirectory: String? = nil
    ) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command)
        process.arguments = arguments

        if let env = environment {
            var processEnv = ProcessInfo.processInfo.environment
            for (key, value) in env {
                processEnv[key] = value
            }
            process.environment = processEnv
        }

        if let dir = currentDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: dir)
        }

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        self.process = process
        self.stdin = stdinPipe.fileHandleForWriting
        self.stdout = stdoutPipe.fileHandleForReading
        self.stderr = stderrPipe.fileHandleForReading

        try process.run()
    }

    func setNotificationHandler(_ handler: @escaping @Sendable (JSONRPCNotification) async -> Void) {
        self.notificationHandler = handler
    }

    func startReading() {
        readTask = Task { [weak self] in
            guard let self = self else { return }
            await self.readLoop()
        }
    }

    private func readLoop() async {
        var buffer = Data()

        while !_isClosed {
            do {
                // Read available data
                let data = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data, Error>) in
                    DispatchQueue.global().async {
                        let available = self.stdout.availableData
                        if available.isEmpty {
                            // Check if process is still running
                            if !self.process.isRunning {
                                cont.resume(throwing: ACPConnectionError.processTerminated(exitCode: self.process.terminationStatus))
                            } else {
                                cont.resume(returning: Data())
                            }
                        } else {
                            cont.resume(returning: available)
                        }
                    }
                }

                if data.isEmpty {
                    try await Task.sleep(nanoseconds: 10_000_000) // 10ms
                    continue
                }

                buffer.append(data)

                // Process complete lines (NDJSON)
                while let newlineIndex = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                    let lineData = buffer.prefix(upTo: newlineIndex)
                    buffer = Data(buffer.suffix(from: buffer.index(after: newlineIndex)))

                    guard !lineData.isEmpty else { continue }

                    do {
                        let message = try decoder.decode(JSONRPCMessage.self, from: Data(lineData))
                        await handleMessage(message)
                    } catch {
                        print("[ACPConnection] Failed to decode message: \(error)")
                        if let str = String(data: Data(lineData), encoding: .utf8) {
                            print("[ACPConnection] Raw line: \(str.prefix(200))")
                        }
                    }
                }

            } catch {
                if !_isClosed {
                    print("[ACPConnection] Read error: \(error)")
                }
                break
            }
        }

        // Clean up pending requests
        for (_, cont) in pendingRequests {
            cont.resume(throwing: ACPConnectionError.connectionClosed)
        }
        pendingRequests.removeAll()
    }

    private func handleMessage(_ message: JSONRPCMessage) async {
        switch message {
        case .response(let response):
            if let cont = pendingRequests.removeValue(forKey: response.id) {
                cont.resume(returning: response)
            }

        case .notification(let notification):
            await notificationHandler?(notification)

        case .request(let request):
            // Handle incoming requests from agent (like permission requests)
            print("[ACPConnection] Received request from agent: \(request.method)")
            // TODO: Implement request handling
        }
    }

    func sendRequest<P: Encodable, R: Decodable>(method: String, params: P?) async throws -> R {
        guard !_isClosed else {
            throw ACPConnectionError.notConnected
        }

        let id = "\(nextRequestId)"
        nextRequestId += 1

        let request = JSONRPCRequest(id: id, method: method, params: params)

        let response = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<JSONRPCResponse, Error>) in
            pendingRequests[id] = cont

            Task {
                do {
                    try await self.writeMessage(request)
                } catch {
                    if let removed = self.pendingRequests.removeValue(forKey: id) {
                        removed.resume(throwing: error)
                    }
                }
            }
        }

        switch response {
        case .success(let successResponse):
            return try successResponse.result.decode(R.self)
        case .error(let errorResponse):
            throw errorResponse.error
        }
    }

    func sendNotification<P: Encodable>(method: String, params: P?) async throws {
        guard !_isClosed else {
            throw ACPConnectionError.notConnected
        }

        let notification = JSONRPCNotification(method: method, params: params)
        try await writeMessage(notification)
    }

    private func writeMessage<T: Encodable>(_ message: T) async throws {
        var data = try encoder.encode(message)
        data.append(UInt8(ascii: "\n"))

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            DispatchQueue.global().async {
                do {
                    try self.stdin.write(contentsOf: data)
                    cont.resume()
                } catch {
                    cont.resume(throwing: ACPConnectionError.encodingError(error.localizedDescription))
                }
            }
        }
    }

    func close() async {
        guard !_isClosed else { return }
        _isClosed = true

        readTask?.cancel()

        // Clean up pending requests
        for (_, cont) in pendingRequests {
            cont.resume(throwing: ACPConnectionError.connectionClosed)
        }
        pendingRequests.removeAll()

        // Terminate process
        if process.isRunning {
            process.terminate()
        }

        try? stdin.close()
        try? stdout.close()
        try? stderr.close()
    }

    deinit {
        if process.isRunning {
            process.terminate()
        }
    }
}

// MARK: - Mock Connection for Testing

actor ACPMockConnection: ACPConnectionProtocol {
    private var _isClosed = false
    var isClosed: Bool { _isClosed }

    var mockResponses: [String: Any] = [:]

    func setMockResponse<R: Codable>(for method: String, response: R) {
        mockResponses[method] = response
    }

    func sendRequest<P: Encodable, R: Decodable>(method: String, params: P?) async throws -> R {
        guard !_isClosed else {
            throw ACPConnectionError.notConnected
        }

        if let response = mockResponses[method] as? R {
            return response
        }

        throw ACPConnectionError.invalidResponse("No mock response for method: \(method)")
    }

    func sendNotification<P: Encodable>(method: String, params: P?) async throws {
        guard !_isClosed else {
            throw ACPConnectionError.notConnected
        }
    }

    func close() async {
        _isClosed = true
    }
}
