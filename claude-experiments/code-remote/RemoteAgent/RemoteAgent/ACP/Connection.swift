import Foundation

// MARK: - ACP Connection Errors

enum ACPConnectionError: Error, LocalizedError {
    case notConnected
    case processTerminated(exitCode: Int32, stderr: String?)
    case encodingError(String)
    case decodingError(String)
    case timeout
    case connectionClosed
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .notConnected:
            return "Not connected to ACP agent"
        case .processTerminated(let code, let stderr):
            if let stderr = stderr, !stderr.isEmpty {
                // Extract the key error message
                let lines = stderr.components(separatedBy: "\n")
                let errorLines = lines.filter {
                    $0.contains("Error") || $0.contains("error:") || $0.contains("SyntaxError")
                }
                if let firstError = errorLines.first {
                    return "Agent failed: \(firstError)"
                }
                return "Agent failed (code \(code)): \(stderr.prefix(200))"
            }
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
    private var stderrBuffer: String = ""

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

        print("[ACPConnection] Starting process: \(command) \(arguments.joined(separator: " "))")
        if let dir = currentDirectory {
            print("[ACPConnection] Working directory: \(dir)")
        }
        try process.run()
        print("[ACPConnection] Process started with PID: \(process.processIdentifier)")
    }

    func setNotificationHandler(_ handler: @escaping @Sendable (JSONRPCNotification) async -> Void) {
        self.notificationHandler = handler
    }

    private var readBuffer = Data()

    func startReading() {
        // Use readabilityHandler for stdout - this works reliably
        stdout.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }

            Task { [weak self] in
                await self?.handleStdoutData(data)
            }
        }

        // Use readabilityHandler for stderr
        stderr.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                Task { [weak self] in
                    await self?.appendStderr(str)
                }
            }
        }

        // Monitor process termination
        readTask = Task { [weak self] in
            while let self = self, await !self.isClosed {
                if !self.process.isRunning {
                    await self.handleProcessTermination()
                    break
                }
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
        }
    }

    private func appendStderr(_ str: String) {
        stderrBuffer += str
        print("[ACPConnection STDERR] \(str)")
    }

    private func handleStdoutData(_ data: Data) async {
        readBuffer.append(data)

        if let str = String(data: data, encoding: .utf8) {
            print("[ACPConnection] Received: \(str.prefix(300))")
        }

        // Process complete lines (NDJSON)
        while let newlineIndex = readBuffer.firstIndex(of: UInt8(ascii: "\n")) {
            let lineData = readBuffer.prefix(upTo: newlineIndex)
            readBuffer = Data(readBuffer.suffix(from: readBuffer.index(after: newlineIndex)))

            guard !lineData.isEmpty else { continue }

            do {
                let message = try decoder.decode(JSONRPCMessage.self, from: Data(lineData))
                print("[ACPConnection] Decoded: \(type(of: message))")
                await handleMessage(message)
            } catch {
                print("[ACPConnection] Decode error: \(error)")
                if let str = String(data: Data(lineData), encoding: .utf8) {
                    print("[ACPConnection] Raw: \(str.prefix(500))")
                }
            }
        }
    }

    private func handleProcessTermination() async {
        let exitCode = process.terminationStatus
        print("[ACPConnection] Process terminated with exit code: \(exitCode)")

        let stderrStr = stderrBuffer
        let error = ACPConnectionError.processTerminated(exitCode: exitCode, stderr: stderrStr.isEmpty ? nil : stderrStr)

        // Resume pending requests with error
        for (_, cont) in pendingRequests {
            cont.resume(throwing: error)
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

        if let str = String(data: data, encoding: .utf8) {
            print("[ACPConnection] Sending: \(str.prefix(500))")
        }

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
