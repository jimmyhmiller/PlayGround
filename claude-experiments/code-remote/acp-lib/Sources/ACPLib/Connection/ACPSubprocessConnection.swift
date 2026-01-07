import Foundation

// MARK: - Subprocess ACP Connection

#if os(macOS)
/// Manages a connection to an ACP agent via subprocess (stdin/stdout)
public actor ACPSubprocessConnection: ACPConnectionProtocol {
    private let process: Process
    private let stdin: FileHandle
    private let stdout: FileHandle
    private let stderr: FileHandle

    private var nextRequestId: Int = 0
    private var pendingRequests: [JSONRPCId: CheckedContinuation<JSONRPCResponse, Error>] = [:]
    private var _isClosed = false
    private var readTask: Task<Void, Never>?
    private var notificationHandler: (@Sendable (JSONRPCNotification) async -> Void)?
    private var requestHandler: (@Sendable (JSONRPCRequest) async -> JSONRPCResponse?)?
    private var stderrBuffer: String = ""

    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public var isClosed: Bool { _isClosed }

    /// Access to stderr buffer for debugging
    public var stderrOutput: String { stderrBuffer }

    public init(
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

        acpLog("Starting process: \(command) \(arguments.joined(separator: " "))")
        if let dir = currentDirectory {
            acpLog("Working directory: \(dir)")
        }
        try process.run()
        acpLog("Process started with PID: \(process.processIdentifier)")
    }

    public func setNotificationHandler(_ handler: @escaping @Sendable (JSONRPCNotification) async -> Void) {
        self.notificationHandler = handler
    }

    public func setRequestHandler(_ handler: @escaping @Sendable (JSONRPCRequest) async -> JSONRPCResponse?) {
        self.requestHandler = handler
    }

    private var readBuffer = Data()

    public func startReading() {
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
        acpLog("STDERR: \(str)")
    }

    private func handleStdoutData(_ data: Data) async {
        readBuffer.append(data)

        if let str = String(data: data, encoding: .utf8) {
            acpLog("Received: \(str.prefix(2000))")
        }

        // Process complete lines (NDJSON)
        while let newlineIndex = readBuffer.firstIndex(of: UInt8(ascii: "\n")) {
            let lineData = readBuffer.prefix(upTo: newlineIndex)
            readBuffer = Data(readBuffer.suffix(from: readBuffer.index(after: newlineIndex)))

            guard !lineData.isEmpty else { continue }

            do {
                let message = try decoder.decode(JSONRPCMessage.self, from: Data(lineData))
                acpLog("Decoded: \(type(of: message))")
                await handleMessage(message)
            } catch {
                acpLog("Decode error: \(error)")
                if let str = String(data: Data(lineData), encoding: .utf8) {
                    acpLog("Raw: \(str.prefix(500))")
                }
            }
        }
    }

    private func handleProcessTermination() async {
        let exitCode = process.terminationStatus
        acpLog("Process terminated with exit code: \(exitCode)")

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
            acpLog("handleMessage: response for id=\(response.id)")
            if let cont = pendingRequests.removeValue(forKey: response.id) {
                acpLog("handleMessage: resuming continuation for id=\(response.id)")
                cont.resume(returning: response)
            } else {
                acpLog("handleMessage: WARNING - no pending request for id=\(response.id)")
            }

        case .notification(let notification):
            acpLog("handleMessage: notification method=\(notification.method)")
            if notificationHandler != nil {
                await notificationHandler?(notification)
                acpLog("handleMessage: notification handler completed for \(notification.method)")
            } else {
                acpLog("handleMessage: WARNING - no notification handler set for \(notification.method)")
            }

        case .request(let request):
            // Handle incoming requests from agent (like permission requests)
            acpLog("handleMessage: incoming request from agent: \(request.method), id=\(request.id)")
            if let handler = requestHandler {
                if let response = await handler(request) {
                    acpLog("handleMessage: sending response for request \(request.id)")
                    try? await writeMessage(response)
                } else {
                    acpLog("handleMessage: no response for request \(request.id)")
                }
            } else {
                acpLog("handleMessage: WARNING - no request handler for \(request.method)")
            }
        }
    }

    public func sendRequest<P: Encodable, R: Decodable>(method: String, params: P?) async throws -> R {
        guard !_isClosed else {
            acpLogError("sendRequest: connection is closed")
            throw ACPConnectionError.notConnected
        }

        let idNum = nextRequestId
        nextRequestId += 1
        let requestId = JSONRPCId.number(idNum)

        acpLog("sendRequest: method=\(method), id=\(requestId)")
        let request = JSONRPCRequest(id: requestId, method: method, params: params)

        let response = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<JSONRPCResponse, Error>) in
            pendingRequests[requestId] = cont
            acpLog("sendRequest: registered pending request id=\(requestId), total pending=\(self.pendingRequests.count)")

            Task {
                do {
                    try await self.writeMessage(request)
                    acpLog("sendRequest: message written for id=\(requestId)")
                } catch {
                    acpLogError("sendRequest: failed to write message for id=\(requestId): \(error)")
                    if let removed = self.pendingRequests.removeValue(forKey: requestId) {
                        removed.resume(throwing: error)
                    }
                }
            }
        }

        acpLog("sendRequest: received response for id=\(requestId)")
        switch response {
        case .success(let successResponse):
            acpLog("sendRequest: success response for id=\(requestId)")
            return try successResponse.result.decode(R.self)
        case .error(let errorResponse):
            acpLogError("sendRequest: error response for id=\(requestId): \(errorResponse.error)")
            throw errorResponse.error
        }
    }

    public func sendNotification<P: Encodable>(method: String, params: P?) async throws {
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
            acpLog("Sending: \(str.prefix(500))")
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

    public func close() async {
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
#endif
