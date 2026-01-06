import Foundation

// MARK: - Mock Connection for Testing

public actor ACPMockConnection: ACPConnectionProtocol {
    private var _isClosed = false
    public var isClosed: Bool { _isClosed }

    private var mockResponses: [String: Any] = [:]
    private var notificationHandler: (@Sendable (JSONRPCNotification) async -> Void)?
    private var requestHandler: (@Sendable (JSONRPCRequest) async -> JSONRPCResponse?)?

    public init() {}

    public func setMockResponse<R: Codable>(for method: String, response: R) {
        mockResponses[method] = response
    }

    public func setNotificationHandler(_ handler: @escaping @Sendable (JSONRPCNotification) async -> Void) {
        self.notificationHandler = handler
    }

    public func setRequestHandler(_ handler: @escaping @Sendable (JSONRPCRequest) async -> JSONRPCResponse?) {
        self.requestHandler = handler
    }

    /// Simulate receiving a notification from the agent
    public func simulateNotification(_ notification: JSONRPCNotification) async {
        await notificationHandler?(notification)
    }

    /// Simulate receiving a session update
    public func simulateSessionUpdate(sessionId: String, update: ACPSessionUpdate) async {
        let params = ACPSessionUpdateParams(sessionId: sessionId, update: update)
        let notification = JSONRPCNotification(method: "session/update", params: params)
        await notificationHandler?(notification)
    }

    /// Simulate receiving an incoming request from the agent (e.g., permission request)
    /// Returns the response from the client's request handler
    public func simulateRequest(_ request: JSONRPCRequest) async -> JSONRPCResponse? {
        return await requestHandler?(request)
    }

    /// Simulate a permission request and return the response
    public func simulatePermissionRequest(
        sessionId: String,
        toolName: String? = nil,
        toolCallId: String? = nil,
        input: AnyCodableValue? = nil,
        prompt: String? = nil
    ) async -> ACPRequestPermissionResult? {
        let options = [
            ACPPermissionOption(kind: "allow_always", name: "Always Allow", optionId: "allow_always"),
            ACPPermissionOption(kind: "allow_once", name: "Allow", optionId: "allow"),
            ACPPermissionOption(kind: "reject_once", name: "Reject", optionId: "reject")
        ]
        let params = ACPRequestPermissionParams(
            options: options,
            sessionId: sessionId,
            toolName: toolName,
            toolCallId: toolCallId,
            input: input,
            prompt: prompt
        )
        let request = JSONRPCRequest(id: .number(0), method: "session/request_permission", params: params)

        if let response = await requestHandler?(request) {
            switch response {
            case .success(let success):
                return try? success.result.decode(ACPRequestPermissionResult.self)
            case .error:
                return nil
            }
        }
        return nil
    }

    public func sendRequest<P: Encodable, R: Decodable>(method: String, params: P?) async throws -> R {
        guard !_isClosed else {
            throw ACPConnectionError.notConnected
        }

        if let response = mockResponses[method] as? R {
            return response
        }

        throw ACPConnectionError.invalidResponse("No mock response for method: \(method)")
    }

    public func sendNotification<P: Encodable>(method: String, params: P?) async throws {
        guard !_isClosed else {
            throw ACPConnectionError.notConnected
        }
    }

    public func close() async {
        _isClosed = true
    }
}
