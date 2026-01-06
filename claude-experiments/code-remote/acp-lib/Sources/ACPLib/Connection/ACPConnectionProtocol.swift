import Foundation

// MARK: - ACP Connection Protocol

/// Protocol for ACP connections
public protocol ACPConnectionProtocol: Actor {
    /// Send a request and wait for the response
    func sendRequest<P: Encodable, R: Decodable>(method: String, params: P?) async throws -> R

    /// Send a notification (no response expected)
    func sendNotification<P: Encodable>(method: String, params: P?) async throws

    /// Close the connection
    func close() async

    /// Whether the connection is closed
    var isClosed: Bool { get async }
}
