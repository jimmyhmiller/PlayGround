import Foundation

// Protocol that both real and mock services conform to
protocol ChatServiceProtocol {
    func sendMessage(
        _ message: String,
        conversationHistory: [ChatMessage],
        onDelta: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    )
}

// Make our services conform to the protocol
extension ClaudeService: ChatServiceProtocol {}
extension MockClaudeService: ChatServiceProtocol {}