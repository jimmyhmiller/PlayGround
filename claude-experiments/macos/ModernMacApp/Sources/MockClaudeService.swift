import Foundation

// Mock service for testing chat functionality without the real claude command
class MockClaudeService: ObservableObject {
    
    func sendMessage(
        _ message: String,
        conversationHistory: [ChatMessage],
        onDelta: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) {
        print("🧪 MockClaudeService: Received message: \(message)")
        
        // Simulate typing delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.simulateStreamingResponse(
                to: message,
                onDelta: onDelta,
                onComplete: onComplete
            )
        }
    }
    
    private func simulateStreamingResponse(
        to message: String,
        onDelta: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void
    ) {
        let responses = [
            "Hello! I'm a mock Claude assistant. ",
            "You sent me the message: '\(message)'. ",
            "This is a simulated streaming response ",
            "to test the chat interface. ",
            "Everything seems to be working well! ",
            "How can I help you today?"
        ]
        
        var fullResponse = ""
        
        for (index, chunk) in responses.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(index) * 0.3) {
                fullResponse += chunk
                onDelta(chunk)
                
                // Call onComplete after the last chunk
                if index == responses.count - 1 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        onComplete(fullResponse)
                    }
                }
            }
        }
    }
}

// Test service that always fails - for testing error handling
class FailingMockClaudeService: ObservableObject {
    
    func sendMessage(
        _ message: String,
        conversationHistory: [ChatMessage],
        onDelta: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) {
        print("🧪 FailingMockClaudeService: Simulating error for message: \(message)")
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            onError(MockError.simulatedFailure)
        }
    }
}

enum MockError: LocalizedError {
    case simulatedFailure
    
    var errorDescription: String? {
        switch self {
        case .simulatedFailure:
            return "This is a simulated error for testing purposes"
        }
    }
}