import Foundation

// Simplified Claude service that just works
class SimpleClaudeService: ObservableObject, ChatServiceProtocol {
    
    func sendMessage(
        _ message: String,
        conversationHistory: [ChatMessage],
        onDelta: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) {
        Task {
            do {
                let response = try await executeClaudeCommand(message: message)
                
                // Simulate streaming
                await simulateStreaming(text: response, onDelta: onDelta)
                
                await MainActor.run {
                    onComplete(response)
                }
            } catch {
                await MainActor.run {
                    onError(error)
                }
            }
        }
    }
    
    private func executeClaudeCommand(message: String) async throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/claude")
        process.arguments = ["--print", message]
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        try process.run()
        process.waitUntilExit()
        
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        
        if process.terminationStatus != 0 {
            let errorString = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw SimpleClaudeError.commandFailed("Claude command failed: \(errorString)")
        }
        
        guard let output = String(data: outputData, encoding: .utf8), !output.isEmpty else {
            throw SimpleClaudeError.noResponse
        }
        
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private func simulateStreaming(text: String, onDelta: @escaping (String) -> Void) async {
        // Break text into word chunks for simulated streaming
        let words = text.components(separatedBy: " ")
        
        for (index, word) in words.enumerated() {
            let chunk = index == 0 ? word : " " + word
            
            await MainActor.run {
                onDelta(chunk)
            }
            
            // Small delay between words to simulate streaming
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }
    }
}

enum SimpleClaudeError: LocalizedError {
    case commandFailed(String)
    case noResponse
    
    var errorDescription: String? {
        switch self {
        case .commandFailed(let message):
            return message
        case .noResponse:
            return "No response from Claude command"
        }
    }
}