import Foundation

class ClaudeService: ObservableObject {
    
    init() {
        // Check if claude command is available
        if !isClaudeAvailable() {
            print("⚠️ Claude CLI not found")
            print("   Make sure 'claude' command is installed and in PATH")
            print("   Install with: npm install -g @anthropics/claude-cli")
        }
    }
    
    private func isClaudeAvailable() -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = ["claude"]
        
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
    
    func sendMessage(
        _ message: String,
        conversationHistory: [ChatMessage],
        onDelta: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) {
        guard isClaudeAvailable() else {
            onError(ClaudeError.claudeNotFound)
            return
        }
        
        Task {
            do {
                let fullResponse = try await streamClaudeCommand(
                    message: message,
                    conversationHistory: conversationHistory,
                    onDelta: onDelta
                )
                
                await MainActor.run {
                    onComplete(fullResponse)
                }
            } catch {
                await MainActor.run {
                    onError(error)
                }
            }
        }
    }
    
    private func streamClaudeCommand(
        message: String,
        conversationHistory: [ChatMessage],
        onDelta: @escaping (String) -> Void
    ) async throws -> String {
        
        // Use direct path to avoid env issues that cause Cocoa error 3587
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/claude")
        
        // Simple message without complex conversation formatting for now
        process.arguments = ["--print", message]
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        var fullResponse = ""
        
        try process.run()
        
        // Read streaming output
        let outputHandle = outputPipe.fileHandleForReading
        
        try process.run()
        process.waitUntilExit()
        
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        
        if process.terminationStatus != 0 {
            let errorString = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw ClaudeError.commandFailed(errorString)
        }
        
        guard let output = String(data: outputData, encoding: .utf8) else {
            throw ClaudeError.invalidResponse
        }
        
        // Parse claude JSON output
        let lines = output.components(separatedBy: .newlines)
        
        for line in lines {
            guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            
            do {
                if let data = line.data(using: .utf8),
                   let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    
                    // Look for assistant message
                    if json["type"] as? String == "assistant",
                       let message = json["message"] as? [String: Any],
                       let content = message["content"] as? [[String: Any]] {
                        
                        for item in content {
                            if item["type"] as? String == "text",
                               let text = item["text"] as? String {
                                
                                fullResponse = text
                                
                                // Simulate streaming by breaking text into chunks
                                await simulateStreaming(text: text, onDelta: onDelta)
                                
                                return fullResponse
                            }
                        }
                    }
                }
            } catch {
                // Skip malformed JSON lines
                continue
            }
        }
        
        return fullResponse
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

enum ClaudeError: LocalizedError {
    case claudeNotFound
    case commandFailed(String)
    case invalidResponse
    case processError
    
    var errorDescription: String? {
        switch self {
        case .claudeNotFound:
            return "Claude CLI not found. Make sure 'claude' command is installed and in PATH."
        case .commandFailed(let error):
            return "Claude command failed: \(error)"
        case .invalidResponse:
            return "Invalid response from Claude command"
        case .processError:
            return "Failed to execute Claude process"
        }
    }
}