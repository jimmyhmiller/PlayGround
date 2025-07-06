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
                print("🚀 Starting Claude command...")
                
                // Stream the response directly as it comes in
                try await streamClaudeCommand(message: message, onDelta: onDelta, onComplete: onComplete)
                
            } catch {
                print("❌ Claude command failed: \(error)")
                await MainActor.run {
                    onError(error)
                }
            }
        }
    }
    
    private func streamClaudeCommand(
        message: String,
        onDelta: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void
    ) async throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/claude")
        process.arguments = ["--print", "--output-format", "stream-json", "--verbose", message]
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        let outputHandle = outputPipe.fileHandleForReading
        
        // Start the process
        try process.run()
        
        var fullResponse = ""
        var buffer = ""
        
        // Read output as it streams in
        await withTaskGroup(of: Void.self) { group in
            group.addTask { [self] in
                // Read streaming output
                while process.isRunning {
                    let data = outputHandle.availableData
                    if !data.isEmpty {
                        if let chunk = String(data: data, encoding: .utf8) {
                            buffer += chunk
                            
                            // Process complete JSON lines
                            let lines = buffer.components(separatedBy: "\n")
                            buffer = lines.last ?? "" // Keep incomplete line in buffer
                            
                            for line in lines.dropLast() {
                                if !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                    if let content = parseStreamingJSON(line) {
                                        fullResponse += content
                                        await MainActor.run {
                                            onDelta(content)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    try? await Task.sleep(nanoseconds: 10_000_000) // 10ms check interval
                }
                
                // Read any remaining data
                let remainingData = outputHandle.readDataToEndOfFile()
                if !remainingData.isEmpty {
                    if let chunk = String(data: remainingData, encoding: .utf8) {
                        buffer += chunk
                        
                        // Process any remaining complete lines
                        let lines = buffer.components(separatedBy: "\n")
                        for line in lines {
                            if !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                if let content = parseStreamingJSON(line) {
                                    fullResponse += content
                                    await MainActor.run {
                                        onDelta(content)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Wait for process to complete
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let errorString = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw SimpleClaudeError.commandFailed("Claude command failed: \(errorString)")
        }
        
        await MainActor.run {
            onComplete(fullResponse.trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }
    
    private func parseStreamingJSON(_ line: String) -> String? {
        guard let data = line.data(using: .utf8) else { return nil }
        
        do {
            let json = try JSONSerialization.jsonObject(with: data, options: [])
            
            if let dict = json as? [String: Any] {
                // Handle different streaming JSON event types
                if let type = dict["type"] as? String {
                    switch type {
                    case "assistant":
                        // Extract text from assistant messages
                        if let message = dict["message"] as? [String: Any],
                           let content = message["content"] as? [[String: Any]] {
                            
                            for contentBlock in content {
                                if let contentType = contentBlock["type"] as? String,
                                   contentType == "text",
                                   let text = contentBlock["text"] as? String {
                                    return text
                                }
                            }
                        }
                        return nil
                    case "system", "user", "result":
                        // Control events, no content to extract
                        return nil
                    default:
                        print("Unknown event type: \(type)")
                        return nil
                    }
                }
            }
        } catch {
            print("Failed to parse JSON line: \(error)")
            print("Line: \(line)")
        }
        
        return nil
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