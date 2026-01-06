import Foundation

func log(_ msg: String) {
    fputs(msg + "\n", stderr)
    fflush(stderr)
}

@main
struct ACPTest {
    static func main() async {
        log("ACP Test Starting...")
        log("Using claude-code-acp at: /opt/homebrew/bin/claude-code-acp")

        let client = ACPClient.forClaudeCode(name: "acp-test", version: "1.0.0")

        // Start event listener
        let eventTask = Task {
            for await event in await client.events {
                switch event {
                case .connected(let info):
                    log("Connected to agent: \(info.title) v\(info.version)")
                case .sessionCreated(let id):
                    log("Session created: \(id)")
                case .textChunk(let text):
                    // Print text to stdout
                    print(text, terminator: "")
                    fflush(stdout)
                case .thinkingChunk(let text):
                    log("[Thinking] \(text.prefix(50))...")
                case .toolCallStarted(let id, let name, _):
                    log("Tool started: \(name) (\(id))")
                case .toolCallUpdate(let id, let status, _, _):
                    log("Tool update: \(id) -> \(status)")
                case .promptComplete(let reason):
                    log("\nPrompt complete: \(reason)")
                case .error(let msg):
                    log("ERROR: \(msg)")
                default:
                    break
                }
            }
        }

        do {
            // Connect to claude-code-acp
            log("Connecting...")
            try await client.connect(
                command: "/opt/homebrew/bin/claude-code-acp",
                arguments: [],
                currentDirectory: FileManager.default.currentDirectoryPath
            )
            log("Connected!")

            // Create a new session
            log("Creating session...")
            let sessionId = try await client.newSession(cwd: FileManager.default.currentDirectoryPath)
            log("Session: \(sessionId)")

            // Send a simple prompt
            log("Sending prompt...")
            let result = try await client.prompt(text: "Say hello in exactly 5 words.")
            log("Result: stopReason=\(result.stopReason)")

            // Wait a bit for events to finish
            try await Task.sleep(nanoseconds: 1_000_000_000)

            // Disconnect
            log("Disconnecting...")
            await client.disconnect()
            log("Done!")

        } catch {
            log("Error: \(error)")
        }

        eventTask.cancel()
    }
}
