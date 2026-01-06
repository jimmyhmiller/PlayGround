import Foundation
import ACPLib

// MARK: - CLI Error

enum CLIError: Error, LocalizedError {
    case missingSubcommand(String)
    case unknownSubcommand(String)
    case missingArgument(String)
    case connectionFailed(String)
    case noSession

    var errorDescription: String? {
        switch self {
        case .missingSubcommand(let cmd):
            return "Missing subcommand for '\(cmd)'"
        case .unknownSubcommand(let cmd):
            return "Unknown subcommand: \(cmd)"
        case .missingArgument(let arg):
            return "Missing required argument: \(arg)"
        case .connectionFailed(let reason):
            return "Connection failed: \(reason)"
        case .noSession:
            return "No active session"
        }
    }
}

// MARK: - CLI Permission Delegate

/// Delegate that prompts for permission via stdin
actor CLIPermissionDelegate: ACPClientDelegate {
    func acpClient(_ client: ACPClient, didReceive event: ACPEvent) async {
        // Events are handled in the prompt command
    }

    func acpClient(_ client: ACPClient, requestPermissionFor toolName: String, input: String?, prompt: String?) async -> (granted: Bool, context: String?) {
        // Print permission request
        print("\n" + String(repeating: "=", count: 60))
        print("PERMISSION REQUEST")
        print(String(repeating: "=", count: 60))
        print("Tool: \(toolName)")
        if let input = input {
            let preview = input.count > 500 ? String(input.prefix(500)) + "..." : input
            print("Input: \(preview)")
        }
        if let prompt = prompt {
            print("Prompt: \(prompt)")
        }
        print(String(repeating: "-", count: 60))
        print("Allow this action? [y/N]: ", terminator: "")
        fflush(stdout)

        // Read from stdin
        guard let response = readLine()?.lowercased() else {
            print("No input received, denying permission")
            return (false, "User did not respond")
        }

        let granted = response == "y" || response == "yes"
        print(granted ? "Permission GRANTED" : "Permission DENIED")
        print(String(repeating: "=", count: 60) + "\n")

        return (granted, nil)
    }
}

let permissionDelegate = CLIPermissionDelegate()

// MARK: - Global State

actor CLIState {
    var client: ACPClient?
    var sessionId: String?

    func setClient(_ client: ACPClient) {
        self.client = client
    }

    func setSessionId(_ id: String) {
        self.sessionId = id
    }
}

let state = CLIState()

// MARK: - Argument Parsing

func parseArg(_ args: [String], flag: String) -> String? {
    guard let index = args.firstIndex(of: flag), index + 1 < args.count else {
        return nil
    }
    return args[index + 1]
}

func hasFlag(_ args: [String], flag: String) -> Bool {
    args.contains(flag)
}

// MARK: - Commands

func connectCommand(args: [String]) async throws {
    let cwd = parseArg(args, flag: "--cwd") ?? FileManager.default.currentDirectoryPath
    let acpPath = parseArg(args, flag: "--acp-path") ?? "claude-code-acp"
    let debug = hasFlag(args, flag: "--debug")

    if debug {
        acpDebugLoggingEnabled = true
    }

    print("Connecting to claude-code-acp...")
    print("  Working directory: \(cwd)")
    print("  ACP path: \(acpPath)")

    let client = ACPClient.forClaudeCode(name: "acp-cli", version: acpLibraryVersion)

    do {
        // Set delegate for permission handling
        await client.setDelegate(permissionDelegate)

        try await client.connectToClaudeCodeACP(
            acpPath: acpPath,
            currentDirectory: cwd
        )
        await state.setClient(client)

        if let agentInfo = await client.agentInfo {
            print("Connected to: \(agentInfo.title) v\(agentInfo.version)")
        } else {
            print("Connected!")
        }
    } catch {
        throw CLIError.connectionFailed(error.localizedDescription)
    }
}

func sessionNewCommand(args: [String]) async throws {
    let cwd = parseArg(args, flag: "--cwd") ?? FileManager.default.currentDirectoryPath

    guard let client = await state.client else {
        // Auto-connect if not connected
        try await connectCommand(args: args)
        try await sessionNewCommand(args: args)
        return
    }

    let sessionId = try await client.newSession(cwd: cwd)
    await state.setSessionId(sessionId)

    print("Created session: \(sessionId)")

    // Show modes if available
    let modes = await client.availableModes
    if !modes.isEmpty {
        print("Available modes: \(modes.map { $0.name }.joined(separator: ", "))")
        if let current = await client.currentMode {
            print("Current mode: \(current.name)")
        }
    }
}

func sessionResumeCommand(args: [String]) async throws {
    guard args.count > 0 else {
        throw CLIError.missingArgument("session-id")
    }

    let sessionId = args[0]
    let cwd = parseArg(Array(args.dropFirst()), flag: "--cwd")
        ?? FileManager.default.currentDirectoryPath

    guard let client = await state.client else {
        // Auto-connect if not connected
        try await connectCommand(args: Array(args.dropFirst()))
        try await sessionResumeCommand(args: args)
        return
    }

    let result = try await client.resumeSession(sessionId: sessionId, cwd: cwd)
    await state.setSessionId(result.sessionId)

    print("Resumed session: \(result.sessionId)")
    print("History: \(result.history.count) messages")

    if let modes = result.modes {
        print("Current mode: \(modes.currentModeId)")
        print("Available modes: \(modes.availableModes.map { $0.name }.joined(separator: ", "))")
    }
}

func sessionHistoryCommand(args: [String]) async throws {
    guard args.count > 0 else {
        throw CLIError.missingArgument("session-id")
    }

    let sessionId = args[0]
    let cwd = parseArg(Array(args.dropFirst()), flag: "--cwd")
        ?? FileManager.default.currentDirectoryPath

    let history = try await SessionHistoryLoader.loadHistory(sessionId: sessionId, cwd: cwd)

    print("Session history (\(history.count) messages):\n")

    for message in history {
        let rolePrefix = message.role == .user ? "[User]" : "[Assistant]"
        let preview = message.content.prefix(200)
        let truncated = message.content.count > 200 ? "..." : ""
        print("\(rolePrefix) \(preview)\(truncated)\n")
    }
}

func sessionListCommand(args: [String]) async throws {
    let cwd = parseArg(args, flag: "--cwd") ?? FileManager.default.currentDirectoryPath

    let sessions = try SessionHistoryLoader.listSessions(cwd: cwd)

    if sessions.isEmpty {
        print("No sessions found for: \(cwd)")
    } else {
        print("Sessions for \(cwd):\n")
        for session in sessions {
            print("  \(session)")
        }
    }
}

func sessionCommand(args: [String]) async throws {
    guard args.count > 0 else {
        throw CLIError.missingSubcommand("session")
    }

    let subcommand = args[0]
    let subArgs = Array(args.dropFirst())

    switch subcommand {
    case "new":
        try await sessionNewCommand(args: subArgs)
    case "resume":
        try await sessionResumeCommand(args: subArgs)
    case "history":
        try await sessionHistoryCommand(args: subArgs)
    case "list":
        try await sessionListCommand(args: subArgs)
    default:
        throw CLIError.unknownSubcommand(subcommand)
    }
}

func promptCommand(args: [String]) async throws {
    guard args.count > 0 else {
        throw CLIError.missingArgument("message")
    }

    let message = args[0]
    let cwd = parseArg(Array(args.dropFirst()), flag: "--cwd")
        ?? FileManager.default.currentDirectoryPath

    guard let client = await state.client else {
        try await connectCommand(args: Array(args.dropFirst()))
        try await promptCommand(args: args)
        return
    }

    // Ensure we have a session
    if await state.sessionId == nil {
        _ = try await client.newSession(cwd: cwd)
    }

    // Start event listener for streaming output
    let eventTask = Task {
        for await event in await client.events {
            switch event {
            case .textChunk(let text, _):
                print(text, terminator: "")
                fflush(stdout)
            case .thinkingChunk(let text, _):
                print("[thinking] \(text.prefix(100))...", terminator: "")
            case .toolCallStarted(_, let name, _, _):
                print("\n[Tool: \(name)]")
            case .toolCallUpdate(_, let status, let output, let error, _):
                if let error = error {
                    print("[Error: \(error)]")
                } else if let output = output, status == "complete" {
                    let preview = output.prefix(100)
                    print("[Output: \(preview)\(output.count > 100 ? "..." : "")]")
                }
            case .planStep(_, let title, let status, _):
                print("[Plan: \(title) - \(status)]")
            case .promptComplete(let reason, _):
                print("\n[Done: \(reason)]")
            case .promptInterrupted(let text, _):
                print("\n\(text)")
            case .modeChanged(let modeId):
                print("[Mode: \(modeId)]")
            case .error(let err):
                print("\n[Error: \(err)]")
            default:
                break
            }
        }
    }

    // Send prompt
    _ = try await client.prompt(text: message)

    eventTask.cancel()
}

func cancelCommand(args: [String]) async throws {
    guard let client = await state.client else {
        print("Not connected")
        return
    }

    try await client.cancel()
    print("Cancelled")
}

func modeListCommand(args: [String]) async throws {
    guard let client = await state.client else {
        print("Not connected")
        return
    }

    let modes = await client.availableModes

    if modes.isEmpty {
        print("No modes available")
    } else {
        let currentId = await client.currentModeId
        print("Available modes:")
        for mode in modes {
            let marker = mode.id == currentId ? " *" : ""
            print("  \(mode.id): \(mode.name)\(marker)")
        }
    }
}

func modeSetCommand(args: [String]) async throws {
    guard args.count > 0 else {
        throw CLIError.missingArgument("mode-id")
    }

    let modeId = args[0]

    guard let client = await state.client else {
        print("Not connected")
        return
    }

    try await client.setMode(modeId)
    print("Mode set to: \(modeId)")
}

func modeCycleCommand(args: [String]) async throws {
    guard let client = await state.client else {
        print("Not connected")
        return
    }

    let mode = try await client.cycleMode()
    print("Cycled to mode: \(mode.name) (\(mode.id))")
}

func modeCommand(args: [String]) async throws {
    guard args.count > 0 else {
        throw CLIError.missingSubcommand("mode")
    }

    let subcommand = args[0]
    let subArgs = Array(args.dropFirst())

    switch subcommand {
    case "list":
        try await modeListCommand(args: subArgs)
    case "set":
        try await modeSetCommand(args: subArgs)
    case "cycle":
        try await modeCycleCommand(args: subArgs)
    default:
        throw CLIError.unknownSubcommand(subcommand)
    }
}

func printUsage() {
    print("""
    acp-cli - ACP Library CLI Tool v\(acpLibraryVersion)

    Usage:
      acp-cli connect [--cwd <path>] [--acp-path <path>] [--debug]
                                           Connect to claude-code-acp
      acp-cli session new [--cwd <path>]   Create new session
      acp-cli session resume <id> [--cwd <path>]
                                           Resume session
      acp-cli session history <id> [--cwd <path>]
                                           Load session history
      acp-cli session list [--cwd <path>]  List sessions for directory
      acp-cli prompt "message"             Send prompt
      acp-cli cancel                       Cancel current prompt
      acp-cli mode list                    List available modes
      acp-cli mode set <mode-id>           Change mode
      acp-cli mode cycle                   Cycle to next mode

    Options:
      --cwd <path>       Working directory (default: current directory)
      --acp-path <path>  Path to claude-code-acp (default: claude-code-acp)
      --debug            Enable debug logging

    Examples:
      acp-cli connect --cwd /path/to/project
      acp-cli session new
      acp-cli prompt "Create a hello world program"
      acp-cli mode set plan
    """)
}

// MARK: - Main

@main
struct ACPCli {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count > 1 else {
            printUsage()
            return
        }

        let command = args[1]
        let subArgs = Array(args.dropFirst(2))

        do {
            switch command {
            case "connect":
                try await connectCommand(args: subArgs)
            case "session":
                try await sessionCommand(args: subArgs)
            case "prompt":
                try await promptCommand(args: subArgs)
            case "cancel":
                try await cancelCommand(args: subArgs)
            case "mode":
                try await modeCommand(args: subArgs)
            case "help", "--help", "-h":
                printUsage()
            default:
                print("Unknown command: \(command)")
                printUsage()
                exit(1)
            }
        } catch {
            print("Error: \(error.localizedDescription)")
            exit(1)
        }
    }
}
