import Foundation

// MARK: - SSH Configuration

/// Configuration for SSH-based terminal execution
public struct SSHConfiguration: Sendable {
    public let host: String
    public let username: String
    public let keyPath: String?
    public let remoteWorkingDirectory: String?

    public init(host: String, username: String, keyPath: String? = nil, remoteWorkingDirectory: String? = nil) {
        self.host = host
        self.username = username
        self.keyPath = keyPath
        self.remoteWorkingDirectory = remoteWorkingDirectory
    }
}

#if os(macOS)
// MARK: - Managed Terminal (macOS only)

/// A managed terminal process
actor ManagedTerminal {
    let id: String
    private var process: Process?
    private var output: String = ""
    private var exitCode: Int32?
    private var exitSignal: String?
    private var hasExited: Bool = false
    private var exitContinuations: [CheckedContinuation<(exitCode: Int32?, signal: String?), Never>] = []

    init(id: String) {
        self.id = id
    }

    func start(command: String, args: [String], cwd: String?, env: [String: String]?, sshConfig: SSHConfiguration?) throws {
        let process = Process()

        if let ssh = sshConfig {
            // Run command via SSH on remote server
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")

            var sshArgs = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]

            if let keyPath = ssh.keyPath, !keyPath.isEmpty {
                sshArgs.append(contentsOf: ["-i", (keyPath as NSString).expandingTildeInPath])
            }

            sshArgs.append("\(ssh.username)@\(ssh.host)")

            // Build remote command with cd and env vars
            var remoteCommand = ""

            // Use cwd from params, fall back to SSH config's remote directory
            let workDir = cwd ?? ssh.remoteWorkingDirectory
            if let dir = workDir {
                remoteCommand += "cd \(dir) && "
            }

            // Add environment variables
            if let env = env {
                for (key, value) in env {
                    // Escape the value for shell
                    let escapedValue = value.replacingOccurrences(of: "'", with: "'\"'\"'")
                    remoteCommand += "\(key)='\(escapedValue)' "
                }
            }

            remoteCommand += command
            sshArgs.append(remoteCommand)

            process.arguments = sshArgs
            acpLog("TerminalManager: SSH command: ssh \(sshArgs.dropFirst(4).joined(separator: " "))")
        } else {
            // Run command locally
            process.executableURL = URL(fileURLWithPath: "/bin/sh")
            process.arguments = ["-c", command]

            if let cwd = cwd {
                process.currentDirectoryURL = URL(fileURLWithPath: cwd)
            }

            if let env = env {
                var processEnv = ProcessInfo.processInfo.environment
                for (key, value) in env {
                    processEnv[key] = value
                }
                process.environment = processEnv
            }
        }

        let outputPipe = Pipe()
        let errorPipe = Pipe()

        process.standardOutput = outputPipe
        process.standardError = errorPipe

        // Capture output
        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                Task { [weak self] in
                    await self?.appendOutput(str)
                }
            }
        }

        errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                Task { [weak self] in
                    await self?.appendOutput(str)
                }
            }
        }

        process.terminationHandler = { [weak self] proc in
            Task { [weak self] in
                await self?.handleTermination(exitCode: proc.terminationStatus)
            }
        }

        self.process = process
        try process.run()
    }

    private func appendOutput(_ str: String) {
        output += str
    }

    private func handleTermination(exitCode: Int32) {
        self.exitCode = exitCode
        self.hasExited = true

        // Resume all waiting continuations
        for continuation in exitContinuations {
            continuation.resume(returning: (exitCode: exitCode, signal: exitSignal))
        }
        exitContinuations.removeAll()
    }

    func getOutput() -> String {
        return output
    }

    func getExitStatus() -> (exitCode: Int32?, signal: String?, exited: Bool) {
        return (exitCode, exitSignal, hasExited)
    }

    func waitForExit() async -> (exitCode: Int32?, signal: String?) {
        if hasExited {
            return (exitCode, exitSignal)
        }

        return await withCheckedContinuation { continuation in
            exitContinuations.append(continuation)
        }
    }

    func kill() {
        if !hasExited, let process = process, process.isRunning {
            process.terminate()
        }
    }
}
#endif

// MARK: - Terminal Manager

/// Manages terminal processes for ACP
public actor TerminalManager {
    #if os(macOS)
    private var terminals: [String: ManagedTerminal] = [:]
    #endif
    private var idCounter: Int = 0
    private var sshConfig: SSHConfiguration?

    public init() {}

    /// Configure SSH for remote terminal execution
    public func setSSHConfiguration(_ config: SSHConfiguration?) {
        self.sshConfig = config
        if let config = config {
            acpLog("TerminalManager: SSH configured for \(config.username)@\(config.host)")
        } else {
            acpLog("TerminalManager: SSH disabled, using local execution")
        }
    }

    /// Check if SSH is configured
    public var isSSHEnabled: Bool {
        sshConfig != nil
    }

    /// Create a new terminal with the given command
    public func createTerminal(
        command: String,
        args: [String]? = nil,
        cwd: String? = nil,
        env: [String: String]? = nil
    ) async throws -> String {
        #if os(macOS)
        idCounter += 1
        let id = "terminal-\(idCounter)"

        let terminal = ManagedTerminal(id: id)
        try await terminal.start(command: command, args: args ?? [], cwd: cwd, env: env, sshConfig: sshConfig)

        terminals[id] = terminal
        let mode = sshConfig != nil ? "SSH" : "local"
        acpLog("TerminalManager: created terminal \(id) for command: \(command) [\(mode)]")

        return id
        #else
        throw NSError(domain: "ACPLib", code: 1, userInfo: [NSLocalizedDescriptionKey: "Terminal execution not supported on iOS"])
        #endif
    }

    /// Get the output from a terminal
    public func getOutput(terminalId: String) async -> (output: String, exitStatus: (exitCode: Int32?, signal: String?)?) {
        #if os(macOS)
        guard let terminal = terminals[terminalId] else {
            return ("", nil)
        }

        let output = await terminal.getOutput()
        let status = await terminal.getExitStatus()

        if status.exited {
            return (output, (status.exitCode, status.signal))
        } else {
            return (output, nil)
        }
        #else
        return ("", nil)
        #endif
    }

    /// Wait for a terminal to exit
    public func waitForExit(terminalId: String) async -> (exitCode: Int32?, signal: String?) {
        #if os(macOS)
        guard let terminal = terminals[terminalId] else {
            return (nil, nil)
        }

        return await terminal.waitForExit()
        #else
        return (nil, nil)
        #endif
    }

    /// Kill a terminal
    public func killTerminal(terminalId: String) async {
        #if os(macOS)
        guard let terminal = terminals[terminalId] else { return }
        await terminal.kill()
        #endif
    }

    /// Release (remove) a terminal
    public func releaseTerminal(terminalId: String) async {
        #if os(macOS)
        if let terminal = terminals.removeValue(forKey: terminalId) {
            await terminal.kill()
        }
        #endif
    }

    /// Clear all terminals
    public func clearAll() async {
        #if os(macOS)
        for (_, terminal) in terminals {
            await terminal.kill()
        }
        terminals.removeAll()
        #endif
    }
}
