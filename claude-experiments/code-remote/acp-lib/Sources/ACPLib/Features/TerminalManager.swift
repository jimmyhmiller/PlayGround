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

// MARK: - Remote Command Executor Protocol

/// Protocol for executing commands on a remote server (used on iOS)
public protocol RemoteCommandExecutor: Actor {
    /// Execute a command and return the output
    func executeCommand(_ command: String) async throws -> String
}

// MARK: - Remote Terminal (iOS)

#if !os(macOS)
/// A managed terminal that runs via SSH on iOS
actor RemoteTerminal {
    let id: String
    private weak var executor: (any RemoteCommandExecutor)?
    private var output: String = ""
    private var exitCode: Int32?
    private var hasExited: Bool = false
    private var command: String = ""
    private var cwd: String?
    private var runTask: Task<Void, Never>?

    init(id: String, executor: any RemoteCommandExecutor) {
        self.id = id
        self.executor = executor
    }

    func start(command: String, cwd: String?, env: [String: String]?) async throws {
        self.command = command
        self.cwd = cwd

        // Build full command with cd and env
        var fullCommand = ""
        if let cwd = cwd {
            fullCommand += "cd '\(cwd)' && "
        }
        if let env = env {
            for (key, value) in env {
                let escapedValue = value.replacingOccurrences(of: "'", with: "'\"'\"'")
                fullCommand += "\(key)='\(escapedValue)' "
            }
        }
        fullCommand += command

        // Run command in background
        runTask = Task { [weak self] in
            guard let self = self, let executor = await self.executor else { return }

            do {
                let result = try await executor.executeCommand(fullCommand)
                await self.setOutput(result)
                await self.setExitCode(0)
            } catch {
                await self.appendOutput("Error: \(error.localizedDescription)")
                await self.setExitCode(1)
            }
        }
    }

    private func setOutput(_ str: String) {
        output = str
    }

    private func appendOutput(_ str: String) {
        output += str
    }

    private func setExitCode(_ code: Int32) {
        exitCode = code
        hasExited = true
    }

    func getOutput() -> String {
        return output
    }

    func getExitStatus() -> (exitCode: Int32?, signal: String?, exited: Bool) {
        return (exitCode, nil, hasExited)
    }

    func waitForExit() async -> (exitCode: Int32?, signal: String?) {
        // Wait for the task to complete
        await runTask?.value
        return (exitCode, nil)
    }

    func kill() {
        runTask?.cancel()
        hasExited = true
    }
}
#endif

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
    #else
    private var terminals: [String: RemoteTerminal] = [:]
    private var remoteExecutor: (any RemoteCommandExecutor)?
    #endif
    private var idCounter: Int = 0
    private var sshConfig: SSHConfiguration?

    public init() {}

    /// Configure SSH for remote terminal execution (macOS)
    public func setSSHConfiguration(_ config: SSHConfiguration?) {
        self.sshConfig = config
        if let config = config {
            acpLog("TerminalManager: SSH configured for \(config.username)@\(config.host)")
        } else {
            acpLog("TerminalManager: SSH disabled, using local execution")
        }
    }

    #if !os(macOS)
    /// Set the remote command executor for iOS terminal execution
    public func setRemoteExecutor(_ executor: any RemoteCommandExecutor) {
        self.remoteExecutor = executor
        acpLog("TerminalManager: remote executor configured for iOS")
    }
    #endif

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
        idCounter += 1
        let id = "terminal-\(idCounter)"

        #if os(macOS)
        let terminal = ManagedTerminal(id: id)
        try await terminal.start(command: command, args: args ?? [], cwd: cwd, env: env, sshConfig: sshConfig)

        terminals[id] = terminal
        let mode = sshConfig != nil ? "SSH" : "local"
        acpLog("TerminalManager: created terminal \(id) for command: \(command) [\(mode)]")
        #else
        guard let executor = remoteExecutor else {
            throw NSError(domain: "ACPLib", code: 1, userInfo: [NSLocalizedDescriptionKey: "No remote executor configured for iOS terminal"])
        }

        let terminal = RemoteTerminal(id: id, executor: executor)
        try await terminal.start(command: command, cwd: cwd, env: env)

        terminals[id] = terminal
        acpLog("TerminalManager: created remote terminal \(id) for command: \(command)")
        #endif

        return id
    }

    /// Get the output from a terminal
    public func getOutput(terminalId: String) async -> (output: String, exitStatus: (exitCode: Int32?, signal: String?)?) {
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
    }

    /// Wait for a terminal to exit
    public func waitForExit(terminalId: String) async -> (exitCode: Int32?, signal: String?) {
        guard let terminal = terminals[terminalId] else {
            return (nil, nil)
        }

        return await terminal.waitForExit()
    }

    /// Kill a terminal
    public func killTerminal(terminalId: String) async {
        guard let terminal = terminals[terminalId] else { return }
        await terminal.kill()
    }

    /// Release (remove) a terminal
    public func releaseTerminal(terminalId: String) async {
        if let terminal = terminals.removeValue(forKey: terminalId) {
            await terminal.kill()
        }
    }

    /// Clear all terminals
    public func clearAll() async {
        for (_, terminal) in terminals {
            await terminal.kill()
        }
        terminals.removeAll()
    }

    // MARK: - File Operations

    /// Check if remote executor is available
    public var hasRemoteExecutor: Bool {
        #if os(macOS)
        return sshConfig != nil
        #else
        return remoteExecutor != nil
        #endif
    }

    /// Read a file from the remote server via SSH
    public func readRemoteFile(_ path: String) async -> String? {
        #if os(macOS)
        guard let ssh = sshConfig else {
            acpLog("TerminalManager: readRemoteFile - no SSH config")
            return nil
        }

        do {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")

            var sshArgs = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
            if let keyPath = ssh.keyPath, !keyPath.isEmpty {
                sshArgs.append(contentsOf: ["-i", (keyPath as NSString).expandingTildeInPath])
            }
            sshArgs.append("\(ssh.username)@\(ssh.host)")
            sshArgs.append("cat '\(path)'")

            process.arguments = sshArgs

            let outputPipe = Pipe()
            process.standardOutput = outputPipe
            process.standardError = FileHandle.nullDevice

            try process.run()
            process.waitUntilExit()

            let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let content = String(data: data, encoding: .utf8)
            acpLog("TerminalManager: readRemoteFile - read \(data.count) bytes from \(path)")
            return content
        } catch {
            acpLog("TerminalManager: readRemoteFile error: \(error)")
            return nil
        }
        #else
        guard let executor = remoteExecutor else {
            acpLog("TerminalManager: readRemoteFile - no remote executor")
            return nil
        }

        do {
            let content = try await executor.executeCommand("cat '\(path)'")
            acpLog("TerminalManager: readRemoteFile - read \(content.count) bytes from \(path)")
            return content
        } catch {
            acpLog("TerminalManager: readRemoteFile error: \(error)")
            return nil
        }
        #endif
    }
}
