import Foundation
import Citadel
import NIO
import NIOSSH
import Crypto
import _CryptoExtras

// MARK: - SSH Errors

enum SSHError: Error, LocalizedError {
    case connectionFailed(String)
    case authenticationFailed(String)
    case channelOpenFailed(String)
    case commandFailed(String)
    case notConnected
    case timeout
    case streamClosed
    case unsupportedKeyType

    var errorDescription: String? {
        switch self {
        case .connectionFailed(let message):
            return "Connection failed: \(message)"
        case .authenticationFailed(let message):
            return "Authentication failed: \(message)"
        case .channelOpenFailed(let message):
            return "Failed to open channel: \(message)"
        case .commandFailed(let message):
            return "Command failed: \(message)"
        case .notConnected:
            return "Not connected to server"
        case .timeout:
            return "Connection timed out"
        case .streamClosed:
            return "Stream closed unexpectedly"
        case .unsupportedKeyType:
            return "Unsupported SSH key type"
        }
    }
}

// MARK: - Remote File Item

struct RemoteFileItem: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let path: String
    let isDirectory: Bool
}

// MARK: - SSH Service

actor SSHService {
    private var client: SSHClient?
    private var currentServer: Server?

    var isConnected: Bool {
        client != nil
    }

    func connect(to server: Server, password: String?) async throws {
        await disconnect()

        // Handle case where host contains user@host format
        let actualHost: String
        let actualUsername: String
        if server.host.contains("@") {
            let parts = server.host.split(separator: "@", maxSplits: 1)
            actualUsername = String(parts[0])
            actualHost = String(parts[1])
        } else {
            actualHost = server.host
            actualUsername = server.username
        }

        do {
            print("[SSHService] Connecting to \(actualUsername)@\(actualHost):\(server.port)")
            print("[SSHService] Auth method: \(server.authMethod)")
            let authMethod: SSHAuthenticationMethod

            switch server.authMethod {
            case .password:
                // Try provided password first, then fall back to Keychain
                let actualPassword: String
                if let pwd = password, !pwd.isEmpty {
                    actualPassword = pwd
                } else if let keychainPwd = try? await KeychainService.shared.getPassword(for: server.id), !keychainPwd.isEmpty {
                    print("[SSHService] Using password from Keychain")
                    actualPassword = keychainPwd
                } else {
                    print("[SSHService] ERROR: Password required but not provided")
                    throw SSHError.authenticationFailed("Password required - please add password in server settings")
                }
                authMethod = .passwordBased(username: actualUsername, password: actualPassword)

            case .privateKey:
                // Auto-detect SSH key if no path specified
                let keyPath: String
                if let specifiedPath = server.privateKeyPath, !specifiedPath.isEmpty {
                    keyPath = specifiedPath
                } else {
                    #if os(macOS)
                    // Try common SSH key locations
                    let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
                    let commonKeys = [
                        "\(homeDir)/.ssh/id_ed25519",
                        "\(homeDir)/.ssh/id_rsa",
                        "\(homeDir)/.ssh/id_ecdsa"
                    ]
                    guard let foundKey = commonKeys.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
                        throw SSHError.authenticationFailed("No SSH key found. Checked: ~/.ssh/id_ed25519, ~/.ssh/id_rsa, ~/.ssh/id_ecdsa")
                    }
                    keyPath = foundKey
                    #else
                    throw SSHError.authenticationFailed("SSH key path must be specified on iOS")
                    #endif
                }
                print("[SSHService] Using SSH key: \(keyPath)")

                let expandedPath = (keyPath as NSString).expandingTildeInPath
                let keyURL = URL(fileURLWithPath: expandedPath)
                let keyData = try Data(contentsOf: keyURL)
                guard let keyString = String(data: keyData, encoding: .utf8) else {
                    throw SSHError.authenticationFailed("Could not read private key")
                }

                // Detect key type and use appropriate authentication method
                let keyType = try SSHKeyDetection.detectPrivateKeyType(from: keyString)

                if keyType == .rsa {
                    let rsaKey = try Insecure.RSA.PrivateKey(sshRsa: keyString)
                    authMethod = .rsa(username: actualUsername, privateKey: rsaKey)
                } else if keyType == .ed25519 {
                    let ed25519Key = try Curve25519.Signing.PrivateKey(sshEd25519: keyString)
                    authMethod = .ed25519(username: actualUsername, privateKey: ed25519Key)
                } else {
                    // ECDSA keys in OpenSSH format not directly supported - would require conversion
                    throw SSHError.authenticationFailed("Key type \(keyType) is not yet supported. Please use RSA or ED25519.")
                }
            }

            print("[SSHService] Calling SSHClient.connect...")
            client = try await SSHClient.connect(
                host: actualHost,
                port: server.port,
                authenticationMethod: authMethod,
                hostKeyValidator: .acceptAnything(),
                reconnect: .never
            )
            print("[SSHService] SSHClient.connect completed")

            currentServer = server

        } catch let error as SSHError {
            throw error
        } catch {
            let errorStr = String(describing: error)
            if errorStr.contains("NIOConnectionError") {
                throw SSHError.connectionFailed("Cannot reach \(actualHost):\(server.port) - check host and network")
            }
            throw SSHError.connectionFailed("Failed to connect to \(actualHost):\(server.port) - \(error)")
        }
    }

    func disconnect() async {
        if let client = client {
            try? await client.close()
        }
        client = nil
        currentServer = nil
    }

    func executeCommand(_ command: String) async throws -> String {
        guard let client = client else {
            throw SSHError.notConnected
        }

        let output = try await client.executeCommand(command)
        return String(buffer: output)
    }

    func listDirectory(_ path: String) async throws -> [RemoteFileItem] {
        guard client != nil else {
            throw SSHError.notConnected
        }

        // Use ls with specific format: type, permissions, name
        // -1 = one per line, -a = all including hidden, -F = append indicator (/ for dirs)
        let command = "ls -1aF \(path.replacingOccurrences(of: "'", with: "'\\''"))"
        let output = try await executeCommand(command)

        var items: [RemoteFileItem] = []
        let lines = output.split(separator: "\n", omittingEmptySubsequences: true)

        for line in lines {
            let name = String(line)
            // Skip . and ..
            if name == "./" || name == "../" || name == "." || name == ".." {
                continue
            }

            let isDirectory = name.hasSuffix("/")
            let cleanName = isDirectory ? String(name.dropLast()) : name

            items.append(RemoteFileItem(
                name: cleanName,
                path: (path as NSString).appendingPathComponent(cleanName),
                isDirectory: isDirectory
            ))
        }

        // Sort: directories first, then alphabetically
        items.sort { lhs, rhs in
            if lhs.isDirectory != rhs.isDirectory {
                return lhs.isDirectory
            }
            return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
        }

        return items
    }

    func getHomeDirectory() async throws -> String {
        guard client != nil else {
            throw SSHError.notConnected
        }
        let output = try await executeCommand("echo $HOME")
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func executeClaudePrompt(
        projectPath: String,
        prompt: String,
        resumeSessionId: String? = nil
    ) async throws -> AsyncThrowingStream<String, Error> {
        guard let client = client else {
            throw SSHError.notConnected
        }

        // Escape the prompt for shell
        let escapedPrompt = prompt
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "$", with: "\\$")
            .replacingOccurrences(of: "`", with: "\\`")

        // Build the claude command with stdin from /dev/null (critical - claude waits for stdin to close)
        var claudeCmd = "/usr/local/bin/claude -p \"\(escapedPrompt)\" --output-format stream-json --verbose"
        if let sessionId = resumeSessionId {
            claudeCmd += " --resume \(sessionId)"
        }

        // Redirect stdin from /dev/null so claude doesn't wait for input
        let command = "cd \(projectPath) && \(claudeCmd) < /dev/null"

        print("[SSHService] Executing: \(command)")

        // Use streaming to get real-time output
        let stream = try await client.executeCommandStream(command)

        return AsyncThrowingStream { continuation in
            Task {
                var buffer = Data()

                do {
                    for try await chunk in stream {
                        switch chunk {
                        case .stdout(var stdoutBuffer):
                            if let bytes = stdoutBuffer.readBytes(length: stdoutBuffer.readableBytes) {
                                buffer.append(contentsOf: bytes)
                            }
                        case .stderr:
                            // Ignore stderr - claude's main output is on stdout
                            break
                        }

                        // Process complete lines (newline-delimited JSON)
                        while let newlineIndex = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                            let lineData = buffer.prefix(upTo: newlineIndex)
                            buffer = Data(buffer.suffix(from: buffer.index(after: newlineIndex)))

                            if let line = String(data: Data(lineData), encoding: .utf8), !line.isEmpty {
                                continuation.yield(line)
                            }
                        }
                    }

                    // Handle any remaining data
                    if !buffer.isEmpty, let remaining = String(data: buffer, encoding: .utf8), !remaining.isEmpty {
                        continuation.yield(remaining)
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
