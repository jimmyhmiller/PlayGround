import Foundation
import Citadel
import NIOSSH
import Crypto

func log(_ msg: String) {
    fputs(msg + "\n", stderr)
    fflush(stderr)
}

@main
struct SSHTest {
    static func main() async {
        log("SSH Test Starting...")

        let host = "computer.jimmyhmiller.com"
        let username = "jimmyhmiller"
        let keyPath = FileManager.default.homeDirectoryForCurrentUser.path + "/.ssh/id_ed25519"

        log("Host: \(username)@\(host)")
        log("Key: \(keyPath)")

        do {
            // Read key
            let keyData = try Data(contentsOf: URL(fileURLWithPath: keyPath))
            guard let keyString = String(data: keyData, encoding: .utf8) else {
                log("ERROR: Could not read key as UTF-8")
                return
            }
            log("Key loaded, length: \(keyString.count)")

            // Parse key
            let ed25519Key = try Curve25519.Signing.PrivateKey(sshEd25519: keyString)
            log("Key parsed successfully")

            let authMethod = SSHAuthenticationMethod.ed25519(username: username, privateKey: ed25519Key)

            // Connect
            log("Connecting...")
            let client = try await SSHClient.connect(
                host: host,
                port: 22,
                authenticationMethod: authMethod,
                hostKeyValidator: .acceptAnything(),
                reconnect: .never
            )
            log("Connected!")

            // Test simple command
            log("\n--- Test 1: Simple echo ---")
            let simpleOutput = try await client.executeCommand("echo 'hello world'")
            log("Output: \(String(buffer: simpleOutput))")

            // Test command that takes a second
            log("\n--- Test 2: Sleep + echo ---")
            let sleepOutput = try await client.executeCommand("sleep 1 && echo 'done sleeping'")
            log("Output: \(String(buffer: sleepOutput))")

            // Test claude --version
            log("\n--- Test 3: claude --version ---")
            let versionOutput = try await client.executeCommand("/usr/local/bin/claude --version")
            log("Output: \(String(buffer: versionOutput))")

            // Test claude command with stdin closed
            log("\n--- Test 4: claude command (with stdin from /dev/null) ---")
            // Pipe stdin from /dev/null to ensure the command doesn't wait for input
            let claudeCommand = "/usr/local/bin/claude -p \"Say just the word hello\" --output-format stream-json --verbose < /dev/null"
            log("Command: \(claudeCommand)")
            log("Executing...")

            let stream = try await client.executeCommandStream(claudeCommand)
            log("Got stream, reading...")

            var totalBytes = 0
            for try await chunk in stream {
                switch chunk {
                case .stdout(let buffer):
                    totalBytes += buffer.readableBytes
                    log("stdout: \(buffer.readableBytes) bytes (total: \(totalBytes))")
                    let text = String(buffer: buffer)
                    log("  > \(text.prefix(200))")
                case .stderr(let buffer):
                    log("stderr: \(buffer.readableBytes) bytes")
                    let text = String(buffer: buffer)
                    log("  > \(text.prefix(200))")
                }
            }

            log("Stream complete, total: \(totalBytes) bytes")

            try await client.close()
            log("\nDisconnected. Done!")

        } catch {
            log("ERROR: \(error)")
        }
    }
}
