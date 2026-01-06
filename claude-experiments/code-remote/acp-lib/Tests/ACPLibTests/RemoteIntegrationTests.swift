import XCTest
@testable import ACPLib

/// Integration tests that connect to a real remote claude-code-acp instance
/// These tests require SSH access to the remote server
final class RemoteIntegrationTests: XCTestCase {

    // Remote server configuration - adjust as needed
    let sshHost = "computer.jimmyhmiller.com"
    let sshUsername = "jimmyhmiller"
    let workingDirectory = "/home/jimmyhmiller/Documents/Code/beagle"

    var client: ACPClient!
    var delegate: TestPermissionDelegate!
    var receivedEvents: [ACPEvent] = []

    override func setUp() async throws {
        client = ACPClient.forClaudeCode(name: "integration-test", version: "1.0.0")
        delegate = TestPermissionDelegate()
        await client.setDelegate(delegate)
        receivedEvents = []
    }

    override func tearDown() async throws {
        if let client = client {
            await client.disconnect()
        }
    }

    // MARK: - Connection Tests

    func testRemoteConnection() async throws {
        // Build SSH command
        let sshArgs = [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "\(sshUsername)@\(sshHost)",
            "cd \(workingDirectory) && claude-code-acp"
        ]

        try await client.connect(
            command: "/usr/bin/ssh",
            arguments: sshArgs
        )

        let isConnected = await client.isConnected
        let agentInfo = await client.agentInfo

        XCTAssertTrue(isConnected, "Should be connected")
        XCTAssertNotNil(agentInfo, "Should have agent info")
        XCTAssertEqual(agentInfo?.title, "Claude Code")
    }

    func testRemoteSessionCreation() async throws {
        try await connectRemote()

        let sessionId = try await client.newSession(cwd: workingDirectory)
        let currentSessionId = await client.currentSessionId

        XCTAssertFalse(sessionId.isEmpty, "Session ID should not be empty")
        XCTAssertEqual(sessionId, currentSessionId)

        // Check modes are available
        let modes = await client.availableModes
        XCTAssertFalse(modes.isEmpty, "Should have available modes")
    }

    // MARK: - Permission Tests

    func testPermissionRequestHandling() async throws {
        try await connectRemote()
        _ = try await client.newSession(cwd: workingDirectory)

        // Start event listener
        let eventTask = Task {
            for await event in await client.events {
                self.receivedEvents.append(event)
            }
        }

        // Send a prompt that will trigger a tool call requiring permission
        await delegate.setAutoGrant(true)

        let result = try await client.prompt(text: "Run: echo hello")

        eventTask.cancel()

        // Verify permission was requested and granted
        let requestCount = await delegate.getPermissionRequestCount()
        XCTAssertTrue(requestCount > 0, "Should have received permission requests")
        XCTAssertEqual(result.stopReason, "end_turn", "Prompt should complete")
    }

    func testPermissionDenial() async throws {
        try await connectRemote()
        _ = try await client.newSession(cwd: workingDirectory)

        // Start event listener
        let eventTask = Task {
            for await event in await client.events {
                self.receivedEvents.append(event)
            }
        }

        // Deny permissions
        await delegate.setAutoGrant(false)

        let result = try await client.prompt(text: "Run: echo hello")

        eventTask.cancel()

        // Verify permission was denied
        let requestCount = await delegate.getPermissionRequestCount()
        XCTAssertTrue(requestCount > 0, "Should have received permission requests")
        // The prompt should still complete, but the tool should have been rejected
        XCTAssertNotNil(result.stopReason)
    }

    // MARK: - Tool Call Tests

    func testToolCallWithOutput() async throws {
        try await connectRemote()
        _ = try await client.newSession(cwd: workingDirectory)

        var toolCallUpdates: [(id: String, status: String, output: String?, error: String?)] = []
        var textChunks: [String] = []

        // Start event listener
        let eventTask = Task {
            for await event in await client.events {
                switch event {
                case .toolCallUpdate(let id, let status, let output, let error, _):
                    toolCallUpdates.append((id, status, output, error))
                    print("[EVENT] toolCallUpdate: id=\(id), status=\(status), output=\(output ?? "nil"), error=\(error ?? "nil")")
                case .textChunk(let text, _):
                    textChunks.append(text)
                    print("[EVENT] textChunk: \(text)")
                default:
                    break
                }
            }
        }

        await delegate.setAutoGrant(true)

        let result = try await client.prompt(text: "Run this exact command: echo 'test123'")

        // Give events time to process
        try await Task.sleep(nanoseconds: 500_000_000)
        eventTask.cancel()

        print("\n=== RESULTS ===")
        print("stopReason: \(result.stopReason)")
        print("toolCallUpdates count: \(toolCallUpdates.count)")
        for update in toolCallUpdates {
            print("  - id: \(update.id)")
            print("    status: \(update.status)")
            print("    output: \(update.output ?? "nil")")
            print("    error: \(update.error ?? "nil")")
        }
        print("textChunks: \(textChunks.joined())")
        print("===============\n")

        XCTAssertEqual(result.stopReason, "end_turn")
    }

    // MARK: - Helpers

    private func connectRemote() async throws {
        let sshArgs = [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "\(sshUsername)@\(sshHost)",
            "cd \(workingDirectory) && claude-code-acp"
        ]

        try await client.connect(
            command: "/usr/bin/ssh",
            arguments: sshArgs
        )
    }
}

// MARK: - Test Permission Delegate

actor TestPermissionDelegate: ACPClientDelegate {
    private var _autoGrant = true
    private var _permissionRequests: [(toolName: String, input: String?, prompt: String?)] = []

    func setAutoGrant(_ value: Bool) {
        _autoGrant = value
    }

    func getPermissionRequestCount() -> Int {
        _permissionRequests.count
    }

    func acpClient(_ client: ACPClient, didReceive event: ACPEvent) async {
        // Just record events
    }

    func acpClient(_ client: ACPClient, requestPermissionFor toolName: String, input: String?, prompt: String?) async -> (granted: Bool, context: String?) {
        _permissionRequests.append((toolName, input, prompt))
        print("[TestDelegate] Permission request: tool=\(toolName), autoGrant=\(_autoGrant)")
        return (_autoGrant, nil)
    }
}
