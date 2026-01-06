import XCTest
@testable import ACPLib

final class MockConnectionTests: XCTestCase {

    func testMockConnectionNotClosed() async {
        let connection = ACPMockConnection()
        let isClosed = await connection.isClosed
        XCTAssertFalse(isClosed)
    }

    func testMockConnectionClose() async {
        let connection = ACPMockConnection()
        await connection.close()
        let isClosed = await connection.isClosed
        XCTAssertTrue(isClosed)
    }

    func testMockConnectionSetResponse() async throws {
        let connection = ACPMockConnection()

        let expectedResult = ACPSessionNewResult(sessionId: "test-123", modes: nil)
        await connection.setMockResponse(for: "session/new", response: expectedResult)

        let result: ACPSessionNewResult = try await connection.sendRequest(
            method: "session/new",
            params: ACPSessionNewParams(cwd: "/test")
        )

        XCTAssertEqual(result.sessionId, "test-123")
    }

    func testMockConnectionNoResponse() async {
        let connection = ACPMockConnection()

        do {
            let _: ACPSessionNewResult = try await connection.sendRequest(
                method: "session/new",
                params: ACPSessionNewParams(cwd: "/test")
            )
            XCTFail("Should have thrown")
        } catch {
            // Expected
            XCTAssertTrue(error is ACPConnectionError)
        }
    }

    func testMockConnectionNotConnected() async {
        let connection = ACPMockConnection()
        await connection.close()

        do {
            let _: ACPSessionNewResult = try await connection.sendRequest(
                method: "session/new",
                params: nil as String?
            )
            XCTFail("Should have thrown")
        } catch ACPConnectionError.notConnected {
            // Expected
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    func testMockConnectionNotificationHandler() async {
        let connection = ACPMockConnection()

        let expectation = XCTestExpectation(description: "Notification received")
        var receivedMethod: String?

        await connection.setNotificationHandler { notification in
            receivedMethod = notification.method
            expectation.fulfill()
        }

        let notification = JSONRPCNotification(method: "test", params: nil as String?)
        await connection.simulateNotification(notification)

        await fulfillment(of: [expectation], timeout: 1.0)
        XCTAssertEqual(receivedMethod, "test")
    }

    func testMockConnectionSimulateSessionUpdate() async {
        let connection = ACPMockConnection()

        let expectation = XCTestExpectation(description: "Update received")
        var receivedMethod: String?

        await connection.setNotificationHandler { notification in
            receivedMethod = notification.method
            expectation.fulfill()
        }

        let update = ACPSessionUpdate.agentMessageChunk(
            ACPAgentMessageChunk(content: .text(ACPAgentTextContent(text: "Hello")))
        )
        await connection.simulateSessionUpdate(sessionId: "test-123", update: update)

        await fulfillment(of: [expectation], timeout: 1.0)
        XCTAssertEqual(receivedMethod, "session/update")
    }

    // MARK: - Integration Tests

    func testClientWithMockConnection() async throws {
        let connection = ACPMockConnection()

        // Set up mock responses
        await connection.setMockResponse(
            for: "initialize",
            response: ACPInitializeResult(
                protocolVersion: 1,
                agentCapabilities: nil,
                agentInfo: ACPAgentInfo(name: "test", title: "Test Agent", version: "1.0"),
                authMethods: nil
            )
        )

        await connection.setMockResponse(
            for: "session/new",
            response: ACPSessionNewResult(
                sessionId: "session-123",
                modes: ACPModeInfo(
                    availableModes: [
                        ACPMode(id: "agent", name: "Agent"),
                        ACPMode(id: "plan", name: "Plan")
                    ],
                    currentModeId: "agent"
                )
            )
        )

        // Create and connect client
        let client = ACPClient.forClaudeCode()
        try await client.connect(using: connection)

        let isConnected = await client.isConnected
        let agentInfo = await client.agentInfo
        XCTAssertTrue(isConnected)
        XCTAssertEqual(agentInfo?.name, "test")

        // Create session
        let sessionId = try await client.newSession(cwd: "/test")
        let currentSessionId = await client.currentSessionId
        XCTAssertEqual(sessionId, "session-123")
        XCTAssertEqual(currentSessionId, "session-123")

        // Check modes
        let modes = await client.availableModes
        let currentMode = await client.currentMode
        XCTAssertEqual(modes.count, 2)
        XCTAssertEqual(currentMode?.name, "Agent")
    }

    func testClientDisconnect() async throws {
        let connection = ACPMockConnection()

        await connection.setMockResponse(
            for: "initialize",
            response: ACPInitializeResult(
                protocolVersion: 1,
                agentCapabilities: nil,
                agentInfo: ACPAgentInfo(name: "test", title: "Test", version: "1.0"),
                authMethods: nil
            )
        )

        let client = ACPClient.forClaudeCode()
        try await client.connect(using: connection)

        let isConnected1 = await client.isConnected
        XCTAssertTrue(isConnected1)

        await client.disconnect()

        let isConnected2 = await client.isConnected
        let agentInfo = await client.agentInfo
        let currentSessionId = await client.currentSessionId
        XCTAssertFalse(isConnected2)
        XCTAssertNil(agentInfo)
        XCTAssertNil(currentSessionId)
    }
}
