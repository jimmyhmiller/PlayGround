import XCTest
@testable import ACPLib

final class SessionHistoryTests: XCTestCase {

    func testPathEncoding() {
        XCTAssertEqual(PathEncoder.encode("/Users/test/project"), "-Users-test-project")
        XCTAssertEqual(PathEncoder.encode("/"), "-")
        XCTAssertEqual(PathEncoder.encode("no-slashes"), "no-slashes")
        XCTAssertEqual(PathEncoder.encode("/a/b/c/d"), "-a-b-c-d")
    }

    func testSessionFilePath() {
        let path = SessionHistoryLoader.sessionFilePath(
            sessionId: "abc123",
            cwd: "/Users/test/project"
        )

        XCTAssertTrue(path.path.contains(".claude/projects"))
        XCTAssertTrue(path.path.contains("-Users-test-project"))
        XCTAssertTrue(path.path.hasSuffix("abc123.jsonl"))
    }

    func testSessionFilePathWithDifferentCwd() {
        let path1 = SessionHistoryLoader.sessionFilePath(sessionId: "test", cwd: "/path/one")
        let path2 = SessionHistoryLoader.sessionFilePath(sessionId: "test", cwd: "/path/two")

        XCTAssertNotEqual(path1, path2)
        XCTAssertTrue(path1.path.contains("-path-one"))
        XCTAssertTrue(path2.path.contains("-path-two"))
    }

    // MARK: - JSONL Parsing Tests

    func testParseSessionFileEntry() throws {
        let jsonl = """
        {"type":"user","uuid":"msg-1","timestamp":"2024-01-01T00:00:00Z","message":{"content":[{"type":"text","text":"Hello"}]}}
        """

        let data = jsonl.data(using: .utf8)!
        let entry = try JSONDecoder().decode(SessionFileEntry.self, from: data)

        XCTAssertEqual(entry.type, "user")
        XCTAssertEqual(entry.uuid, "msg-1")
        XCTAssertNotNil(entry.message)
        XCTAssertEqual(entry.message?.content?.count, 1)
    }

    func testParseTextContentBlock() throws {
        let json = """
        {"type":"text","text":"Hello World"}
        """.data(using: .utf8)!

        let block = try JSONDecoder().decode(SessionContentBlock.self, from: json)

        if case .text(let text) = block {
            XCTAssertEqual(text, "Hello World")
        } else {
            XCTFail("Expected text block")
        }
    }

    func testParseToolUseContentBlock() throws {
        let json = """
        {"type":"tool_use","id":"tool-123","name":"Bash"}
        """.data(using: .utf8)!

        let block = try JSONDecoder().decode(SessionContentBlock.self, from: json)

        if case .toolUse(let id, let name) = block {
            XCTAssertEqual(id, "tool-123")
            XCTAssertEqual(name, "Bash")
        } else {
            XCTFail("Expected tool_use block")
        }
    }

    func testParseUnknownContentBlock() throws {
        let json = """
        {"type":"image","data":"base64..."}
        """.data(using: .utf8)!

        let block = try JSONDecoder().decode(SessionContentBlock.self, from: json)

        if case .other = block {
            // Success
        } else {
            XCTFail("Expected other block for unknown type")
        }
    }

    // MARK: - History Message Role Tests

    func testMessageRoleUser() {
        XCTAssertEqual(ACPMessageRole(rawValue: "user"), .user)
    }

    func testMessageRoleAssistant() {
        XCTAssertEqual(ACPMessageRole(rawValue: "assistant"), .assistant)
    }

    func testMessageRoleUnknown() {
        XCTAssertNil(ACPMessageRole(rawValue: "system"))
        XCTAssertNil(ACPMessageRole(rawValue: "tool"))
    }

    // MARK: - History Message Tests

    func testHistoryMessageCreation() {
        let message = ACPHistoryMessage(
            id: "msg-1",
            role: .user,
            content: "Hello, Claude!",
            timestamp: Date(),
            toolCalls: nil
        )

        XCTAssertEqual(message.id, "msg-1")
        XCTAssertEqual(message.role, .user)
        XCTAssertEqual(message.content, "Hello, Claude!")
        XCTAssertNil(message.toolCalls)
    }

    func testHistoryMessageWithToolCalls() {
        let toolCalls = [
            ACPHistoryToolCall(id: "tool-1", name: "Bash", input: "ls", output: "file.txt")
        ]

        let message = ACPHistoryMessage(
            id: "msg-1",
            role: .assistant,
            content: "Let me list the files...",
            timestamp: Date(),
            toolCalls: toolCalls
        )

        XCTAssertNotNil(message.toolCalls)
        XCTAssertEqual(message.toolCalls?.count, 1)
        XCTAssertEqual(message.toolCalls?.first?.name, "Bash")
    }

    // MARK: - Interrupted State Tests

    func testInterruptedPromptStateFormattedText() {
        let state = InterruptedPromptState(
            text: "Hello",
            toolCalls: [:],
            timestamp: Date()
        )

        XCTAssertEqual(state.formattedText, "Hello\n\n[Interrupted]")
    }

    func testInterruptedPromptStateEmptyText() {
        let state = InterruptedPromptState(
            text: "",
            toolCalls: [:],
            timestamp: Date()
        )

        XCTAssertEqual(state.formattedText, "[Interrupted]")
    }

    func testInterruptedPromptStateWhitespaceOnly() {
        let state = InterruptedPromptState(
            text: "   \n\n   ",
            toolCalls: [:],
            timestamp: Date()
        )

        XCTAssertEqual(state.formattedText, "[Interrupted]")
    }
}
