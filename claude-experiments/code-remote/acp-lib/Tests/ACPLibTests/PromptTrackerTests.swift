import XCTest
@testable import ACPLib

final class PromptTrackerTests: XCTestCase {

    func testTextAccumulation() async {
        let tracker = PromptTracker()

        await tracker.appendText("Hello ")
        await tracker.appendText("World")

        let text = await tracker.currentText
        XCTAssertEqual(text, "Hello World")
    }

    func testHasContentWithText() async {
        let tracker = PromptTracker()

        let hasContent1 = await tracker.hasContent
        XCTAssertFalse(hasContent1)

        await tracker.appendText("test")
        let hasContent2 = await tracker.hasContent
        XCTAssertTrue(hasContent2)
    }

    func testHasContentWithToolCalls() async {
        let tracker = PromptTracker()

        let hasContent1 = await tracker.hasContent
        XCTAssertFalse(hasContent1)

        await tracker.startToolCall(id: "tool-1", name: "Bash", input: "ls")
        let hasContent2 = await tracker.hasContent
        XCTAssertTrue(hasContent2)
    }

    func testHasContentEmpty() async {
        let tracker = PromptTracker()
        let hasContent = await tracker.hasContent
        XCTAssertFalse(hasContent)
    }

    func testToolCallTracking() async {
        let tracker = PromptTracker()

        await tracker.startToolCall(id: "tool-1", name: "Bash", input: "ls -la")

        let toolCalls = await tracker.toolCalls
        XCTAssertEqual(toolCalls.count, 1)
        XCTAssertEqual(toolCalls["tool-1"]?.name, "Bash")
        XCTAssertEqual(toolCalls["tool-1"]?.status, .running)
    }

    func testToolCallCompletion() async {
        let tracker = PromptTracker()

        await tracker.startToolCall(id: "tool-1", name: "Bash", input: "ls")
        await tracker.completeToolCall(id: "tool-1", status: .complete, output: "file.txt", error: nil)

        let toolCalls = await tracker.toolCalls
        XCTAssertEqual(toolCalls["tool-1"]?.status, .complete)
        XCTAssertEqual(toolCalls["tool-1"]?.output, "file.txt")
    }

    func testCaptureInterruptedState() async {
        let tracker = PromptTracker()

        await tracker.appendText("Partial response")
        await tracker.startToolCall(id: "tool-1", name: "Bash", input: "ls")

        let state = await tracker.captureInterruptedState()

        XCTAssertEqual(state.text, "Partial response")
        XCTAssertTrue(state.formattedText.contains("[Interrupted]"))
        XCTAssertEqual(state.toolCalls.count, 1)

        // Verify state was cleared
        let hasContent = await tracker.hasContent
        let currentText = await tracker.currentText
        XCTAssertFalse(hasContent)
        XCTAssertEqual(currentText, "")
    }

    func testInterruptedTextFormatting() async {
        let tracker = PromptTracker()

        await tracker.appendText("Response text")
        let state = await tracker.captureInterruptedState()

        XCTAssertEqual(state.formattedText, "Response text\n\n[Interrupted]")
    }

    func testInterruptedTextFormattingWithTrailingNewline() async {
        let tracker = PromptTracker()

        await tracker.appendText("Response text\n")
        let state = await tracker.captureInterruptedState()

        XCTAssertEqual(state.formattedText, "Response text\n\n[Interrupted]")
    }

    func testInterruptedTextFormattingEmpty() async {
        let tracker = PromptTracker()

        let state = await tracker.captureInterruptedState()
        XCTAssertEqual(state.formattedText, "[Interrupted]")
    }

    func testClearState() async {
        let tracker = PromptTracker()

        await tracker.setPromptId("test-id")
        await tracker.appendText("test")
        await tracker.startToolCall(id: "tool-1", name: "Bash", input: nil)

        await tracker.clearState()

        let promptId = await tracker.promptId
        let currentText = await tracker.currentText
        let toolCalls = await tracker.toolCalls
        let hasContent = await tracker.hasContent

        XCTAssertNil(promptId)
        XCTAssertEqual(currentText, "")
        XCTAssertTrue(toolCalls.isEmpty)
        XCTAssertFalse(hasContent)
    }

    func testPromptIdTracking() async {
        let tracker = PromptTracker()

        let promptId1 = await tracker.promptId
        XCTAssertNil(promptId1)

        await tracker.setPromptId("test-123")
        let promptId2 = await tracker.promptId
        XCTAssertEqual(promptId2, "test-123")
    }
}
