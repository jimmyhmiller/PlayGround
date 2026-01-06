import XCTest
@testable import ACPLib

final class MessageQueueTests: XCTestCase {

    func testGeneratePromptIdUniqueness() async {
        let queue = MessageQueue()

        let id1 = await queue.generatePromptId()
        let id2 = await queue.generatePromptId()

        XCTAssertNotEqual(id1, id2)
        XCTAssertTrue(id1.hasPrefix("prompt-"))
        XCTAssertTrue(id2.hasPrefix("prompt-"))
    }

    func testSetActivePrompt() async {
        let queue = MessageQueue()

        let activeId1 = await queue.activePromptId
        let isActive1 = await queue.isPromptActive
        XCTAssertNil(activeId1)
        XCTAssertFalse(isActive1)

        await queue.setActivePrompt("test-id")

        let activeId2 = await queue.activePromptId
        let isActive2 = await queue.isPromptActive
        XCTAssertEqual(activeId2, "test-id")
        XCTAssertTrue(isActive2)
    }

    func testClearActivePrompt() async {
        let queue = MessageQueue()

        await queue.setActivePrompt("test-id")
        let isActive1 = await queue.isPromptActive
        XCTAssertTrue(isActive1)

        await queue.clearActivePrompt()

        let activeId = await queue.activePromptId
        let isActive2 = await queue.isPromptActive
        XCTAssertNil(activeId)
        XCTAssertFalse(isActive2)
    }

    func testInvalidatePrompt() async {
        let queue = MessageQueue()

        await queue.setActivePrompt("test-id")

        // Invalidating a different prompt should return false
        let result1 = await queue.invalidatePrompt("other-id")
        XCTAssertFalse(result1)
        let activeId1 = await queue.activePromptId
        XCTAssertEqual(activeId1, "test-id")

        // Invalidating the active prompt should return true
        let result2 = await queue.invalidatePrompt("test-id")
        XCTAssertTrue(result2)
        let activeId2 = await queue.activePromptId
        XCTAssertNil(activeId2)
    }

    func testEventFilteringWithNoActivePrompt() async {
        let queue = MessageQueue()

        // No active prompt set - should not process events
        let shouldProcess = await queue.shouldProcessEvent(forPromptId: "some-id")
        XCTAssertFalse(shouldProcess)
    }

    func testEventFilteringWithMatchingPromptId() async {
        let queue = MessageQueue()

        let id = await queue.generatePromptId()
        await queue.setActivePrompt(id)

        let shouldProcess = await queue.shouldProcessEvent(forPromptId: id)
        XCTAssertTrue(shouldProcess)
    }

    func testEventFilteringWithMismatchedPromptId() async {
        let queue = MessageQueue()

        let id = await queue.generatePromptId()
        await queue.setActivePrompt(id)

        let shouldProcess = await queue.shouldProcessEvent(forPromptId: "different-id")
        XCTAssertFalse(shouldProcess)
    }

    func testEventFilteringWithNilPromptId() async {
        let queue = MessageQueue()

        let id = await queue.generatePromptId()
        await queue.setActivePrompt(id)

        // Events without prompt ID should be processed if there's an active prompt
        let shouldProcess = await queue.shouldProcessEvent(forPromptId: nil)
        XCTAssertTrue(shouldProcess)
    }

    func testIsActivePrompt() async {
        let queue = MessageQueue()

        let id = await queue.generatePromptId()
        await queue.setActivePrompt(id)

        let isActive = await queue.isActivePrompt(id)
        let isOtherActive = await queue.isActivePrompt("other-id")
        XCTAssertTrue(isActive)
        XCTAssertFalse(isOtherActive)
    }

    func testProcessingState() async {
        let queue = MessageQueue()

        let processing1 = await queue.isCurrentlyProcessing
        XCTAssertFalse(processing1)

        await queue.startProcessing()
        let processing2 = await queue.isCurrentlyProcessing
        XCTAssertTrue(processing2)

        await queue.finishProcessing()
        let processing3 = await queue.isCurrentlyProcessing
        XCTAssertFalse(processing3)
    }
}
