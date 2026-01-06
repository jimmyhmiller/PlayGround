import XCTest
@testable import ACPLib

final class ModeManagerTests: XCTestCase {

    func testInitialState() async {
        let manager = ModeManager()

        let availableModes = await manager.availableModes
        let currentModeId = await manager.currentModeId
        let currentMode = await manager.currentMode
        let hasMultipleModes = await manager.hasMultipleModes

        XCTAssertTrue(availableModes.isEmpty)
        XCTAssertNil(currentModeId)
        XCTAssertNil(currentMode)
        XCTAssertFalse(hasMultipleModes)
    }

    func testUpdateModes() async {
        let manager = ModeManager()

        let modes = [
            ACPMode(id: "agent", name: "Agent"),
            ACPMode(id: "plan", name: "Plan")
        ]
        let modeInfo = ACPModeInfo(availableModes: modes, currentModeId: "agent")

        await manager.updateModes(from: modeInfo)

        let availableModes = await manager.availableModes
        let currentModeId = await manager.currentModeId
        let currentMode = await manager.currentMode
        let hasMultipleModes = await manager.hasMultipleModes

        XCTAssertEqual(availableModes.count, 2)
        XCTAssertEqual(currentModeId, "agent")
        XCTAssertEqual(currentMode?.name, "Agent")
        XCTAssertTrue(hasMultipleModes)
    }

    func testUpdateModesNil() async {
        let manager = ModeManager()

        let modes = [ACPMode(id: "agent", name: "Agent")]
        let modeInfo = ACPModeInfo(availableModes: modes, currentModeId: "agent")
        await manager.updateModes(from: modeInfo)

        // Updating with nil should not change state
        await manager.updateModes(from: nil)

        let availableModes = await manager.availableModes
        let currentModeId = await manager.currentModeId

        XCTAssertEqual(availableModes.count, 1)
        XCTAssertEqual(currentModeId, "agent")
    }

    func testSetCurrentMode() async {
        let manager = ModeManager()

        let modes = [
            ACPMode(id: "agent", name: "Agent"),
            ACPMode(id: "plan", name: "Plan")
        ]
        await manager.updateModes(from: ACPModeInfo(availableModes: modes, currentModeId: "agent"))

        await manager.setCurrentMode("plan")

        let currentModeId = await manager.currentModeId
        let currentMode = await manager.currentMode

        XCTAssertEqual(currentModeId, "plan")
        XCTAssertEqual(currentMode?.name, "Plan")
    }

    func testClear() async {
        let manager = ModeManager()

        let modes = [ACPMode(id: "agent", name: "Agent")]
        await manager.updateModes(from: ACPModeInfo(availableModes: modes, currentModeId: "agent"))

        await manager.clear()

        let availableModes = await manager.availableModes
        let currentModeId = await manager.currentModeId
        let currentMode = await manager.currentMode

        XCTAssertTrue(availableModes.isEmpty)
        XCTAssertNil(currentModeId)
        XCTAssertNil(currentMode)
    }

    func testNextMode() async {
        let manager = ModeManager()

        let modes = [
            ACPMode(id: "agent", name: "Agent"),
            ACPMode(id: "plan", name: "Plan"),
            ACPMode(id: "approval", name: "Approval")
        ]
        await manager.updateModes(from: ACPModeInfo(availableModes: modes, currentModeId: "agent"))

        // agent -> plan
        var next = await manager.nextMode()
        XCTAssertEqual(next?.id, "plan")

        await manager.setCurrentMode("plan")

        // plan -> approval
        next = await manager.nextMode()
        XCTAssertEqual(next?.id, "approval")

        await manager.setCurrentMode("approval")

        // approval -> agent (wrap around)
        next = await manager.nextMode()
        XCTAssertEqual(next?.id, "agent")
    }

    func testNextModeEmpty() async {
        let manager = ModeManager()

        let next = await manager.nextMode()
        XCTAssertNil(next)
    }

    func testNextModeNoCurrentMode() async {
        let manager = ModeManager()

        let modes = [
            ACPMode(id: "agent", name: "Agent"),
            ACPMode(id: "plan", name: "Plan")
        ]
        // Set modes but don't set current mode
        await manager.updateModes(from: ACPModeInfo(availableModes: modes, currentModeId: "nonexistent"))
        await manager.setCurrentMode("") // Clear current

        let next = await manager.nextMode()
        // Should return first mode when current is not found
        XCTAssertEqual(next?.id, "agent")
    }

    func testPreviousMode() async {
        let manager = ModeManager()

        let modes = [
            ACPMode(id: "agent", name: "Agent"),
            ACPMode(id: "plan", name: "Plan"),
            ACPMode(id: "approval", name: "Approval")
        ]
        await manager.updateModes(from: ACPModeInfo(availableModes: modes, currentModeId: "plan"))

        // plan -> agent
        var prev = await manager.previousMode()
        XCTAssertEqual(prev?.id, "agent")

        await manager.setCurrentMode("agent")

        // agent -> approval (wrap around)
        prev = await manager.previousMode()
        XCTAssertEqual(prev?.id, "approval")
    }

    func testHasMultipleModesSingleMode() async {
        let manager = ModeManager()

        let modes = [ACPMode(id: "agent", name: "Agent")]
        await manager.updateModes(from: ACPModeInfo(availableModes: modes, currentModeId: "agent"))

        let hasMultipleModes = await manager.hasMultipleModes
        XCTAssertFalse(hasMultipleModes)
    }

    func testHasMultipleModesMultipleModes() async {
        let manager = ModeManager()

        let modes = [
            ACPMode(id: "agent", name: "Agent"),
            ACPMode(id: "plan", name: "Plan")
        ]
        await manager.updateModes(from: ACPModeInfo(availableModes: modes, currentModeId: "agent"))

        let hasMultipleModes = await manager.hasMultipleModes
        XCTAssertTrue(hasMultipleModes)
    }
}
