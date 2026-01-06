import Foundation

// MARK: - Mode Manager

/// Manages mode state and cycling for ACP sessions
public actor ModeManager {
    // MARK: - State

    private var _availableModes: [ACPMode] = []
    private var _currentModeId: String?

    // MARK: - Properties

    /// Available modes for the current session
    public var availableModes: [ACPMode] { _availableModes }

    /// Current mode ID
    public var currentModeId: String? { _currentModeId }

    /// Current mode object
    public var currentMode: ACPMode? {
        guard let id = _currentModeId else { return nil }
        return _availableModes.first { $0.id == id }
    }

    /// Whether multiple modes are available
    public var hasMultipleModes: Bool {
        _availableModes.count > 1
    }

    // MARK: - Update

    /// Update mode info from session creation/resume
    public func updateModes(from modeInfo: ACPModeInfo?) {
        guard let modeInfo = modeInfo else { return }
        _availableModes = modeInfo.availableModes
        _currentModeId = modeInfo.currentModeId
    }

    /// Update current mode ID
    public func setCurrentMode(_ modeId: String) {
        _currentModeId = modeId
    }

    /// Clear all mode state
    public func clear() {
        _availableModes = []
        _currentModeId = nil
    }

    // MARK: - Cycling

    /// Get the next mode in the cycle
    public func nextMode() -> ACPMode? {
        guard !_availableModes.isEmpty else { return nil }

        guard let currentId = _currentModeId,
              let currentIndex = _availableModes.firstIndex(where: { $0.id == currentId }) else {
            return _availableModes.first
        }

        let nextIndex = (currentIndex + 1) % _availableModes.count
        return _availableModes[nextIndex]
    }

    /// Get the previous mode in the cycle
    public func previousMode() -> ACPMode? {
        guard !_availableModes.isEmpty else { return nil }

        guard let currentId = _currentModeId,
              let currentIndex = _availableModes.firstIndex(where: { $0.id == currentId }) else {
            return _availableModes.last
        }

        let prevIndex = currentIndex == 0 ? _availableModes.count - 1 : currentIndex - 1
        return _availableModes[prevIndex]
    }
}
