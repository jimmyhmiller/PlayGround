import Foundation

// MARK: - Message Queue

/// Manages prompt ID tracking for handling new prompts during streaming
/// and filtering stale events
public actor MessageQueue {
    // MARK: - State

    private var currentPromptId: String?
    private var isProcessing = false

    // MARK: - Prompt ID Generation

    /// Generate a unique prompt ID
    public func generatePromptId() -> String {
        "prompt-\(Date().timeIntervalSince1970)-\(UUID().uuidString.prefix(8))"
    }

    // MARK: - Current Prompt Tracking

    /// Get the currently active prompt ID
    public var activePromptId: String? { currentPromptId }

    /// Whether a prompt is currently being processed
    public var isPromptActive: Bool { currentPromptId != nil }

    /// Set the active prompt ID
    public func setActivePrompt(_ id: String) {
        currentPromptId = id
        isProcessing = true
    }

    /// Clear the active prompt
    public func clearActivePrompt() {
        currentPromptId = nil
        isProcessing = false
    }

    /// Invalidate a specific prompt ID
    /// Returns true if the prompt was the active one
    public func invalidatePrompt(_ id: String) -> Bool {
        if currentPromptId == id {
            currentPromptId = nil
            isProcessing = false
            return true
        }
        return false
    }

    // MARK: - Event Filtering

    /// Check if an event should be processed based on prompt ID
    /// This filters out stale events from cancelled prompts
    public func shouldProcessEvent(forPromptId eventPromptId: String?) -> Bool {
        // If no active prompt, ignore streaming events (stale from cancelled prompt)
        guard let activeId = currentPromptId else { return false }

        // If event has a prompt ID, it must match
        if let eventPromptId = eventPromptId {
            return eventPromptId == activeId
        }

        // Events without prompt ID are processed if we have an active prompt
        return true
    }

    /// Check if an event for a specific prompt ID should be processed
    public func isActivePrompt(_ promptId: String) -> Bool {
        currentPromptId == promptId
    }

    // MARK: - Processing State

    /// Mark processing as started
    public func startProcessing() {
        isProcessing = true
    }

    /// Mark processing as completed
    public func finishProcessing() {
        isProcessing = false
    }

    /// Whether we're currently processing a prompt
    public var isCurrentlyProcessing: Bool { isProcessing }
}
