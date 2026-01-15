import ActivityKit
import Foundation

// Logging helper that works in both app and widget
fileprivate func activityLog(_ message: String) {
    #if !WIDGET_EXTENSION
    appLog(message, category: "LiveActivity")
    #else
    print("[LiveActivity] \(message)")
    #endif
}

/// Live Activity attributes for maintaining SSH/ACP connection in background
struct ConnectionActivityAttributes: ActivityAttributes {
    /// Static properties that don't change during the activity
    public struct ContentState: Codable, Hashable {
        var status: ConnectionStatus
        var currentOperation: String?
        var messagesExchanged: Int
        var lastActivityTime: Date

        enum ConnectionStatus: String, Codable {
            case connecting
            case connected
            case streaming
            case toolRunning
            case idle
            case reconnecting
            case error
        }
    }

    // Fixed properties set when activity starts
    var serverName: String
    var projectName: String
    var sessionId: String
}

/// Manager for Live Activity lifecycle
@MainActor
class ConnectionActivityManager: ObservableObject {
    static let shared = ConnectionActivityManager()

    @Published private(set) var currentActivity: Activity<ConnectionActivityAttributes>?
    @Published private(set) var isActivityActive = false

    private init() {}

    /// Start a new Live Activity for a connection
    func startActivity(
        serverName: String,
        projectName: String,
        sessionId: String
    ) {
        activityLog("startActivity called: server=\(serverName), project=\(projectName)")

        // End any existing activity synchronously (don't use Task)
        if let existingActivity = currentActivity {
            activityLog("Ending existing activity: \(existingActivity.id)")
            Task {
                await existingActivity.end(nil, dismissalPolicy: .immediate)
            }
            currentActivity = nil
            isActivityActive = false
        }

        let authInfo = ActivityAuthorizationInfo()
        activityLog("areActivitiesEnabled: \(authInfo.areActivitiesEnabled), frequentPushesEnabled: \(authInfo.frequentPushesEnabled)")

        guard authInfo.areActivitiesEnabled else {
            activityLog("Live Activities are not enabled - user needs to enable in Settings")
            return
        }

        let attributes = ConnectionActivityAttributes(
            serverName: serverName,
            projectName: projectName,
            sessionId: sessionId
        )

        let initialState = ConnectionActivityAttributes.ContentState(
            status: .connecting,
            currentOperation: "Establishing connection...",
            messagesExchanged: 0,
            lastActivityTime: Date()
        )

        let content = ActivityContent(state: initialState, staleDate: nil)

        do {
            let activity = try Activity.request(
                attributes: attributes,
                content: content,
                pushType: nil
            )
            currentActivity = activity
            isActivityActive = true
            activityLog("Started Live Activity: \(activity.id)")
        } catch {
            activityLog("Failed to start Live Activity: \(error)")
        }
    }

    /// Update the activity status
    func updateStatus(
        _ status: ConnectionActivityAttributes.ContentState.ConnectionStatus,
        operation: String? = nil,
        messagesExchanged: Int? = nil
    ) {
        guard let activity = currentActivity else { return }

        Task {
            let currentState = activity.content.state
            let newState = ConnectionActivityAttributes.ContentState(
                status: status,
                currentOperation: operation ?? currentState.currentOperation,
                messagesExchanged: messagesExchanged ?? currentState.messagesExchanged,
                lastActivityTime: Date()
            )

            let content = ActivityContent(state: newState, staleDate: nil)
            await activity.update(content)
            activityLog("Updated Live Activity: \(status.rawValue)")
        }
    }

    /// Update when streaming/receiving messages
    func updateStreaming(operation: String) {
        updateStatus(.streaming, operation: operation)
    }

    /// Update when a tool is running
    func updateToolRunning(toolName: String) {
        updateStatus(.toolRunning, operation: "Running: \(toolName)")
    }

    /// Update to idle state
    func updateIdle(messagesExchanged: Int) {
        updateStatus(.idle, operation: "Ready", messagesExchanged: messagesExchanged)
    }

    /// Update to connected state
    func updateConnected() {
        updateStatus(.connected, operation: "Connected")
    }

    /// End the Live Activity
    func endActivity() async {
        guard let activity = currentActivity else { return }

        let finalState = ConnectionActivityAttributes.ContentState(
            status: .idle,
            currentOperation: "Disconnected",
            messagesExchanged: activity.content.state.messagesExchanged,
            lastActivityTime: Date()
        )

        let content = ActivityContent(state: finalState, staleDate: nil)
        await activity.end(content, dismissalPolicy: .immediate)

        currentActivity = nil
        isActivityActive = false
        activityLog("Ended Live Activity")
    }

    /// End all activities (cleanup)
    func endAllActivities() async {
        for activity in Activity<ConnectionActivityAttributes>.activities {
            await activity.end(nil, dismissalPolicy: .immediate)
        }
        currentActivity = nil
        isActivityActive = false
    }
}
