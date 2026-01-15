import Foundation
#if os(iOS)
import UIKit

/// Service that keeps network operations alive when the app is backgrounded
/// Uses UIApplication background tasks in combination with location tracking
class NetworkKeepAliveService {
    static let shared = NetworkKeepAliveService()

    private var backgroundTaskId: UIBackgroundTaskIdentifier = .invalid
    private var heartbeatTimer: Timer?
    private var isActive = false

    private init() {
        // Observe app lifecycle
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appWillEnterForeground),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )
    }

    /// Start the network keep-alive service
    func start() {
        guard !isActive else { return }
        isActive = true
        appLog("NetworkKeepAlive: started", category: "Network")
        startHeartbeat()
    }

    /// Stop the network keep-alive service
    func stop() {
        guard isActive else { return }
        isActive = false
        appLog("NetworkKeepAlive: stopped", category: "Network")
        stopHeartbeat()
        endBackgroundTask()
    }

    @objc private func appDidEnterBackground() {
        guard isActive else { return }
        appLog("NetworkKeepAlive: app entering background, starting background task", category: "Network")
        beginBackgroundTask()
    }

    @objc private func appWillEnterForeground() {
        appLog("NetworkKeepAlive: app entering foreground", category: "Network")
        endBackgroundTask()
    }

    private func beginBackgroundTask() {
        // End any existing task first
        endBackgroundTask()

        backgroundTaskId = UIApplication.shared.beginBackgroundTask(withName: "NetworkKeepAlive") { [weak self] in
            appLog("NetworkKeepAlive: background task expiring, will restart", category: "Network")
            self?.endBackgroundTask()
            // Try to start a new one
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                self?.beginBackgroundTask()
            }
        }

        let remaining = UIApplication.shared.backgroundTimeRemaining
        appLog("NetworkKeepAlive: background task started, remaining time: \(remaining)s", category: "Network")
    }

    private func endBackgroundTask() {
        guard backgroundTaskId != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTaskId)
        backgroundTaskId = .invalid
        appLog("NetworkKeepAlive: background task ended", category: "Network")
    }

    private func startHeartbeat() {
        stopHeartbeat()

        // Create timer on main run loop to ensure it fires in background
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.heartbeat()
        }

        // Make sure timer fires in all run loop modes (including tracking mode)
        if let timer = heartbeatTimer {
            RunLoop.main.add(timer, forMode: .common)
        }

        appLog("NetworkKeepAlive: heartbeat timer started (5s interval)", category: "Network")
    }

    private func stopHeartbeat() {
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
    }

    private func heartbeat() {
        let backgroundState = UIApplication.shared.applicationState == .background
        let remaining = UIApplication.shared.backgroundTimeRemaining
        appLog("NetworkKeepAlive: heartbeat (background=\(backgroundState), remaining=\(String(format: "%.1f", remaining))s)", category: "Network")

        // If we're in background and running low on time, try to extend
        if backgroundState && remaining < 30 && remaining > 0 {
            appLog("NetworkKeepAlive: low background time, attempting to extend", category: "Network")
            // The background task will auto-restart when it expires
        }
    }
}
#endif
