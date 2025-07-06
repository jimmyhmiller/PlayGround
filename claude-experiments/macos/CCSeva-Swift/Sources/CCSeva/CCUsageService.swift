//
//  CCUsageService.swift
//  CCSeva
//
//  Service for fetching Claude usage data natively from ~/.claude files
//  Based on original CCSeva: https://github.com/Iamshankhadeep/ccseva
//

import Foundation
import ClaudeUsageCore
import Dispatch

// MARK: - Notification Names
extension Notification.Name {
    static let usageDataUpdated = Notification.Name("usageDataUpdated")
}

class CCUsageService: ObservableObject {
    static let shared = CCUsageService()
    
    @Published var currentStats: UsageStats?
    @Published var lastError: String?
    @Published var isLoading: Bool = true
    
    private var refreshTimer: Timer?
    private let refreshInterval: TimeInterval = 30 // 30 seconds like original
    private let usageReader = ClaudeUsageReader()
    
    // File system watcher components
    private var fileWatcher: DispatchSourceFileSystemObject?
    private let watcherQueue = DispatchQueue(label: "com.ccseva.filewatcher", qos: .utility)
    private var claudeProjectsURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".claude")
            .appendingPathComponent("projects")
    }
    
    private init() {}
    
    func startMonitoring() {
        print("🔄 Starting usage monitoring...")
        fetchUsageData()
        
        // Try to start file system watcher first, fall back to timer if needed
        if startFileSystemWatcher() {
            print("✅ File system watcher started successfully")
            // Still use a timer as backup for initial load and periodic checks
            refreshTimer = Timer.scheduledTimer(withTimeInterval: refreshInterval * 4, repeats: true) { _ in
                self.fetchUsageData()
            }
        } else {
            print("⚠️ File system watcher failed, using timer-only mode")
            // Fall back to original timer-based approach
            refreshTimer = Timer.scheduledTimer(withTimeInterval: refreshInterval, repeats: true) { _ in
                self.fetchUsageData()
            }
        }
    }
    
    func stopMonitoring() {
        print("⏹️ Stopping usage monitoring...")
        refreshTimer?.invalidate()
        refreshTimer = nil
        
        stopFileSystemWatcher()
    }
    
    func fetchUsageData() {
        lastError = nil
        
        // Use Task.detached for better concurrency and to avoid inheriting task priorities
        Task.detached(priority: .utility) { [weak self] in
            guard let self = self else { return }
            
            do {
                let stats = try await self.readClaudeUsageData()
                
                await MainActor.run {
                    self.currentStats = stats
                    self.isLoading = false
                    print("✅ Usage data updated: \(stats.percentageUsed)% used, \(stats.tokensUsed)/\(stats.tokenLimit) tokens")
                    
                    // Log cache performance
                    let cacheStats = self.usageReader.getCacheStats()
                    if cacheStats.filesInCache > 0 {
                        print("💾 Cache stats: \(cacheStats.filesInCache) files cached, \(cacheStats.totalEntries) total entries")
                    }
                    
                    // Notify observers
                    NotificationCenter.default.post(name: .usageDataUpdated, object: nil)
                }
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.isLoading = false
                    print("❌ Failed to read Claude usage data: \(error)")
                    
                    // Keep existing data if we have it
                    if self.currentStats == nil {
                        // Only set to nil if we don't have any data yet
                        self.currentStats = nil
                    }
                }
            }
        }
    }
    
    private func readClaudeUsageData() async throws -> UsageStats {
        // Use the existing concurrency pattern but optimize the priority
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                do {
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let stats = try self.usageReader.generateUsageStats()
                    let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
                    print("⏱️ Usage stats generation took \(String(format: "%.3f", timeElapsed))s")
                    continuation.resume(returning: stats)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    
    func getMenuBarData() -> MenuBarData {
        guard let stats = currentStats else {
            return MenuBarData(
                tokensUsed: 0,
                tokenLimit: 7000,
                percentageUsed: 0,
                status: .safe,
                cost: 0.0,
                timeUntilReset: nil
            )
        }
        
        let status: UsageStatus
        if stats.percentageUsed >= 90 {
            status = .critical
        } else if stats.percentageUsed >= 70 {
            status = .warning
        } else {
            status = .safe
        }
        
        return MenuBarData(
            tokensUsed: stats.tokensUsed,
            tokenLimit: stats.tokenLimit,
            percentageUsed: stats.percentageUsed,
            status: status,
            cost: stats.today.totalCost,
            timeUntilReset: stats.resetInfo.timeUntilReset
        )
    }
    
    // MARK: - Cache Management
    
    func clearCache() {
        usageReader.clearCache()
    }
    
    func getCacheStats() -> (filesInCache: Int, totalEntries: Int) {
        return usageReader.getCacheStats()
    }
    
    // MARK: - File System Watcher
    
    private func startFileSystemWatcher() -> Bool {
        guard FileManager.default.fileExists(atPath: claudeProjectsURL.path) else {
            print("❌ Claude projects directory not found: \(claudeProjectsURL.path)")
            return false
        }
        
        let fileDescriptor = open(claudeProjectsURL.path, O_EVTONLY)
        guard fileDescriptor >= 0 else {
            print("❌ Failed to open file descriptor for: \(claudeProjectsURL.path)")
            return false
        }
        
        fileWatcher = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fileDescriptor,
            eventMask: [.write, .extend, .attrib, .link, .rename, .revoke],
            queue: watcherQueue
        )
        
        guard let watcher = fileWatcher else {
            close(fileDescriptor)
            return false
        }
        
        watcher.setEventHandler { [weak self] in
            print("📁 File system change detected in Claude projects directory")
            
            // Debounce rapid file changes by adding a small delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self?.fetchUsageData()
            }
        }
        
        watcher.setCancelHandler {
            close(fileDescriptor)
            print("🔒 File system watcher file descriptor closed")
        }
        
        watcher.resume()
        return true
    }
    
    private func stopFileSystemWatcher() {
        fileWatcher?.cancel()
        fileWatcher = nil
        print("⏹️ File system watcher stopped")
    }
}

