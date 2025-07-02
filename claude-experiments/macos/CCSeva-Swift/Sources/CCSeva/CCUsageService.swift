//
//  CCUsageService.swift
//  CCSeva
//
//  Service for fetching Claude usage data natively from ~/.claude files
//  Based on original CCSeva: https://github.com/Iamshankhadeep/ccseva
//

import Foundation
import ClaudeUsageCore

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
    
    private init() {}
    
    func startMonitoring() {
        print("🔄 Starting usage monitoring...")
        fetchUsageData()
        
        refreshTimer = Timer.scheduledTimer(withTimeInterval: refreshInterval, repeats: true) { _ in
            self.fetchUsageData()
        }
    }
    
    func stopMonitoring() {
        print("⏹️ Stopping usage monitoring...")
        refreshTimer?.invalidate()
        refreshTimer = nil
    }
    
    func fetchUsageData() {
        lastError = nil
        
        Task {
            do {
                let stats = try await self.readClaudeUsageData()
                
                await MainActor.run {
                    self.currentStats = stats
                    self.isLoading = false
                    print("✅ Usage data updated: \(stats.percentageUsed)% used, \(stats.tokensUsed)/\(stats.tokenLimit) tokens")
                    
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
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let stats = try self.usageReader.generateUsageStats()
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
}

