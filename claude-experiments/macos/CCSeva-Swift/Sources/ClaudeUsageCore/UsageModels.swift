//
//  UsageModels.swift
//  CCSeva
//
//  Data models for Claude usage statistics
//  Based on original CCSeva: https://github.com/Iamshankhadeep/ccseva
//

import Foundation

// MARK: - Main Usage Statistics
public struct UsageStats: Codable {
    public let today: DailyUsage
    public let thisWeek: [DailyUsage]
    public let thisMonth: [DailyUsage]
    public let burnRate: Double
    public let velocity: VelocityInfo
    public let prediction: PredictionInfo
    public let resetInfo: ResetTimeInfo
    public let predictedDepleted: String?
    public let currentPlan: String
    public let tokenLimit: Int
    public let tokensUsed: Int
    public let tokensRemaining: Int
    public let percentageUsed: Double
}

// MARK: - Daily Usage
public struct DailyUsage: Codable {
    public let date: String
    public let totalTokens: Int
    public let totalCost: Double
    public let models: [String: ModelUsage]
}

public struct ModelUsage: Codable {
    public let tokens: Int
    public let cost: Double
}

// MARK: - Velocity Information
public struct VelocityInfo: Codable {
    public let current: Double
    public let average24h: Double
    public let average7d: Double
    public let trend: String // "increasing", "decreasing", "stable"
    public let trendPercent: Double
    public let peakHour: Int
    public let isAccelerating: Bool
}

// MARK: - Prediction Information
public struct PredictionInfo: Codable {
    public let depletionTime: String?
    public let confidence: Int
    public let daysRemaining: Double
    public let recommendedDailyLimit: Int
    public let onTrackForReset: Bool
}

// MARK: - Reset Time Information
public struct ResetTimeInfo: Codable {
    public let timeUntilReset: String
    public let nextResetDate: String
    public let hoursUntilReset: Double
    public let timezone: String
}

// MARK: - Menu Bar Data
public struct MenuBarData {
    public let tokensUsed: Int
    public let tokenLimit: Int
    public let percentageUsed: Double
    public let status: UsageStatus
    public let cost: Double
    public let timeUntilReset: String?
    
    public init(tokensUsed: Int, tokenLimit: Int, percentageUsed: Double, status: UsageStatus, cost: Double, timeUntilReset: String?) {
        self.tokensUsed = tokensUsed
        self.tokenLimit = tokenLimit
        self.percentageUsed = percentageUsed
        self.status = status
        self.cost = cost
        self.timeUntilReset = timeUntilReset
    }
}

public enum UsageStatus {
    case safe
    case warning
    case critical
    
    var color: String {
        switch self {
        case .safe: return "green"
        case .warning: return "orange"
        case .critical: return "red"
        }
    }
}