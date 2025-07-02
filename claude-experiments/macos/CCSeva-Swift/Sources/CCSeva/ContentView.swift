//
//  ContentView.swift
//  CCSeva
//
//  Main SwiftUI view for the CCSeva menu bar popover
//  Original concept from: https://github.com/Iamshankhadeep/ccseva
//

import SwiftUI
import ClaudeUsageCore

struct ContentView: View {
    @State private var selectedTab = 0
    @ObservedObject private var usageService = CCUsageService.shared
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                // Claude icon placeholder
                Circle()
                    .fill(Color(.systemOrange))
                    .frame(width: 24, height: 24)
                
                VStack(alignment: .leading) {
                    Text("Claude Usage")
                        .font(.title2)
                        .fontWeight(.bold)
                    if let stats = usageService.currentStats {
                        Text("\(stats.currentPlan) Plan · API Equiv. Costs")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        Text("Track API usage")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                Button("Quit") {
                    print("🛑 Quit button pressed")
                    NSApplication.shared.terminate(nil)
                }
                .buttonStyle(.bordered)
            }
            .padding()
            .background(Color(.controlBackgroundColor))
            
            // Tab selector
            Picker("View", selection: $selectedTab) {
                Text("Dashboard").tag(0)
                Text("Analytics").tag(1)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 8)
            
            // Content area
            Group {
                if usageService.isLoading {
                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.5)
                        
                        Text("Loading Claude usage data...")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Text("Parsing usage files from ~/.claude")
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding()
                } else {
                    switch selectedTab {
                    case 0:
                        DashboardView(usageService: usageService)
                    case 1:
                        AnalyticsView(usageService: usageService)
                    default:
                        DashboardView(usageService: usageService)
                    }
                }
            }
        }
        .frame(width: 600, height: 600)
        .onAppear {
            print("✅ ContentView appeared")
        }
    }
}

struct DashboardView: View {
    @ObservedObject var usageService: CCUsageService
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Dashboard")
                .font(.title)
                .fontWeight(.bold)
            
            if let stats = usageService.currentStats {
                VStack(spacing: 16) {
                    // Usage Progress
                    VStack(spacing: 8) {
                        HStack {
                            Text("Today's Usage Progress")
                                .font(.headline)
                            Spacer()
                            Text(String(format: "%.0f%%", stats.percentageUsed))
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(getUsageColor(stats.percentageUsed))
                        }
                        
                        ProgressView(value: stats.percentageUsed, total: 100)
                            .progressViewStyle(LinearProgressViewStyle(tint: getUsageColor(stats.percentageUsed)))
                        
                        HStack {
                            Text("\(formatNumber(stats.tokensUsed)) of \(formatNumber(stats.tokenLimit)) tokens")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("Resets at \(stats.resetInfo.timezone)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        HStack {
                            Text("Percentage relative to your highest usage day")
                                .font(.caption2)
                                .foregroundColor(.secondary.opacity(0.7))
                            Spacer()
                        }
                    }
                    .padding()
                    .background(Color(.controlBackgroundColor))
                    .cornerRadius(12)
                    
                    // Stats Grid
                    HStack(spacing: 12) {
                        VStack(spacing: 8) {
                            Text("API Equiv. Cost")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "$%.2f", stats.today.totalCost))
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.controlBackgroundColor))
                        .cornerRadius(8)
                        
                        VStack(spacing: 8) {
                            Text("Plan")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(stats.currentPlan)
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.controlBackgroundColor))
                        .cornerRadius(8)
                        
                        VStack(spacing: 8) {
                            Text("Burn Rate")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(Int(stats.burnRate))/hr")
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.controlBackgroundColor))
                        .cornerRadius(8)
                    }
                    
                    // Model Usage
                    if !stats.today.models.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Today's Model Usage")
                                .font(.headline)
                            
                            ForEach(Array(stats.today.models.keys.sorted()), id: \.self) { modelName in
                                if let modelData = stats.today.models[modelName] {
                                    HStack {
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(modelName.replacingOccurrences(of: "claude-", with: ""))
                                                .font(.subheadline)
                                                .fontWeight(.medium)
                                            Text("\(formatNumber(modelData.tokens)) tokens")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                        Spacer()
                                        VStack(alignment: .trailing, spacing: 2) {
                                            Text(String(format: "$%.2f", modelData.cost))
                                                .font(.subheadline)
                                                .fontWeight(.medium)
                                            Text("API equiv.")
                                                .font(.caption2)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                    .padding(.vertical, 4)
                                }
                            }
                        }
                        .padding()
                        .background(Color(.controlBackgroundColor))
                        .cornerRadius(12)
                    }
                }
            } else if let error = usageService.lastError {
                VStack(spacing: 16) {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundColor(.orange)
                        .font(.system(size: 48))
                    
                    Text("Unable to Load Claude Usage Data")
                        .font(.title2)
                        .fontWeight(.semibold)
                        .multilineTextAlignment(.center)
                    
                    Text(error)
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    
                    if error.contains("Claude directory not found") {
                        VStack(spacing: 8) {
                            Text("Make sure you have used Claude Code before.")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text("Usage data is stored in ~/.claude after your first Claude Code session.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                        }
                        .padding(.top)
                    }
                    
                    Button("Try Again") {
                        usageService.fetchUsageData()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
            } else {
                VStack(spacing: 16) {
                    Image(systemName: "chart.bar")
                        .foregroundColor(.blue)
                        .font(.system(size: 48))
                    
                    Text("No Usage Data")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    Text("Click below to load your Claude usage statistics")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    
                    Button("Load Usage Data") {
                        usageService.fetchUsageData()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
            }
            
            Spacer()
        }
        .padding()
    }
    
    private func getUsageColor(_ percentage: Double) -> Color {
        if percentage >= 90 {
            return .red
        } else if percentage >= 70 {
            return .orange
        } else {
            return .green
        }
    }
    
    private func formatNumber(_ number: Int) -> String {
        if number >= 1000000 {
            return String(format: "%.1fM", Double(number) / 1000000)
        } else if number >= 1000 {
            return String(format: "%.1fK", Double(number) / 1000)
        } else {
            return "\(number)"
        }
    }
}

struct AnalyticsView: View {
    @ObservedObject var usageService: CCUsageService
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Analytics")
                .font(.title)
                .fontWeight(.bold)
            
            if let stats = usageService.currentStats {
                ScrollView {
                    VStack(spacing: 16) {
                        // Velocity Info
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Usage Velocity")
                                .font(.headline)
                            
                            HStack(spacing: 16) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Current")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text("\(Int(stats.velocity.current))/hr")
                                        .font(.title3)
                                        .fontWeight(.semibold)
                                }
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("24h Avg")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text("\(Int(stats.velocity.average24h))/hr")
                                        .font(.title3)
                                        .fontWeight(.semibold)
                                }
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("7d Avg")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text("\(Int(stats.velocity.average7d))/hr")
                                        .font(.title3)
                                        .fontWeight(.semibold)
                                }
                                
                                Spacer()
                                
                                VStack(alignment: .trailing, spacing: 4) {
                                    Text("Trend")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    HStack {
                                        Image(systemName: getTrendIcon(stats.velocity.trend))
                                            .foregroundColor(getTrendColor(stats.velocity.trend))
                                        Text(stats.velocity.trend.capitalized)
                                            .font(.subheadline)
                                            .fontWeight(.medium)
                                    }
                                }
                            }
                        }
                        .padding()
                        .background(Color(.controlBackgroundColor))
                        .cornerRadius(12)
                        
                        // Weekly Usage
                        VStack(alignment: .leading, spacing: 12) {
                            Text("This Week")
                                .font(.headline)
                            
                            ForEach(stats.thisWeek.reversed(), id: \.date) { day in
                                HStack {
                                    Text(formatDate(day.date))
                                        .font(.subheadline)
                                        .frame(width: 80, alignment: .leading)
                                    
                                    VStack(alignment: .leading, spacing: 2) {
                                        ProgressView(value: Double(day.totalTokens), total: Double(stats.tokenLimit))
                                            .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                                        HStack {
                                            Text("\(formatNumber(day.totalTokens)) tokens")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                            Spacer()
                                            Text(String(format: "$%.2f (API)", day.totalCost))
                                                .font(.caption)
                                                .fontWeight(.medium)
                                        }
                                    }
                                }
                                .padding(.vertical, 4)
                            }
                        }
                        .padding()
                        .background(Color(.controlBackgroundColor))
                        .cornerRadius(12)
                        
                        // Session Info
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Usage Pattern")
                                .font(.headline)
                            
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Until Reset")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text(String(format: "%.1f hours", stats.prediction.daysRemaining * 24))
                                        .font(.title3)
                                        .fontWeight(.semibold)
                                }
                                
                                Spacer()
                                
                                VStack(alignment: .trailing, spacing: 4) {
                                    Text("Plan Type")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text("\(stats.currentPlan) (Session-based)")
                                        .font(.title3)
                                        .fontWeight(.semibold)
                                        .foregroundColor(.blue)
                                }
                            }
                            
                            VStack(alignment: .leading, spacing: 8) {
                                Text("About your plan:")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                
                                Text("Claude \(stats.currentPlan) uses session-based rate limiting (5-hour blocks) rather than hard daily limits. Usage resets at \(stats.resetInfo.timezone) and you can continue using Claude normally.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.top, 8)
                        }
                        .padding()
                        .background(Color(.controlBackgroundColor))
                        .cornerRadius(12)
                        
                        // Reset Info
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Reset Information")
                                .font(.headline)
                            
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Time Until Reset")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text(stats.resetInfo.timeUntilReset)
                                        .font(.title3)
                                        .fontWeight(.semibold)
                                }
                                
                                Spacer()
                                
                                VStack(alignment: .trailing, spacing: 4) {
                                    Text("Resets At")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text(stats.resetInfo.timezone)
                                        .font(.title3)
                                        .fontWeight(.semibold)
                                }
                            }
                        }
                        .padding()
                        .background(Color(.controlBackgroundColor))
                        .cornerRadius(12)
                    }
                }
            } else {
                Text("No analytics data available")
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding()
    }
    
    private func formatDate(_ dateString: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        
        if let date = formatter.date(from: dateString) {
            formatter.dateFormat = "MMM dd"
            return formatter.string(from: date)
        }
        return dateString
    }
    
    private func formatNumber(_ number: Int) -> String {
        if number >= 1000000 {
            return String(format: "%.1fM", Double(number) / 1000000)
        } else if number >= 1000 {
            return String(format: "%.1fK", Double(number) / 1000)
        } else {
            return "\(number)"
        }
    }
    
    private func getTrendIcon(_ trend: String) -> String {
        switch trend {
        case "increasing":
            return "arrow.up.right"
        case "decreasing":
            return "arrow.down.right"
        default:
            return "arrow.right"
        }
    }
    
    private func getTrendColor(_ trend: String) -> Color {
        switch trend {
        case "increasing":
            return .red
        case "decreasing":
            return .green
        default:
            return .blue
        }
    }
    
    private func getConfidenceColor(_ confidence: Int) -> Color {
        if confidence >= 80 {
            return .green
        } else if confidence >= 60 {
            return .orange
        } else {
            return .red
        }
    }
}


#Preview {
    ContentView()
}