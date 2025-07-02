import SwiftUI

struct ContentView: View {
    @State private var selectedTab = "home"
    @State private var sidebarVisible = true
    
    var body: some View {
        GeometryReader { geometry in
            HStack(spacing: 0) {
                // Floating Sidebar
                if sidebarVisible {
                    FloatingSidebar(selectedTab: $selectedTab)
                        .transition(.move(edge: .leading))
                }
                
                // Main content area
                VStack(spacing: 0) {
                    // Floating navigation bar
                    FloatingNavigationBar(
                        selectedTab: $selectedTab,
                        sidebarVisible: $sidebarVisible
                    )
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    
                    // Content area
                    MainContentView(selectedTab: selectedTab)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding(.horizontal, 20)
                        .padding(.bottom, 20)
                }
                .background(Color(NSColor.controlBackgroundColor))
            }
        }
        .animation(.spring(response: 0.5, dampingFraction: 0.8), value: sidebarVisible)
    }
}

struct FloatingNavigationBar: View {
    @Binding var selectedTab: String
    @Binding var sidebarVisible: Bool
    
    let tabs = [
        ("home", "house.fill", "Home"),
        ("projects", "folder.fill", "Projects"),
        ("settings", "gear", "Settings")
    ]
    
    var body: some View {
        HStack {
            // Sidebar toggle
            GlassButton(
                icon: "sidebar.left",
                action: { sidebarVisible.toggle() }
            )
            
            Spacer()
            
            // Tab navigation
            HStack(spacing: 8) {
                ForEach(tabs, id: \.0) { tab in
                    GlassTabButton(
                        icon: tab.1,
                        title: tab.2,
                        isSelected: selectedTab == tab.0,
                        action: { selectedTab = tab.0 }
                    )
                }
            }
            
            Spacer()
            
            // Action buttons
            HStack(spacing: 8) {
                GlassButton(
                    icon: "testtube.2",
                    title: "Test Claude",
                    action: { 
                        TestRunner.runClaudeTests()
                    }
                )
                
                GlassButton(
                    icon: "plus",
                    action: { /* Add action */ }
                )
                
                GlassButton(
                    icon: "magnifyingglass",
                    action: { /* Search action */ }
                )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(.white.opacity(0.2), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.1), radius: 20, x: 0, y: 10)
    }
}

struct MainContentView: View {
    let selectedTab: String
    
    var body: some View {
        VStack {
            switch selectedTab {
            case "home":
                HomeContentView()
            case "projects":
                ProjectsContentView()
            case "settings":
                SettingsContentView()
            default:
                HomeContentView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(.easeInOut(duration: 0.3), value: selectedTab)
    }
}

struct HomeContentView: View {
    var body: some View {
        HStack(spacing: 20) {
            // Left side - Agent cards
            ScrollView {
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 16), count: 2), spacing: 16) {
                    ForEach(0..<4, id: \.self) { index in
                        AgentCard(
                            agentName: ["Agent Alpha", "Agent Beta", "Agent Gamma", "Agent Delta"][index],
                            status: ["Online", "Busy", "Online", "Offline"][index],
                            taskCount: [3, 1, 5, 0][index]
                        )
                    }
                }
                .padding(.top, 20)
            }
            .frame(maxWidth: .infinity)
            
            // Right side - Chat widget
            ChatWidget()
                .frame(width: 400, height: 600)
        }
    }
}

struct AgentCard: View {
    let agentName: String
    let status: String
    let taskCount: Int
    
    private var statusColor: Color {
        switch status {
        case "Online": return .green
        case "Busy": return .orange
        case "Offline": return .gray
        default: return .gray
        }
    }
    
    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    AgentAvatar(agentName: agentName, size: 40)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text(agentName)
                            .font(.headline)
                            .foregroundStyle(.primary)
                        
                        HStack(spacing: 4) {
                            Circle()
                                .fill(statusColor)
                                .frame(width: 8, height: 8)
                            
                            Text(status)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    Spacer()
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Active Tasks: \(taskCount)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    
                    Text("AI-powered automation agent ready to help with complex workflows and data processing.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                HStack {
                    GlassButton(
                        icon: "message",
                        title: "Chat",
                        compact: false,
                        action: { /* Start chat */ }
                    )
                    
                    Spacer()
                    
                    GlassButton(
                        icon: "gear",
                        compact: true,
                        action: { /* Configure */ }
                    )
                }
            }
            .padding(20)
        }
        .frame(height: 200)
    }
}

struct ProjectsContentView: View {
    var body: some View {
        VStack {
            Text("Projects")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundStyle(.primary)
            
            Text("Your projects will appear here")
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct SettingsContentView: View {
    var body: some View {
        VStack {
            Text("Settings")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundStyle(.primary)
            
            Text("App settings and preferences")
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}