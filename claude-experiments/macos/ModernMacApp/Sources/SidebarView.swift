import SwiftUI

struct SidebarView: View {
    @Binding var selectedTab: String
    @State private var searchText = ""
    
    let navigationItems = [
        SidebarSection(title: "Main", items: [
            SidebarItem(id: "home", icon: "house.fill", title: "Home", badge: nil),
            SidebarItem(id: "projects", icon: "folder.fill", title: "Projects", badge: "5"),
            SidebarItem(id: "favorites", icon: "heart.fill", title: "Favorites", badge: nil),
        ]),
        SidebarSection(title: "Tools", items: [
            SidebarItem(id: "analytics", icon: "chart.bar.fill", title: "Analytics", badge: nil),
            SidebarItem(id: "notifications", icon: "bell.fill", title: "Notifications", badge: "3"),
            SidebarItem(id: "settings", icon: "gear", title: "Settings", badge: nil),
        ]),
        SidebarSection(title: "Account", items: [
            SidebarItem(id: "profile", icon: "person.crop.circle.fill", title: "Profile", badge: nil),
            SidebarItem(id: "help", icon: "questionmark.circle.fill", title: "Help & Support", badge: nil),
        ])
    ]
    
    var body: some View {
        GlassPanel(cornerRadius: 0) {
            VStack(spacing: 0) {
                // Header
                VStack(spacing: 16) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Modern App")
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundStyle(.primary)
                            
                            Text("Liquid Glass Design")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        
                        Spacer()
                        
                        GlassButton(
                            icon: "ellipsis",
                            compact: true,
                            action: { /* Menu action */ }
                        )
                    }
                    
                    // Search field
                    GlassSearchField(text: $searchText, placeholder: "Search...")
                }
                .padding(.horizontal, 20)
                .padding(.top, 20)
                .padding(.bottom, 16)
                
                Divider()
                    .background(.white.opacity(0.1))
                
                // Navigation content
                ScrollView {
                    LazyVStack(spacing: 24) {
                        ForEach(filteredSections, id: \.title) { section in
                            SidebarSectionView(
                                section: section,
                                selectedTab: $selectedTab
                            )
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 20)
                }
                
                Spacer()
                
                // Footer
                VStack(spacing: 12) {
                    Divider()
                        .background(.white.opacity(0.1))
                    
                    HStack {
                        Circle()
                            .fill(.blue.gradient)
                            .frame(width: 32, height: 32)
                            .overlay {
                                Text("JD")
                                    .font(.system(size: 12, weight: .semibold))
                                    .foregroundStyle(.white)
                            }
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("John Doe")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(.primary)
                            
                            Text("john@example.com")
                                .font(.system(size: 12))
                                .foregroundStyle(.secondary)
                        }
                        
                        Spacer()
                        
                        GlassButton(
                            icon: "arrow.right.square",
                            compact: true,
                            action: { /* Logout action */ }
                        )
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 20)
                }
            }
        }
    }
    
    private var filteredSections: [SidebarSection] {
        if searchText.isEmpty {
            return navigationItems
        }
        
        return navigationItems.compactMap { section in
            let filteredItems = section.items.filter { item in
                item.title.localizedCaseInsensitiveContains(searchText)
            }
            
            return filteredItems.isEmpty ? nil : SidebarSection(
                title: section.title,
                items: filteredItems
            )
        }
    }
}

struct SidebarSectionView: View {
    let section: SidebarSection
    @Binding var selectedTab: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(section.title)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
                .padding(.horizontal, 4)
            
            VStack(spacing: 2) {
                ForEach(section.items, id: \.id) { item in
                    SidebarItemView(
                        item: item,
                        isSelected: selectedTab == item.id,
                        action: { selectedTab = item.id }
                    )
                }
            }
        }
    }
}

struct SidebarItemView: View {
    let item: SidebarItem
    let isSelected: Bool
    let action: () -> Void
    
    @State private var isHovered = false
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: item.icon)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(isSelected ? .white : .primary)
                    .frame(width: 20)
                
                Text(item.title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(isSelected ? .white : .primary)
                
                Spacer()
                
                if let badge = item.badge {
                    Text(badge)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(isSelected ? .white.opacity(0.9) : .white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background {
                            Capsule()
                                .fill(isSelected ? .white.opacity(0.2) : .blue)
                        }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background {
                if isSelected {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.blue.gradient)
                        .overlay {
                            RoundedRectangle(cornerRadius: 12)
                                .strokeBorder(.white.opacity(0.3), lineWidth: 1)
                        }
                } else if isHovered {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.ultraThinMaterial)
                        .overlay {
                            RoundedRectangle(cornerRadius: 12)
                                .strokeBorder(.white.opacity(0.2), lineWidth: 1)
                        }
                }
            }
            .scaleEffect(isHovered && !isSelected ? 1.02 : 1.0)
            .shadow(
                color: isSelected ? .blue.opacity(0.3) : .clear,
                radius: isSelected ? 8 : 0,
                x: 0,
                y: isSelected ? 4 : 0
            )
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.2)) {
                isHovered = hovering
            }
        }
        .animation(.easeInOut(duration: 0.2), value: isSelected)
    }
}

// MARK: - Data Models
struct SidebarSection {
    let title: String
    let items: [SidebarItem]
}

struct SidebarItem {
    let id: String
    let icon: String
    let title: String
    let badge: String?
}