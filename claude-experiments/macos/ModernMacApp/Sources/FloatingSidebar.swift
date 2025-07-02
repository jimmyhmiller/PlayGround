import SwiftUI

struct FloatingSidebar: View {
    @Binding var selectedTab: String
    @State private var searchText = ""
    
    let navigationSections = [
        FloatingSidebarSection(title: "Main", items: [
            FloatingSidebarItem(id: "home", icon: "house.fill", title: "Home", subtitle: "Dashboard"),
            FloatingSidebarItem(id: "projects", icon: "folder.fill", title: "Projects", subtitle: "5 active"),
            FloatingSidebarItem(id: "favorites", icon: "heart.fill", title: "Favorites", subtitle: "Quick access"),
        ]),
        FloatingSidebarSection(title: "Tools", items: [
            FloatingSidebarItem(id: "analytics", icon: "chart.bar.fill", title: "Analytics", subtitle: "Usage stats"),
            FloatingSidebarItem(id: "notifications", icon: "bell.fill", title: "Notifications", subtitle: "3 new"),
            FloatingSidebarItem(id: "settings", icon: "gear", title: "Settings", subtitle: "Preferences"),
        ])
    ]
    
    var body: some View {
        VStack(spacing: 0) {
            // Floating header with enhanced design
            VStack(spacing: 20) {
                HStack {
                    // App icon with gradient
                    ZStack {
                        Circle()
                            .fill(.blue.gradient)
                            .frame(width: 44, height: 44)
                            .shadow(color: .blue.opacity(0.3), radius: 8, x: 0, y: 4)
                        
                        Image(systemName: "sparkles")
                            .font(.system(size: 18, weight: .medium))
                            .foregroundStyle(.white)
                    }
                    
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Modern")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundStyle(.primary)
                        
                        Text("Glass UI")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    
                    Spacer()
                    
                    // Status indicator
                    HStack(spacing: 4) {
                        Circle()
                            .fill(.green)
                            .frame(width: 6, height: 6)
                        
                        Text("Online")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                
                // Enhanced search field
                EnhancedGlassSearchField(text: $searchText, placeholder: "Search everything...")
            }
            .padding(24)
            .background {
                RoundedRectangle(cornerRadius: 20)
                    .fill(.regularMaterial)
                    .overlay {
                        RoundedRectangle(cornerRadius: 20)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [.white.opacity(0.3), .white.opacity(0.1)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    }
                    .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
            }
            .padding(.horizontal, 16)
            .padding(.top, 20)
            
            // Navigation content with enhanced styling
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(filteredSections, id: \.title) { section in
                        FloatingSidebarSectionView(
                            section: section,
                            selectedTab: $selectedTab
                        )
                    }
                    
                    // Quick actions section
                    QuickActionsSection()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 20)
            }
            
            Spacer()
            
            // Enhanced footer with user info
            UserFooterSection()
                .padding(.horizontal, 16)
                .padding(.bottom, 20)
        }
        .frame(width: 300)
        .background(.clear)
    }
    
    private var filteredSections: [FloatingSidebarSection] {
        if searchText.isEmpty {
            return navigationSections
        }
        
        return navigationSections.compactMap { section in
            let filteredItems = section.items.filter { item in
                item.title.localizedCaseInsensitiveContains(searchText) ||
                (item.subtitle?.localizedCaseInsensitiveContains(searchText) ?? false)
            }
            
            return filteredItems.isEmpty ? nil : FloatingSidebarSection(
                title: section.title,
                items: filteredItems
            )
        }
    }
}

struct FloatingSidebarSectionView: View {
    let section: FloatingSidebarSection
    @Binding var selectedTab: String
    
    var body: some View {
        VStack(spacing: 8) {
            ForEach(section.items, id: \.id) { item in
                FloatingSidebarItemView(
                    item: item,
                    isSelected: selectedTab == item.id,
                    action: { selectedTab = item.id }
                )
            }
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 18)
                .fill(.ultraThinMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 18)
                        .strokeBorder(.white.opacity(0.1), lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.05), radius: 10, x: 0, y: 4)
        }
    }
}

struct FloatingSidebarItemView: View {
    let item: FloatingSidebarItem
    let isSelected: Bool
    let action: () -> Void
    
    @State private var isHovered = false
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                // Icon with background
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(isSelected ? .white.opacity(0.2) : .white.opacity(isHovered ? 0.1 : 0.05))
                        .frame(width: 36, height: 36)
                    
                    Image(systemName: item.icon)
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(isSelected ? .white : .primary)
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.title)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(isSelected ? .white : .primary)
                    
                    if let subtitle = item.subtitle {
                        Text(subtitle)
                            .font(.system(size: 12))
                            .foregroundStyle(isSelected ? .white.opacity(0.8) : .secondary)
                    }
                }
                
                Spacer()
                
                if isSelected {
                    Circle()
                        .fill(.white.opacity(0.8))
                        .frame(width: 6, height: 6)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background {
                if isSelected {
                    RoundedRectangle(cornerRadius: 14)
                        .fill(.blue.gradient)
                        .shadow(color: .blue.opacity(0.4), radius: 12, x: 0, y: 6)
                } else if isHovered {
                    RoundedRectangle(cornerRadius: 14)
                        .fill(.white.opacity(0.08))
                }
            }
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.2)) {
                isHovered = hovering
            }
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: isSelected)
    }
}

struct QuickActionsSection: View {
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Quick Actions")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                
                Spacer()
            }
            .padding(.horizontal, 4)
            
            HStack(spacing: 12) {
                QuickActionButton(
                    icon: "plus.circle.fill",
                    label: "New",
                    color: .blue,
                    action: { /* New action */ }
                )
                
                QuickActionButton(
                    icon: "square.and.arrow.up",
                    label: "Share",
                    color: .green,
                    action: { /* Share action */ }
                )
                
                QuickActionButton(
                    icon: "trash.fill",
                    label: "Delete",
                    color: .red,
                    action: { /* Delete action */ }
                )
            }
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 18)
                .fill(.ultraThinMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 18)
                        .strokeBorder(.white.opacity(0.1), lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.05), radius: 10, x: 0, y: 4)
        }
    }
}

struct QuickActionButton: View {
    let icon: String
    let label: String
    let color: Color
    let action: () -> Void
    
    @State private var isPressed = false
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.white)
                    .frame(width: 40, height: 40)
                    .background {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(color.gradient)
                    }
                
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.plain)
        .scaleEffect(isPressed ? 0.95 : 1.0)
        .onLongPressGesture(minimumDuration: 0, maximumDistance: .infinity, pressing: { pressing in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = pressing
            }
        }, perform: {
            // Long press action
        })
    }
}

struct UserFooterSection: View {
    var body: some View {
        VStack(spacing: 16) {
            Divider()
                .background(.white.opacity(0.1))
            
            HStack(spacing: 12) {
                // User avatar with status
                ZStack(alignment: .bottomTrailing) {
                    Circle()
                        .fill(.blue.gradient)
                        .frame(width: 40, height: 40)
                        .overlay {
                            Text("JD")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundStyle(.white)
                        }
                    
                    Circle()
                        .fill(.green)
                        .frame(width: 12, height: 12)
                        .overlay {
                            Circle()
                                .strokeBorder(.white, lineWidth: 2)
                        }
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("John Doe")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(.primary)
                    
                    Text("john@example.com")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                GlassButton(
                    icon: "ellipsis",
                    compact: true,
                    action: { /* User menu */ }
                )
            }
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 18)
                .fill(.regularMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 18)
                        .strokeBorder(.white.opacity(0.1), lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.08), radius: 12, x: 0, y: 6)
        }
    }
}

struct EnhancedGlassSearchField: View {
    @Binding var text: String
    let placeholder: String
    
    @State private var isFocused = false
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
                .font(.system(size: 16, weight: .medium))
            
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 15))
                .onTapGesture {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isFocused = true
                    }
                }
            
            if !text.isEmpty {
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        text = ""
                    }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                        .font(.system(size: 14))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background {
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(
                            isFocused ? .blue.opacity(0.5) : .white.opacity(0.2),
                            lineWidth: isFocused ? 2 : 1
                        )
                }
        }
        .shadow(
            color: isFocused ? .blue.opacity(0.2) : .black.opacity(0.05),
            radius: isFocused ? 8 : 4,
            x: 0,
            y: 2
        )
    }
}

// MARK: - Data Models
struct FloatingSidebarSection {
    let title: String
    let items: [FloatingSidebarItem]
}

struct FloatingSidebarItem {
    let id: String
    let icon: String
    let title: String
    let subtitle: String?
    
    init(id: String, icon: String, title: String, subtitle: String? = nil) {
        self.id = id
        self.icon = icon
        self.title = title
        self.subtitle = subtitle
    }
}