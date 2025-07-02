import SwiftUI

// MARK: - Glass Button Component
struct GlassButton: View {
    let icon: String
    let title: String?
    let compact: Bool
    let tinted: Bool
    let action: () -> Void
    
    @State private var isHovered = false
    
    init(
        icon: String,
        title: String? = nil,
        compact: Bool = false,
        tinted: Bool = false,
        action: @escaping () -> Void
    ) {
        self.icon = icon
        self.title = title
        self.compact = compact
        self.tinted = tinted
        self.action = action
    }
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: compact ? 0 : 8) {
                Image(systemName: icon)
                    .font(compact ? .system(size: 14, weight: .medium) : .system(size: 16, weight: .medium))
                
                if let title = title, !compact {
                    Text(title)
                        .font(.system(size: 14, weight: .medium))
                }
            }
            .foregroundStyle(tinted ? .white : .primary)
            .padding(.horizontal, compact ? 8 : 12)
            .padding(.vertical, compact ? 6 : 8)
            .background {
                if tinted {
                    RoundedRectangle(cornerRadius: compact ? 8 : 12)
                        .fill(.blue.gradient)
                        .overlay(
                            RoundedRectangle(cornerRadius: compact ? 8 : 12)
                                .strokeBorder(.white.opacity(0.3), lineWidth: 1)
                        )
                } else {
                    RoundedRectangle(cornerRadius: compact ? 8 : 12)
                        .fill(.ultraThinMaterial)
                        .overlay(
                            RoundedRectangle(cornerRadius: compact ? 8 : 12)
                                .strokeBorder(.white.opacity(isHovered ? 0.4 : 0.2), lineWidth: 1)
                        )
                }
            }
            .scaleEffect(isHovered ? 1.05 : 1.0)
            .shadow(
                color: tinted ? .blue.opacity(0.3) : .black.opacity(0.1),
                radius: isHovered ? 8 : 4,
                x: 0,
                y: isHovered ? 4 : 2
            )
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.2)) {
                isHovered = hovering
            }
        }
    }
}

// MARK: - Glass Tab Button
struct GlassTabButton: View {
    let icon: String
    let title: String
    let isSelected: Bool
    let action: () -> Void
    
    @State private var isHovered = false
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 14, weight: .medium))
                
                Text(title)
                    .font(.system(size: 13, weight: .medium))
            }
            .foregroundStyle(isSelected ? .white : .primary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background {
                if isSelected {
                    Capsule()
                        .fill(.blue.gradient)
                        .overlay(
                            Capsule()
                                .strokeBorder(.white.opacity(0.3), lineWidth: 1)
                        )
                } else {
                    Capsule()
                        .fill(.ultraThinMaterial.opacity(isHovered ? 1.0 : 0.5))
                        .overlay(
                            Capsule()
                                .strokeBorder(.white.opacity(isHovered ? 0.3 : 0.1), lineWidth: 1)
                        )
                }
            }
            .scaleEffect(isHovered && !isSelected ? 1.02 : 1.0)
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
        .animation(.easeInOut(duration: 0.2), value: isSelected)
    }
}

// MARK: - Glass Card Container
struct GlassCard<Content: View>: View {
    let content: Content
    @State private var isHovered = false
    
    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }
    
    var body: some View {
        content
            .background {
                RoundedRectangle(cornerRadius: 16)
                    .fill(.ultraThinMaterial)
                    .overlay {
                        RoundedRectangle(cornerRadius: 16)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [
                                        .white.opacity(isHovered ? 0.4 : 0.2),
                                        .white.opacity(isHovered ? 0.2 : 0.1)
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    }
            }
            .shadow(
                color: .black.opacity(isHovered ? 0.15 : 0.08),
                radius: isHovered ? 20 : 10,
                x: 0,
                y: isHovered ? 8 : 4
            )
            .scaleEffect(isHovered ? 1.02 : 1.0)
            .onHover { hovering in
                withAnimation(.easeInOut(duration: 0.3)) {
                    isHovered = hovering
                }
            }
    }
}

// MARK: - Glass Panel Container
struct GlassPanel<Content: View>: View {
    let content: Content
    let cornerRadius: CGFloat
    
    init(cornerRadius: CGFloat = 20, @ViewBuilder content: () -> Content) {
        self.cornerRadius = cornerRadius
        self.content = content()
    }
    
    var body: some View {
        content
            .background {
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(.regularMaterial)
                    .overlay {
                        RoundedRectangle(cornerRadius: cornerRadius)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [
                                        .white.opacity(0.3),
                                        .white.opacity(0.1),
                                        .clear
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    }
            }
            .shadow(color: .black.opacity(0.1), radius: 15, x: 0, y: 5)
    }
}

// MARK: - Glass Search Field
struct GlassSearchField: View {
    @Binding var text: String
    let placeholder: String
    
    @State private var isFocused = false
    
    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
                .font(.system(size: 14))
            
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 14))
                .onTapGesture {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isFocused = true
                    }
                }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
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
            radius: isFocused ? 8 : 2,
            x: 0,
            y: 2
        )
    }
}