import SwiftUI

struct WidgetContainer<Content: View>: View {
    let theme: ThemeData
    let themeType: ThemeType
    let content: Content

    init(theme: ThemeData, themeType: ThemeType, @ViewBuilder content: () -> Content) {
        self.theme = theme
        self.themeType = themeType
        self.content = content()
    }

    var body: some View {
        ZStack {
            theme.colors.widgetBg
                .cornerRadius(theme.cornerRadius)

            content
                .padding(20)

            // Theme-specific decorations
            decorationOverlay
        }
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius)
                .stroke(theme.colors.accent, lineWidth: theme.borderWidth)
        )
        .if(themeType == .pulse) { view in
            view.background(.ultraThinMaterial)
                .cornerRadius(theme.cornerRadius)
                .shadow(color: theme.colors.accent.opacity(0.1), radius: 25, x: 0, y: 0)
                .overlay(
                    RoundedRectangle(cornerRadius: theme.cornerRadius)
                        .fill(theme.colors.accent.opacity(0.03))
                        .blendMode(.screen)
                )
        }
    }

    @ViewBuilder
    private var decorationOverlay: some View {
        switch themeType {
        case .alchemist:
            // Corner flourish (top-right)
            VStack {
                HStack {
                    Spacer()
                    Path { path in
                        path.move(to: CGPoint(x: 0, y: 10))
                        path.addLine(to: CGPoint(x: 0, y: 0))
                        path.addLine(to: CGPoint(x: 10, y: 0))
                    }
                    .stroke(theme.colors.accent, lineWidth: 2)
                    .frame(width: 10, height: 10)
                    .padding([.top, .trailing], 2)
                }
                Spacer()
            }
        case .construct:
            // Crosshairs in corners
            VStack {
                HStack {
                    Text("+")
                        .font(.system(size: 10))
                        .foregroundColor(theme.colors.accent)
                        .padding([.leading, .top], 5)
                    Spacer()
                }
                Spacer()
                HStack {
                    Spacer()
                    Text("+")
                        .font(.system(size: 10))
                        .foregroundColor(theme.colors.accent)
                        .padding([.trailing, .bottom], 5)
                }
            }
        case .pulse:
            EmptyView()
        }
    }
}

struct BarChartWidget: View {
    let theme: ThemeData
    let themeType: ThemeType
    @State private var barHeights: [CGFloat]

    init(theme: ThemeData, themeType: ThemeType) {
        self.theme = theme
        self.themeType = themeType
        self._barHeights = State(initialValue: (0..<15).map { _ in CGFloat.random(in: 0.2...1.0) })
    }

    var body: some View {
        WidgetContainer(theme: theme, themeType: themeType) {
            VStack(alignment: .leading, spacing: 15) {
                HStack {
                    Text("Token Frequency")
                        .font(.system(size: 12))
                        .foregroundColor(theme.colors.textSecondary)
                        .textCase(.uppercase)

                    Spacer()

                    Text("LIVE")
                        .font(.system(size: 12))
                        .foregroundColor(theme.colors.textSecondary)
                        .textCase(.uppercase)
                }

                HStack(alignment: .bottom, spacing: 5) {
                    ForEach(0..<15, id: \.self) { index in
                        barView(for: index)
                    }
                }
                .frame(height: 80)
            }
        }
        .onChange(of: themeType) { _, _ in
            withAnimation {
                barHeights = (0..<15).map { _ in CGFloat.random(in: 0.2...1.0) }
            }
        }
    }

    @ViewBuilder
    private func barView(for index: Int) -> some View {
        let height = 80 * barHeights[index]

        switch themeType {
        case .alchemist:
            // Gradient bars for Alchemist
            BarShape(theme: theme, themeType: themeType)
                .fill(
                    LinearGradient(
                        gradient: Gradient(colors: [theme.colors.accent, Color.clear]),
                        startPoint: .bottom,
                        endPoint: .top
                    )
                )
                .opacity(0.7)
                .frame(height: height)
                .animation(.easeOut(duration: 0.5).delay(Double(index) * 0.02), value: barHeights)

        case .construct:
            // Sharp bars with top border for Construct
            BarShape(theme: theme, themeType: themeType)
                .fill(theme.colors.accent.opacity(0.5))
                .overlay(
                    Rectangle()
                        .fill(theme.colors.textPrimary)
                        .frame(height: 2),
                    alignment: .top
                )
                .frame(height: height)
                .animation(.easeOut(duration: 0.5).delay(Double(index) * 0.02), value: barHeights)

        case .pulse:
            // Glowing bars for Pulse
            BarShape(theme: theme, themeType: themeType)
                .fill(theme.colors.accent.opacity(0.7))
                .shadow(color: theme.colors.accent, radius: 5, x: 0, y: 0)
                .frame(height: height)
                .animation(.easeOut(duration: 0.5).delay(Double(index) * 0.02), value: barHeights)
        }
    }
}

struct BarShape: Shape {
    let theme: ThemeData
    let themeType: ThemeType

    func path(in rect: CGRect) -> Path {
        var path = Path()

        switch themeType {
        case .alchemist:
            path.addRoundedRect(
                in: rect,
                cornerSize: CGSize(width: 5, height: 5),
                style: .continuous
            )
        case .construct:
            path.addRect(rect)
        case .pulse:
            path.addRoundedRect(
                in: rect,
                cornerSize: CGSize(width: 10, height: 10),
                style: .continuous
            )
        }

        return path
    }
}

struct BigStatWidget: View {
    let theme: ThemeData
    let themeType: ThemeType
    let stat: String

    var body: some View {
        WidgetContainer(theme: theme, themeType: themeType) {
            VStack(alignment: .leading, spacing: 15) {
                Text("Pass Rate")
                    .font(.system(size: 12))
                    .foregroundColor(theme.colors.textSecondary)
                    .textCase(.uppercase)

                Spacer()

                Text(stat)
                    .font(fontForTheme)
                    .foregroundColor(theme.colors.textPrimary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    var fontForTheme: Font {
        switch theme.fontStyle {
        case .serif:
            return .system(size: 48, design: .serif)
        case .mono:
            return .system(size: 48, design: .monospaced)
        case .system:
            return .system(size: 48, design: .default)
        }
    }
}

struct ArtifactsWidget: View {
    let theme: ThemeData
    let themeType: ThemeType
    let items: [String]

    var body: some View {
        WidgetContainer(theme: theme, themeType: themeType) {
            VStack(alignment: .leading, spacing: 15) {
                Text("Artifacts")
                    .font(.system(size: 12))
                    .foregroundColor(theme.colors.textSecondary)
                    .textCase(.uppercase)

                VStack(spacing: 10) {
                    ForEach(items, id: \.self) { item in
                        HStack {
                            Text(item)
                                .font(fontForTheme)
                                .foregroundColor(theme.colors.textPrimary)

                            Spacer()

                            Text("Edit")
                                .font(fontForTheme)
                                .foregroundColor(theme.colors.textPrimary)
                        }
                        .padding(.bottom, 5)
                        .overlay(
                            Rectangle()
                                .fill(Color.white.opacity(0.05))
                                .frame(height: 1),
                            alignment: .bottom
                        )
                    }
                }

                Spacer()
            }
        }
    }

    var fontForTheme: Font {
        switch theme.fontStyle {
        case .serif:
            return .system(size: 14, design: .serif)
        case .mono:
            return .system(size: 14, design: .monospaced)
        case .system:
            return .system(size: 14, design: .default)
        }
    }
}

struct SystemLoadWidget: View {
    let theme: ThemeData
    let themeType: ThemeType

    var body: some View {
        WidgetContainer(theme: theme, themeType: themeType) {
            VStack(alignment: .leading, spacing: 15) {
                Text("System Load")
                    .font(.system(size: 12))
                    .foregroundColor(theme.colors.textSecondary)
                    .textCase(.uppercase)

                VStack(spacing: 10) {
                    StatRow(label: "GPU 0", value: "4%", theme: theme)
                    StatRow(label: "GPU 1", value: "12%", theme: theme)
                    StatRow(label: "Mem", value: "1.4GB", theme: theme)
                }

                Spacer()
            }
        }
    }
}

struct StatRow: View {
    let label: String
    let value: String
    let theme: ThemeData

    var body: some View {
        HStack {
            Text(label)
                .font(fontForTheme)
                .foregroundColor(theme.colors.textPrimary)

            Spacer()

            Text(value)
                .font(fontForTheme)
                .foregroundColor(theme.colors.textPrimary)
        }
        .padding(.bottom, 5)
        .overlay(
            Rectangle()
                .fill(Color.white.opacity(0.05))
                .frame(height: 1),
            alignment: .bottom
        )
    }

    var fontForTheme: Font {
        switch theme.fontStyle {
        case .serif:
            return .system(size: 14, design: .serif)
        case .mono:
            return .system(size: 14, design: .monospaced)
        case .system:
            return .system(size: 14, design: .default)
        }
    }
}

// Helper extension for conditional modifiers
extension View {
    @ViewBuilder func `if`<Content: View>(_ condition: Bool, transform: (Self) -> Content) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
}
