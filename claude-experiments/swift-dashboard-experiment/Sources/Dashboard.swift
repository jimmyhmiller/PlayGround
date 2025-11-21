import SwiftUI

struct Dashboard: View {
    let theme: ThemeData
    let themeType: ThemeType

    var body: some View {
        ZStack {
            // Background
            theme.colors.bgApp
                .ignoresSafeArea()

            // Background Pattern (subtle)
            BackgroundPattern(themeType: themeType, accent: theme.colors.accent)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 40) {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text(theme.subtitle)
                        .font(fontForBody(size: 11))
                        .foregroundColor(theme.colors.accent)
                        .textCase(.uppercase)
                        .tracking(2)

                    Text(theme.title)
                        .font(fontForTitle)
                        .foregroundColor(theme.colors.textPrimary)
                        .fontWeight(.ultraLight)
                        .tracking(1)
                }

                // Widgets Grid
                VStack(spacing: 20) {
                    // Row 1: Chart (full width)
                    BarChartWidget(theme: theme, themeType: themeType)
                        .frame(height: 150)

                    // Row 2 & 3: Two columns
                    HStack(alignment: .top, spacing: 20) {
                        // Left column
                        VStack(spacing: 20) {
                            BigStatWidget(theme: theme, themeType: themeType, stat: theme.stat)
                                .frame(height: 150)

                            SystemLoadWidget(theme: theme, themeType: themeType)
                                .frame(height: 150)
                        }
                        .frame(maxWidth: .infinity)

                        // Right column (taller widget)
                        ArtifactsWidget(theme: theme, themeType: themeType, items: theme.items)
                            .frame(maxWidth: .infinity)
                            .frame(height: 320)
                    }
                }

                Spacer()
            }
            .padding(50)
        }
    }

    var fontForTitle: Font {
        switch theme.fontStyle {
        case .serif:
            return .system(size: 40, design: .serif)
        case .mono:
            return .system(size: 40, design: .monospaced)
        case .system:
            return .system(size: 40, design: .default)
        }
    }

    func fontForBody(size: CGFloat) -> Font {
        switch theme.fontStyle {
        case .serif:
            return .system(size: size, design: .serif)
        case .mono:
            return .system(size: size, design: .monospaced)
        case .system:
            return .system(size: size, design: .default)
        }
    }
}

struct BackgroundPattern: View {
    let themeType: ThemeType
    let accent: Color

    var body: some View {
        switch themeType {
        case .alchemist:
            // Subtle vines pattern
            Color.clear
                .background(
                    Image(systemName: "leaf.fill")
                        .resizable()
                        .renderingMode(.template)
                        .foregroundColor(accent.opacity(0.02))
                )
        case .construct:
            // Technical grid
            GridPattern(color: accent)
                .opacity(0.1)
        case .pulse:
            // Radial gradient blob
            RadialGradient(
                gradient: Gradient(colors: [
                    accent.opacity(0.15),
                    Color.clear
                ]),
                center: .topTrailing,
                startRadius: 0,
                endRadius: 400
            )
        }
    }
}

struct GridPattern: View {
    let color: Color

    var body: some View {
        GeometryReader { geometry in
            Path { path in
                let spacing: CGFloat = 40
                let width = geometry.size.width
                let height = geometry.size.height

                // Vertical lines
                var x: CGFloat = 0
                while x <= width {
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: height))
                    x += spacing
                }

                // Horizontal lines
                var y: CGFloat = 0
                while y <= height {
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: width, y: y))
                    y += spacing
                }
            }
            .stroke(color, lineWidth: 1)
        }
    }
}

