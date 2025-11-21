import SwiftUI

struct Sidebar: View {
    @Binding var currentTheme: ThemeType
    let themeData: ThemeData

    var body: some View {
        ZStack(alignment: .topLeading) {
            // Background
            Color.black.opacity(0.2)
                .background(.ultraThinMaterial)

            VStack(spacing: 40) {
                Spacer()
                    .frame(height: 80)

                // Alchemist Icon
                ThemeIcon(type: .alchemist, isActive: currentTheme == .alchemist)
                    .onHover { isHovering in
                        if isHovering {
                            withAnimation(.easeOut(duration: 0.6)) {
                                currentTheme = .alchemist
                            }
                        }
                    }
                    .onTapGesture {
                        withAnimation(.easeOut(duration: 0.6)) {
                            currentTheme = .alchemist
                        }
                    }

                // Construct Icon
                ThemeIcon(type: .construct, isActive: currentTheme == .construct)
                    .onHover { isHovering in
                        if isHovering {
                            withAnimation(.easeOut(duration: 0.6)) {
                                currentTheme = .construct
                            }
                        }
                    }
                    .onTapGesture {
                        withAnimation(.easeOut(duration: 0.6)) {
                            currentTheme = .construct
                        }
                    }

                // Pulse Icon
                ThemeIcon(type: .pulse, isActive: currentTheme == .pulse)
                    .onHover { isHovering in
                        if isHovering {
                            withAnimation(.easeOut(duration: 0.6)) {
                                currentTheme = .pulse
                            }
                        }
                    }
                    .onTapGesture {
                        withAnimation(.easeOut(duration: 0.6)) {
                            currentTheme = .pulse
                        }
                    }

                Spacer()
            }
            .frame(maxWidth: .infinity)
        }
        .frame(width: 90)
        .overlay(
            Rectangle()
                .fill(Color.white.opacity(0.05))
                .frame(width: 1),
            alignment: .trailing
        )
    }
}

struct ThemeIcon: View {
    let type: ThemeType
    let isActive: Bool

    var body: some View {
        ZStack {
            switch type {
            case .alchemist:
                AlchemistIcon()
            case .construct:
                ConstructIcon()
            case .pulse:
                PulseIcon()
            }
        }
        .frame(width: 50, height: 50)
        .opacity(isActive ? 1.0 : 0.5)
        .scaleEffect(isActive ? 1.1 : 1.0)
        .animation(.easeOut(duration: 0.3), value: isActive)
    }
}

struct AlchemistIcon: View {
    var body: some View {
        ZStack {
            // Cross lines (background)
            Path { path in
                path.move(to: CGPoint(x: 25, y: 5))
                path.addLine(to: CGPoint(x: 25, y: 45))
                path.move(to: CGPoint(x: 5, y: 25))
                path.addLine(to: CGPoint(x: 45, y: 25))
            }
            .stroke(Color(hex: "d4af37").opacity(0.3), lineWidth: 2)

            // Outer circle
            Circle()
                .stroke(Color(hex: "d4af37"), lineWidth: 2)
                .frame(width: 30, height: 30)

            // Inner curved path
            Path { path in
                path.move(to: CGPoint(x: 25, y: 10))
                path.addQuadCurve(
                    to: CGPoint(x: 35, y: 25),
                    control: CGPoint(x: 35, y: 10)
                )
                path.addQuadCurve(
                    to: CGPoint(x: 25, y: 40),
                    control: CGPoint(x: 35, y: 40)
                )
                path.addQuadCurve(
                    to: CGPoint(x: 15, y: 25),
                    control: CGPoint(x: 15, y: 40)
                )
                path.addQuadCurve(
                    to: CGPoint(x: 25, y: 10),
                    control: CGPoint(x: 15, y: 10)
                )
            }
            .stroke(Color(hex: "d4af37"), lineWidth: 2)
        }
    }
}

struct ConstructIcon: View {
    var body: some View {
        ZStack {
            // Main rectangle
            Rectangle()
                .stroke(Color(hex: "00f0ff"), lineWidth: 1)
                .frame(width: 30, height: 30)

            // Top-left corner
            Path { path in
                path.move(to: CGPoint(x: 5, y: 5))
                path.addLine(to: CGPoint(x: 15, y: 5))
                path.move(to: CGPoint(x: 5, y: 5))
                path.addLine(to: CGPoint(x: 5, y: 15))
            }
            .stroke(Color(hex: "00f0ff"), lineWidth: 1)

            // Bottom-right corner
            Path { path in
                path.move(to: CGPoint(x: 45, y: 45))
                path.addLine(to: CGPoint(x: 35, y: 45))
                path.move(to: CGPoint(x: 45, y: 45))
                path.addLine(to: CGPoint(x: 45, y: 35))
            }
            .stroke(Color(hex: "00f0ff"), lineWidth: 1)
        }
    }
}

struct PulseIcon: View {
    var body: some View {
        ZStack {
            // Wave path
            Path { path in
                path.move(to: CGPoint(x: 10, y: 25))
                path.addQuadCurve(
                    to: CGPoint(x: 40, y: 25),
                    control: CGPoint(x: 25, y: 5)
                )
            }
            .stroke(Color(hex: "ff2a68"), lineWidth: 1.5)

            // Circle
            Circle()
                .stroke(Color(hex: "ff2a68"), lineWidth: 1.5)
                .frame(width: 8, height: 8)
                .offset(y: 7)
        }
    }
}
