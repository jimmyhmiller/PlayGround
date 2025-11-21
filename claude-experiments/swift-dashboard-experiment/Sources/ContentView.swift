import SwiftUI

struct ContentView: View {
    @State private var currentTheme: ThemeType = .alchemist

    var themeData: ThemeData {
        ThemeData.forType(currentTheme)
    }

    var body: some View {
        HStack(spacing: 0) {
            Sidebar(currentTheme: $currentTheme, themeData: themeData)

            Dashboard(theme: themeData, themeType: currentTheme)
        }
        .background(Color.black.ignoresSafeArea())
        .cornerRadius(themeData.cornerRadius)
        .overlay(
            RoundedRectangle(cornerRadius: themeData.cornerRadius)
                .stroke(themeData.colors.borderColor, lineWidth: 1)
        )
        // Add inner glow for Pulse theme
        .overlay(
            Group {
                if currentTheme == .pulse {
                    RoundedRectangle(cornerRadius: themeData.cornerRadius)
                        .strokeBorder(
                            themeData.colors.accent.opacity(0.1),
                            lineWidth: 25
                        )
                        .blur(radius: 15)
                }
            }
        )
        .shadow(color: .black.opacity(0.9), radius: 50, x: 0, y: 50)
        .animation(.easeOut(duration: 0.6), value: currentTheme)
        .edgesIgnoringSafeArea(.all)
    }
}

#Preview {
    ContentView()
        .frame(width: 1050, height: 720)
}
