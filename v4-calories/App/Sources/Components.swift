import SwiftUI

/// Thin horizontal divider used between sections.
struct SectionDivider: View {
    var body: some View {
        Rectangle().fill(Theme.hair).frame(height: 0.5).padding(.vertical, 14)
    }
}

/// Tracked monospace caption used as a section header (e.g. "CUMULATIVE DEFICIT").
struct CapsLabel: View {
    let text: String
    var color: Color = Theme.textDim(0.4)
    var body: some View {
        Text(text)
            .font(.mono(10, .regular))
            .tracking(1.5)
            .foregroundStyle(color)
    }
}

/// A small labelled metric ("PROJECTED LOSS" over a value).
struct StatBlock: View {
    let label: String
    let value: String
    var color: Color = Theme.text
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.mono(9)).tracking(1).foregroundStyle(Theme.textDim(0.4))
            Text(value).font(.mono(24, .bold)).foregroundStyle(color)
        }
    }
}

/// Horizontal progress bar (rounded), used for the calorie budget.
struct BarProgress: View {
    let fraction: Double
    var color: Color = Theme.green
    var height: CGFloat = 3
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.08))
                Capsule().fill(color)
                    .frame(width: max(0, min(1, fraction)) * geo.size.width)
            }
        }
        .frame(height: height)
    }
}

extension View {
    /// Standard horizontal page padding from the design.
    func pagePadding() -> some View { self.padding(.horizontal, 20) }
}
