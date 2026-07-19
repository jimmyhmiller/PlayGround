import SwiftUI

// MARK: - Small shared controls

struct CheckBox: View {
    let on: Bool
    var accent: Color = Th.accent
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                RoundedRectangle(cornerRadius: 4)
                    .fill(on ? accent : Color.clear)
                RoundedRectangle(cornerRadius: 4)
                    .strokeBorder(on ? accent : Color(hex: 0x55555c), lineWidth: 1.5)
                if on {
                    Text("✓")
                        .font(.system(size: 9, weight: .heavy))
                        .foregroundColor(.white)
                }
            }
            .frame(width: 15, height: 15)
        }
        .buttonStyle(.plain)
    }
}

struct Avatar: View {
    let isAI: Bool
    let label: String
    var size: CGFloat = 20
    var accent: Color = Th.accent

    var body: some View {
        ZStack {
            Circle().fill(isAI ? Th.ai : accent)
            Text(isAI ? "✦" : label)
                .font(.system(size: size * 0.42, weight: .bold))
                .foregroundColor(.white)
        }
        .frame(width: size, height: size)
    }
}

struct SegmentedPill<T: Hashable>: View {
    struct Item {
        let value: T
        let label: String
        var activeColor: Color = Th.text
    }

    let items: [Item]
    let selected: T
    let action: (T) -> Void

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                let active = item.value == selected
                Button(action: { action(item.value) }) {
                    Text(item.label)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(active ? item.activeColor : Th.text3)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(active ? Color.white.opacity(0.14) : Color.clear)
                        )
                        .shadow(color: active ? Color.black.opacity(0.3) : .clear, radius: 1, y: 1)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(2)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.white.opacity(0.07)))
    }
}

struct StatusChip: View {
    let status: String

    var body: some View {
        let (fg, bg) = Th.statusColors(status)
        Text(display)
            .font(.system(size: 9.5, weight: .bold, design: .monospaced))
            .foregroundColor(fg)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(RoundedRectangle(cornerRadius: 4).fill(bg))
    }

    private var display: String {
        switch status {
        case "?": return "A"
        default: return status
        }
    }
}

/// Wrapping horizontal layout for saved-reply chips.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var maxX: CGFloat = 0
        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            if x > 0, x + size.width > maxWidth {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            maxX = max(maxX, x)
            rowHeight = max(rowHeight, size.height)
        }
        let width = maxWidth.isFinite ? maxWidth : maxX
        return CGSize(width: width, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0
        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            sub.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

struct ToastView: View {
    let message: String

    var body: some View {
        Text(message)
            .font(.system(size: 12.5, weight: .semibold))
            .foregroundColor(Th.text)
            .padding(.horizontal, 16)
            .padding(.vertical, 9)
            .background(
                Capsule()
                    .fill(Color(hex: 0x0a0a0c, alpha: 0.92))
                    .overlay(Capsule().strokeBorder(Color.white.opacity(0.1), lineWidth: 1))
            )
            .shadow(color: .black.opacity(0.5), radius: 15, y: 8)
            .padding(.bottom, 16)
    }
}
