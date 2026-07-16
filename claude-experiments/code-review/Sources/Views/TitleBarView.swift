import SwiftUI

struct TitleBarView: View {
    @EnvironmentObject var store: AppStore

    var body: some View {
        ZStack {
            WindowDragArea()
            HStack(spacing: 14) {
                // Space for the standard traffic-light buttons.
                Spacer().frame(width: 64)

                VStack(alignment: .leading, spacing: 1) {
                    Text(store.currentProject?.name ?? "Agent Review")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(Th.text)
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundColor(Th.dim)
                        .lineLimit(1)
                }

                if let pr = store.currentPR {
                    ChecksBadge(pr: pr)
                }

                Spacer()

                if store.selection != .none {
                    Text("VERDICT")
                        .font(.system(size: 10, weight: .semibold))
                        .kerning(0.5)
                        .foregroundColor(Color(hex: 0x7a7a80))

                    SegmentedPill(
                        items: Verdict.allCases.map { v in
                            SegmentedPill.Item(
                                value: v,
                                label: v.rawValue,
                                activeColor: v == .requestChanges ? Th.redSoft : v == .approve ? Th.greenSoft : Th.text
                            )
                        },
                        selected: store.verdict,
                        action: { store.setVerdict($0) }
                    )

                    Button(action: { store.copySummary() }) {
                        Text("Copy review summary")
                            .font(.system(size: 12.5, weight: .semibold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 15)
                            .padding(.vertical, 6)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Th.accent))
                            .shadow(color: .black.opacity(0.4), radius: 1.5, y: 1)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 16)
        }
        .frame(height: 46)
        .background(Th.titlebar)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Th.border).frame(height: 1)
        }
    }

    private var subtitle: String {
        switch store.selection {
        case .none:
            return "No pull request selected"
        case .pr:
            if let pr = store.currentPR {
                return "#\(pr.number) · \(pr.title)"
            }
            return ""
        case .workingTree:
            let n = store.changeEntries.count + store.stagedEntries.count
            return "working tree · \(n) file\(n == 1 ? "" : "s") changed"
        }
    }
}

struct ChecksBadge: View {
    let pr: PullRequest

    var body: some View {
        HStack(spacing: 8) {
            HStack(spacing: 4) {
                Circle().fill(dotColor).frame(width: 7, height: 7)
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Th.text3)
            }
            if pr.checks.total > 0 {
                Text(checkText)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(checksColor)
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 3)
        .background(Capsule().fill(Color.white.opacity(0.06)))
    }

    private var label: String { pr.statusLabel }

    private var dotColor: Color {
        switch pr.statusLabel {
        case "merged": return Th.purple
        case "closed": return Th.red
        case "approved": return Th.green
        case "changes requested": return Th.red
        case "draft": return Th.faint
        default: return Th.yellow
        }
    }

    private var checksColor: Color {
        if pr.checks.failed > 0 { return Th.redSoft }
        if pr.checks.pending > 0 { return Th.yellow }
        return Th.greenSoft
    }

    private var checkText: String {
        if pr.checks.failed > 0 { return "✕ \(pr.checks.failed)/\(pr.checks.total) checks" }
        if pr.checks.pending > 0 { return "● \(pr.checks.passed)/\(pr.checks.total) checks" }
        return "✓ \(pr.checks.total) checks"
    }
}
