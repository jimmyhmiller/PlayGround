import SwiftUI

struct ContentView: View {
    @EnvironmentObject var store: AppStore

    var body: some View {
        VStack(spacing: 0) {
            TitleBarView()
            HStack(spacing: 0) {
                SidebarView()
                FileListView()
                mainPane
            }
        }
        .background(Th.bg)
        .overlay(alignment: .bottom) {
            if let toast = store.toast {
                ToastView(message: toast)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.18), value: store.toast)
    }

    @ViewBuilder
    private var mainPane: some View {
        if store.screen == .submitted {
            SubmittedView()
        } else if store.selection == .none {
            EmptyStateView()
        } else {
            DiffView()
        }
    }
}

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 18) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color(hex: 0x3a3a40), lineWidth: 2)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Th.panel))
                    .frame(width: 60, height: 74)
                VStack(alignment: .leading, spacing: 7) {
                    RoundedRectangle(cornerRadius: 2).fill(Color(hex: 0x3a3a40)).frame(width: 36, height: 3)
                    RoundedRectangle(cornerRadius: 2).fill(Color(hex: 0x3a3a40)).frame(width: 28, height: 3)
                    RoundedRectangle(cornerRadius: 2).fill(Color(hex: 0x3a3a40)).frame(width: 32, height: 3)
                }
                .offset(y: -12)
            }
            VStack(spacing: 5) {
                Text("No pull request selected")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(Th.text3)
                Text("Choose a PR from the sidebar to start reviewing the agent's changes.")
                    .font(.system(size: 13))
                    .foregroundColor(Th.dimmer)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 340)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Th.bg)
    }
}

struct SubmittedView: View {
    @EnvironmentObject var store: AppStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 11) {
                    ZStack {
                        Circle().fill(Color(hex: 0x28c840))
                        Text("✓")
                            .font(.system(size: 13, weight: .heavy))
                            .foregroundColor(Color(hex: 0x0b2010))
                    }
                    .frame(width: 26, height: 26)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Review summary copied to clipboard")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(Th.text)
                        Text("Paste it into any agent chat to hand off your feedback.")
                            .font(.system(size: 12.5))
                            .foregroundColor(Th.dim)
                    }
                }
                .padding(.bottom, 18)

                ScrollView(.horizontal) {
                    Text(store.summarySnapshot)
                        .font(.system(size: 12.5, design: .monospaced))
                        .lineSpacing(4)
                        .foregroundColor(Th.codeText)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 18)
                }
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Th.cardDark)
                        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.white.opacity(0.08), lineWidth: 1))
                )

                HStack(spacing: 10) {
                    Button(action: { store.copySummary() }) {
                        Text("Copy again")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 17)
                            .padding(.vertical, 8)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Th.accent))
                    }
                    .buttonStyle(.plain)

                    Button(action: { store.backToReview() }) {
                        Text("Back to review")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(Th.text)
                            .padding(.horizontal, 17)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.white.opacity(0.06))
                                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.white.opacity(0.14), lineWidth: 1))
                            )
                    }
                    .buttonStyle(.plain)

                    if store.isPR {
                        Button(action: { store.submitToGitHub() }) {
                            HStack(spacing: 6) {
                                Image(systemName: "paperplane.fill")
                                    .font(.system(size: 11))
                                Text("Submit review to GitHub")
                                    .font(.system(size: 13, weight: .semibold))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 17)
                            .padding(.vertical, 8)
                            .background(RoundedRectangle(cornerRadius: 8).fill(verdictColor))
                        }
                        .buttonStyle(.plain)
                        .help("Posts this summary as a \(store.verdict.rawValue) review on the PR")
                    }
                }
                .padding(.top, 16)
            }
            .frame(maxWidth: 700)
            .padding(.horizontal, 24)
            .padding(.vertical, 36)
            .frame(maxWidth: .infinity)
        }
        .background(Th.bg)
    }

    private var verdictColor: Color {
        switch store.verdict {
        case .approve: return Color(hex: 0x1f883d)
        case .requestChanges: return Color(hex: 0xb0342c)
        case .comment: return Color(hex: 0x444a52)
        }
    }
}
