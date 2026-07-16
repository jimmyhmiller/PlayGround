import SwiftUI

struct FileListView: View {
    @EnvironmentObject var store: AppStore

    var body: some View {
        Group {
            if store.selection == .none {
                Text("Select a pull request or a working tree to see its files.")
                    .font(.system(size: 12.5))
                    .foregroundColor(Th.dimmer)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 44)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        if store.isPR, let pr = store.currentPR {
                            PRMetaBox(pr: pr)
                        }

                        HStack(alignment: .firstTextBaseline) {
                            Text(store.isWorkingTree ? "CHANGES" : "FILES")
                                .font(.system(size: 11, weight: .semibold))
                                .kerning(0.5)
                                .foregroundColor(Th.dimmer)
                            Spacer()
                            Text(store.isWorkingTree ? "git add -p" : "\(store.changeEntries.count) changed")
                                .font(.system(size: 10.5))
                                .foregroundColor(Th.faint)
                        }
                        .padding(.horizontal, 14)
                        .padding(.top, 12)
                        .padding(.bottom, 6)

                        ForEach(store.changeEntries) { entry in
                            FileRow(entry: entry)
                        }

                        if store.changeEntries.isEmpty {
                            Text(store.isWorkingTree ? "nothing unstaged" : "no files")
                                .font(.system(size: 11.5))
                                .foregroundColor(Th.faint)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 4)
                        }

                        if store.isWorkingTree {
                            HStack(alignment: .firstTextBaseline) {
                                Text("STAGED")
                                    .font(.system(size: 11, weight: .semibold))
                                    .kerning(0.5)
                                    .foregroundColor(Th.green)
                                Spacer()
                                Text("\(store.stagedEntries.count) file\(store.stagedEntries.count == 1 ? "" : "s")")
                                    .font(.system(size: 10.5))
                                    .foregroundColor(Th.faint)
                            }
                            .padding(.horizontal, 14)
                            .padding(.top, 14)
                            .padding(.bottom, 6)
                            .overlay(alignment: .top) {
                                Rectangle().fill(Th.border).frame(height: 1).padding(.top, 6)
                            }

                            ForEach(store.stagedEntries) { entry in
                                FileRow(entry: entry)
                            }

                            if store.stagedEntries.isEmpty {
                                Text("nothing staged yet")
                                    .font(.system(size: 11.5))
                                    .foregroundColor(Th.faint)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 4)
                            }
                        }
                    }
                    .padding(.bottom, 20)
                }
            }
        }
        .frame(width: 252)
        .frame(maxHeight: .infinity)
        .background(Th.bgHeader)
        .overlay(alignment: .trailing) {
            Rectangle().fill(Th.border).frame(width: 1)
        }
    }
}

private struct PRMetaBox: View {
    @EnvironmentObject var store: AppStore
    let pr: PullRequest

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 7) {
                Text(stateLabel)
                    .font(.system(size: 10, weight: .bold))
                    .kerning(0.3)
                    .foregroundColor(stateColor)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(stateColor.opacity(0.16)))
                Text("+\(pr.additions)")
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundColor(Th.green)
                Text("−\(pr.deletions)")
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundColor(Th.red)
                Spacer()
                Button(action: { store.openCurrentPROnGitHub() }) {
                    Image(systemName: "arrow.up.right.square")
                        .font(.system(size: 12))
                        .foregroundColor(Th.dim)
                }
                .buttonStyle(.plain)
                .help("Open on GitHub")
            }
            if pr.checks.total > 0 {
                HStack(spacing: 10) {
                    if pr.checks.passed > 0 {
                        Label("\(pr.checks.passed)", systemImage: "checkmark.circle.fill")
                            .font(.system(size: 10.5))
                            .foregroundColor(Th.greenSoft)
                    }
                    if pr.checks.failed > 0 {
                        Label("\(pr.checks.failed)", systemImage: "xmark.circle.fill")
                            .font(.system(size: 10.5))
                            .foregroundColor(Th.redSoft)
                    }
                    if pr.checks.pending > 0 {
                        Label("\(pr.checks.pending)", systemImage: "circle.dotted")
                            .font(.system(size: 10.5))
                            .foregroundColor(Th.yellow)
                    }
                    Text("checks")
                        .font(.system(size: 10.5))
                        .foregroundColor(Th.faint)
                    Spacer()
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 9).fill(Color.white.opacity(0.04)))
        .padding(.horizontal, 10)
        .padding(.top, 10)
    }

    private var stateLabel: String {
        if pr.state == "MERGED" { return "MERGED" }
        if pr.state == "CLOSED" { return "CLOSED" }
        if pr.isDraft { return "DRAFT" }
        return "OPEN"
    }

    private var stateColor: Color {
        switch pr.state {
        case "MERGED": return Th.purple
        case "CLOSED": return Th.red
        default: return pr.isDraft ? Th.faint : Th.green
        }
    }
}

private struct FileRow: View {
    @EnvironmentObject var store: AppStore
    let entry: FileEntry

    var body: some View {
        let selected = store.selectedPath == entry.path

        HStack(spacing: 9) {
            CheckBox(on: entry.checked) {
                store.toggleEntry(entry)
            }
            .help(store.isWorkingTree
                  ? (entry.inStagedSection ? "Unstage file" : "Stage file")
                  : "Mark as viewed")

            StatusChip(status: entry.status)

            VStack(alignment: .leading, spacing: 0) {
                Text(entry.name)
                    .font(.system(size: 13))
                    .foregroundColor(selected ? Color(hex: 0x7db4ff) : Th.text)
                    .lineLimit(1)
                Text(entry.dir)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundColor(Th.dimmer)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if entry.commentCount > 0 {
                Text("\(entry.commentCount)")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(Th.text)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(Capsule().fill(Color.white.opacity(0.12)))
            }

            if entry.add > 0 {
                Text("+\(entry.add)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Th.green)
            }
            if entry.del > 0 {
                Text("−\(entry.del)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Th.red)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(selected ? Th.accent.opacity(0.16) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture { store.selectFile(entry.path) }
        .opacity(!store.isWorkingTree && entry.checked ? 0.55 : 1)
    }
}
