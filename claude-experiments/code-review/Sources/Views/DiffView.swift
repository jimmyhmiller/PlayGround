import SwiftUI

struct DiffView: View {
    @EnvironmentObject var store: AppStore

    var body: some View {
        VStack(spacing: 0) {
            if let file = store.selectedFile {
                FileHeaderView(file: file)
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(store.rows(for: file)) { row in
                            DiffRowView(row: row, file: file)
                        }
                        Color.clear.frame(height: 40)
                    }
                }
            } else if store.isLoading {
                Spacer()
                ProgressView("Loading diff…")
                    .controlSize(.small)
                    .foregroundColor(Th.dim)
                Spacer()
            } else if let error = store.loadError {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 22))
                        .foregroundColor(Th.orange)
                    Text(error)
                        .font(.system(size: 12.5))
                        .foregroundColor(Th.text3)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 420)
                }
                Spacer()
            } else {
                Spacer()
                Text("No changes to show.")
                    .font(.system(size: 13))
                    .foregroundColor(Th.dimmer)
                Spacer()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Th.bg)
    }
}

private struct FileHeaderView: View {
    @EnvironmentObject var store: AppStore
    let file: DisplayFile

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 1) {
                Text(file.name)
                    .font(.system(size: 13.5, weight: .semibold))
                    .foregroundColor(Th.text)
                Text(file.path)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundColor(Th.dimmer)
                    .lineLimit(1)
            }
            Text("+\(file.additions)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Th.green)
            Text("−\(file.deletions)")
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Th.red)

            Spacer()

            SegmentedPill(
                items: DiffMode.allCases.map { SegmentedPill.Item(value: $0, label: $0.rawValue) },
                selected: store.mode,
                action: { store.mode = $0 }
            )

            if store.isPR {
                HStack(spacing: 6) {
                    CheckBox(on: store.viewedSet.contains(file.path)) {
                        store.toggleViewed(file.path)
                    }
                    Text("Viewed")
                        .font(.system(size: 12))
                        .foregroundColor(Th.text3)
                }
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 9)
        .background(Th.bgHeader)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Th.border).frame(height: 1)
        }
    }
}

private struct DiffRowView: View {
    let row: DiffRowModel
    let file: DisplayFile

    var body: some View {
        switch row {
        case .hunk(let hunk):
            HunkHeaderView(hunk: hunk, file: file)
        case .uline(_, let line):
            UnifiedLineView(line: line, path: file.path)
        case .pair(_, let left, let right):
            SplitPairView(left: left, right: right, path: file.path)
        case .thread(let thread):
            ThreadView(thread: thread)
        case .note(_, let text):
            Text(text)
                .font(.system(size: 12))
                .foregroundColor(Th.dimmer)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
        }
    }
}

private struct HunkHeaderView: View {
    @EnvironmentObject var store: AppStore
    let hunk: Hunk
    let file: DisplayFile

    var body: some View {
        HStack(spacing: 10) {
            Text(hunk.header)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(Th.hunkText)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)

            if store.isWorkingTree && !file.untracked {
                Button(action: { store.toggleHunk(hunk) }) {
                    Text(hunk.staged ? "✓ Staged" : "Stage hunk")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(hunk.staged ? Th.greenSoft : Th.text2)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 3)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(hunk.staged ? Th.green.opacity(0.15) : Color.white.opacity(0.05))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .strokeBorder(hunk.staged ? Th.green.opacity(0.4) : Color.white.opacity(0.14), lineWidth: 1)
                                )
                        )
                }
                .buttonStyle(.plain)
                .help(hunk.staged ? "Unstage this hunk" : "Stage this hunk (git apply --cached)")
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 5)
        .background(Th.hunkBg)
        .overlay(alignment: .top) { Rectangle().fill(Th.hunkBorder).frame(height: 1) }
        .overlay(alignment: .bottom) { Rectangle().fill(Th.hunkBorder).frame(height: 1) }
    }
}

private struct UnifiedLineView: View {
    @EnvironmentObject var store: AppStore
    let line: DiffLine
    let path: String
    @State private var hovered = false

    var body: some View {
        HStack(spacing: 0) {
            Text(line.oldNo.map(String.init) ?? "")
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Th.faint)
                .frame(width: 44, alignment: .trailing)
                .padding(.trailing, 8)
                .background(gutterColor)
            Text(line.newNo.map(String.init) ?? "")
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Th.faint)
                .frame(width: 44, alignment: .trailing)
                .padding(.trailing, 8)
                .background(gutterColor)
            Text(sign)
                .font(.system(size: 12.5, design: .monospaced))
                .foregroundColor(signColor)
                .frame(width: 20)
            Text(line.text.isEmpty ? " " : line.text)
                .font(.system(size: 12.5, design: .monospaced))
                .foregroundColor(Th.codeText)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.trailing, 12)
        }
        .frame(minHeight: 20)
        .background(rowColor)
        .overlay(hovered ? Color.white.opacity(0.04) : Color.clear)
        .contentShape(Rectangle())
        .onHover { hovered = $0 }
        .onTapGesture {
            store.openComposer(path: path, line: line.newNo ?? line.oldNo)
        }
        .help("Click to comment on this line")
    }

    private var sign: String {
        switch line.kind {
        case .add: return "+"
        case .del: return "-"
        case .ctx: return " "
        }
    }

    private var signColor: Color {
        switch line.kind {
        case .add: return Th.green
        case .del: return Th.red
        case .ctx: return Th.faint
        }
    }

    private var rowColor: Color {
        switch line.kind {
        case .add: return Th.addBg
        case .del: return Th.delBg
        case .ctx: return .clear
        }
    }

    private var gutterColor: Color {
        switch line.kind {
        case .add: return Th.addGutter
        case .del: return Th.delGutter
        case .ctx: return Th.ctxGutter
        }
    }
}

private struct SplitPairView: View {
    @EnvironmentObject var store: AppStore
    let left: DiffLine?
    let right: DiffLine?
    let path: String

    var body: some View {
        HStack(spacing: 0) {
            sideView(line: left, isLeft: true)
                .overlay(alignment: .trailing) {
                    Rectangle().fill(Color.white.opacity(0.06)).frame(width: 1)
                }
            sideView(line: right, isLeft: false)
        }
        .frame(minHeight: 20)
    }

    @ViewBuilder
    private func sideView(line: DiffLine?, isLeft: Bool) -> some View {
        let bg: Color = {
            guard let line else { return Color.white.opacity(0.02) }
            switch line.kind {
            case .add: return Th.addBg
            case .del: return Th.delBg
            case .ctx: return .clear
            }
        }()
        let gutter: Color = {
            guard let line else { return Color.white.opacity(0.02) }
            switch line.kind {
            case .add: return Th.addGutter
            case .del: return Th.delGutter
            case .ctx: return Th.ctxGutter
            }
        }()
        let number = isLeft ? line?.oldNo : line?.newNo

        HStack(spacing: 0) {
            Text(number.map(String.init) ?? "")
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Th.faint)
                .frame(width: 40, alignment: .trailing)
                .padding(.trailing, 6)
                .background(gutter)
            Text(line?.text.isEmpty == false ? line!.text : " ")
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(Th.codeText)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
        }
        .background(bg)
        .contentShape(Rectangle())
        .onTapGesture {
            if let line {
                store.openComposer(path: path, line: line.newNo ?? line.oldNo)
            }
        }
    }
}
