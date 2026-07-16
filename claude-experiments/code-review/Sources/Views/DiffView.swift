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
        case .uline(_, let line, let hunk, let index):
            UnifiedLineView(line: line, path: file.path, hunk: hunk, index: index)
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
        let selected = store.selection(for: hunk)
        let editable = store.isWorkingTree && !file.untracked

        HStack(spacing: 8) {
            Text(hunk.header)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(Th.hunkText)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)

            if editable {
                if !selected.isEmpty {
                    // Line picking is active: staging applies to the picks.
                    Button(action: { store.clearSelection(in: hunk) }) {
                        HunkChipLabel(text: "Clear", color: Th.text2)
                    }
                    .buttonStyle(.plain)
                    .help("Clear the selected lines")

                    Button(action: { store.applySelectedLines(in: hunk) }) {
                        HunkChipLabel(
                            text: "\(hunk.staged ? "Unstage" : "Stage") \(selected.count) line\(selected.count == 1 ? "" : "s")",
                            color: .white,
                            fill: Th.accent,
                            border: Th.accent
                        )
                    }
                    .buttonStyle(.plain)
                    .help("Apply only the selected lines")
                } else {
                    if store.isSubHunk(hunk) {
                        Button(action: { store.unsplitHunk(hunk) }) {
                            HunkChipLabel(text: "Rejoin", color: Th.text2)
                        }
                        .buttonStyle(.plain)
                        .help("Undo the split and show the original hunk")
                    } else if hunk.isSplittable {
                        Button(action: { store.splitHunk(hunk) }) {
                            HunkChipLabel(text: "Split", color: Th.text2)
                        }
                        .buttonStyle(.plain)
                        .help("Split into smaller hunks at context boundaries")
                    }

                    Button(action: { store.selectAllLines(in: hunk) }) {
                        HunkChipLabel(text: "Pick lines", color: Th.text2)
                    }
                    .buttonStyle(.plain)
                    .help("Select individual lines to stage (or click a line's +/− sign)")

                    Button(action: { store.beginHunkEdit(hunk) }) {
                        HunkChipLabel(text: "Edit", color: Th.text2)
                    }
                    .buttonStyle(.plain)
                    .help("Edit the raw patch before applying (git add -e)")
                }

                Button(action: { store.toggleHunk(hunk) }) {
                    HunkChipLabel(
                        text: hunk.staged ? "✓ Staged" : "Stage hunk",
                        color: hunk.staged ? Th.greenSoft : Th.text2,
                        fill: hunk.staged ? Th.green.opacity(0.15) : Color.white.opacity(0.05),
                        border: hunk.staged ? Th.green.opacity(0.4) : Color.white.opacity(0.14)
                    )
                }
                .buttonStyle(.plain)
                .help(hunk.staged ? "Unstage this whole hunk" : "Stage this whole hunk")
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 5)
        .background(Th.hunkBg)
        .overlay(alignment: .top) { Rectangle().fill(Th.hunkBorder).frame(height: 1) }
        .overlay(alignment: .bottom) { Rectangle().fill(Th.hunkBorder).frame(height: 1) }
    }
}

private struct HunkChipLabel: View {
    let text: String
    var color: Color = Th.text2
    var fill: Color = Color.white.opacity(0.05)
    var border: Color = Color.white.opacity(0.14)

    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .foregroundColor(color)
            .lineLimit(1)
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(fill)
                    .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(border, lineWidth: 1))
            )
    }
}

private struct UnifiedLineView: View {
    @EnvironmentObject var store: AppStore
    let line: DiffLine
    let path: String
    let hunk: Hunk
    let index: Int
    @State private var hovered = false

    private var isChange: Bool { line.kind != .ctx }
    private var selectable: Bool { store.isWorkingTree && isChange && !hunk.fileHeader.isEmpty }
    private var isSelected: Bool { store.selection(for: hunk).contains(index) }

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

            // The sign column doubles as the line picker in working-tree mode.
            Group {
                if selectable {
                    Button(action: { store.toggleLineSelection(hunk: hunk, index: index) }) {
                        ZStack {
                            if isSelected {
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Th.accent)
                                    .frame(width: 15, height: 15)
                                Text("✓")
                                    .font(.system(size: 9, weight: .heavy))
                                    .foregroundColor(.white)
                            } else if hovered {
                                RoundedRectangle(cornerRadius: 3)
                                    .strokeBorder(Th.accent.opacity(0.7), lineWidth: 1)
                                    .frame(width: 15, height: 15)
                                Text(sign)
                                    .font(.system(size: 12.5, design: .monospaced))
                                    .foregroundColor(signColor)
                            } else {
                                Text(sign)
                                    .font(.system(size: 12.5, design: .monospaced))
                                    .foregroundColor(signColor)
                            }
                        }
                        .frame(width: 20)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(isSelected ? "Deselect this line" : "Select this line to stage on its own")
                } else {
                    Text(sign)
                        .font(.system(size: 12.5, design: .monospaced))
                        .foregroundColor(signColor)
                        .frame(width: 20)
                }
            }

            Text(line.text.isEmpty ? " " : line.text)
                .font(.system(size: 12.5, design: .monospaced))
                .foregroundColor(Th.codeText)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.trailing, 12)
                .contentShape(Rectangle())
                .onTapGesture {
                    store.openComposer(path: path, line: line.newNo ?? line.oldNo)
                }
                .help("Click to comment on this line")
        }
        .frame(minHeight: 20)
        .background(rowColor)
        .overlay(isSelected ? Th.accent.opacity(0.14) : (hovered ? Color.white.opacity(0.04) : Color.clear))
        .overlay(alignment: .leading) {
            if isSelected {
                Rectangle().fill(Th.accent).frame(width: 2)
            }
        }
        .onHover { hovered = $0 }
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
