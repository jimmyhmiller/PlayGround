import SwiftUI

/// Runtime editor for the canned replies: edit text in place, add, delete,
/// and drag to reorder. Every change persists immediately.
struct SavedRepliesEditor: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Saved replies")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(Th.text)
                    Text("Edit the text, drag to reorder, ⌫ to delete.")
                        .font(.system(size: 11.5))
                        .foregroundColor(Th.dim)
                }
                Spacer()
                Button(action: { store.addBlankReply() }) {
                    Label("Add", systemImage: "plus")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 5)
                        .background(RoundedRectangle(cornerRadius: 7).fill(Th.accent))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(Th.bgHeader)
            .overlay(alignment: .bottom) { Rectangle().fill(Th.border).frame(height: 1) }

            if store.savedReplies.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "text.bubble")
                        .font(.system(size: 22))
                        .foregroundColor(Th.faint)
                    Text("No saved replies yet.")
                        .font(.system(size: 12.5))
                        .foregroundColor(Th.dimmer)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(Array(store.savedReplies.enumerated()), id: \.offset) { index, _ in
                        ReplyEditorRow(index: index)
                            .listRowBackground(Color.clear)
                            .listRowSeparatorTint(Color.white.opacity(0.06))
                    }
                    .onMove { store.moveReplies(from: $0, to: $1) }
                    .onDelete { store.removeReplies(at: $0) }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
            }

            HStack {
                Text("\(store.savedReplies.count) repl\(store.savedReplies.count == 1 ? "y" : "ies")")
                    .font(.system(size: 11.5))
                    .foregroundColor(Th.dimmer)
                Spacer()
                Button(action: {
                    store.tidyReplies()
                    dismiss()
                }) {
                    Text("Done")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 17)
                        .padding(.vertical, 6)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Th.accent))
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .background(Th.bgHeader)
            .overlay(alignment: .top) { Rectangle().fill(Th.border).frame(height: 1) }
        }
        .frame(width: 520, height: 460)
        .background(Th.bg)
    }
}

private struct ReplyEditorRow: View {
    @EnvironmentObject var store: AppStore
    let index: Int

    private var binding: Binding<String> {
        Binding(
            get: { store.savedReplies.indices.contains(index) ? store.savedReplies[index] : "" },
            set: { store.updateReply(at: index, to: $0) }
        )
    }

    var body: some View {
        HStack(spacing: 9) {
            Image(systemName: "line.3.horizontal")
                .font(.system(size: 10))
                .foregroundColor(Th.faint)

            TextField("Reply text…", text: binding, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .font(.system(size: 12.5))
                .foregroundColor(Th.text)
                .padding(.horizontal, 9)
                .padding(.vertical, 6)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.white.opacity(0.05)))

            Button(action: { store.removeReplies(at: IndexSet(integer: index)) }) {
                Image(systemName: "trash")
                    .font(.system(size: 11))
                    .foregroundColor(Th.dim)
            }
            .buttonStyle(.plain)
            .help("Delete this reply")
        }
        .padding(.vertical, 3)
    }
}
