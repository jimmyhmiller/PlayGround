import SwiftUI

/// Raw patch editor — the equivalent of `git add -e`. The text here is applied
/// verbatim with `git apply --cached`, so git validates it; a rejected patch
/// keeps the sheet open with the error rather than losing the edit.
struct HunkEditorSheet: View {
    @EnvironmentObject var store: AppStore
    let hunk: Hunk

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(hunk.staged ? "Edit patch to unstage" : "Edit patch to stage")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(Th.text)
                    Text(hunk.staged
                         ? "Applied in reverse against the index. Drop a line to keep it staged."
                         : "Applied against the index. Delete a + line to leave it unstaged; turn a − into a space to keep the line.")
                        .font(.system(size: 11.5))
                        .foregroundColor(Th.dim)
                }
                Spacer()
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(Th.bgHeader)
            .overlay(alignment: .bottom) { Rectangle().fill(Th.border).frame(height: 1) }

            TextEditor(text: $store.editingPatchText)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(Th.codeText)
                .scrollContentBackground(.hidden)
                .padding(8)
                .background(Th.cardDark)

            if let error = store.editingPatchError {
                HStack(alignment: .top, spacing: 7) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11))
                        .foregroundColor(Th.redSoft)
                    Text(error)
                        .font(.system(size: 11.5, design: .monospaced))
                        .foregroundColor(Th.redSoft)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 9)
                .background(Th.red.opacity(0.1))
            }

            HStack(spacing: 10) {
                Button(action: { store.resetHunkEdit() }) {
                    Text("Reset")
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundColor(Th.text2)
                        .padding(.horizontal, 13)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 7)
                                .fill(Color.white.opacity(0.06))
                                .overlay(RoundedRectangle(cornerRadius: 7).strokeBorder(Th.borderStrong, lineWidth: 1))
                        )
                }
                .buttonStyle(.plain)
                .help("Restore the original patch text")

                Spacer()

                Button(action: { store.cancelHunkEdit() }) {
                    Text("Cancel")
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundColor(Th.text2)
                        .padding(.horizontal, 13)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.cancelAction)

                Button(action: { store.applyHunkEdit() }) {
                    Text(hunk.staged ? "Apply (unstage)" : "Apply (stage)")
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 15)
                        .padding(.vertical, 6)
                        .background(RoundedRectangle(cornerRadius: 7).fill(Th.accent))
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .background(Th.bgHeader)
            .overlay(alignment: .top) { Rectangle().fill(Th.border).frame(height: 1) }
        }
        .frame(width: 760, height: 560)
        .background(Th.bg)
    }
}
