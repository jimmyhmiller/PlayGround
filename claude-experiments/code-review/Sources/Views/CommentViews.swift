import SwiftUI

struct ThreadView: View {
    @EnvironmentObject var store: AppStore
    let thread: ThreadModel

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            ForEach(thread.comments) { comment in
                CommentCard(comment: comment)
            }

            if thread.showComposer {
                ComposerView()
            } else if !thread.comments.isEmpty {
                Button(action: { store.openComposer(path: thread.path, line: thread.line) }) {
                    Text("Reply…")
                        .font(.system(size: 11.5, weight: .medium))
                        .foregroundColor(Th.dim)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.white.opacity(0.05))
                                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Th.borderStrong, lineWidth: 1))
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.top, 12)
        .padding(.bottom, 14)
        .padding(.leading, 60)
        .padding(.trailing, 20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Th.panel)
        .overlay(alignment: .top) { Rectangle().fill(Th.border).frame(height: 1) }
        .overlay(alignment: .bottom) { Rectangle().fill(Th.border).frame(height: 1) }
    }
}

struct CommentCard: View {
    @EnvironmentObject var store: AppStore
    let comment: ReviewComment
    @State private var hovered = false

    var body: some View {
        let editing = store.editingID == comment.id

        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                Avatar(isAI: comment.isAI, label: comment.isMine ? initials(store.login.isEmpty ? "You" : store.login) : initials(comment.author))
                Text(displayName)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundColor(Th.text)
                if let cat = comment.category {
                    let (fg, bg) = Th.categoryColors(cat)
                    Text(cat.uppercased())
                        .font(.system(size: 9.5, weight: .bold))
                        .kerning(0.3)
                        .foregroundColor(fg)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2)
                        .background(RoundedRectangle(cornerRadius: 5).fill(bg))
                }
                Spacer()
                if !editing {
                    HStack(spacing: 5) {
                        Button(action: { store.beginEdit(comment) }) {
                            Text(comment.isAI ? "Reword" : "Edit")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(Th.text2)
                                .padding(.horizontal, 9)
                                .padding(.vertical, 2)
                                .background(
                                    RoundedRectangle(cornerRadius: 5)
                                        .fill(Color.white.opacity(0.06))
                                        .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(Th.borderStrong, lineWidth: 1))
                                )
                        }
                        .buttonStyle(.plain)
                        Button(action: { store.deleteComment(comment) }) {
                            Text("✕")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(Th.red)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(
                                    RoundedRectangle(cornerRadius: 5)
                                        .fill(Th.red.opacity(0.12))
                                        .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(Th.red.opacity(0.3), lineWidth: 1))
                                )
                        }
                        .buttonStyle(.plain)
                        .help(comment.isAI ? "Remove AI comment from this review" : "Delete comment")
                    }
                    .opacity(hovered ? 1 : 0)
                    .animation(.easeInOut(duration: 0.12), value: hovered)
                }
            }

            if editing {
                VStack(alignment: .leading, spacing: 6) {
                    TextEditor(text: $store.editText)
                        .font(.system(size: 13))
                        .foregroundColor(Th.text)
                        .scrollContentBackground(.hidden)
                        .padding(5)
                        .frame(minHeight: 56)
                        .background(
                            RoundedRectangle(cornerRadius: 7)
                                .fill(Th.editorField)
                                .overlay(RoundedRectangle(cornerRadius: 7).strokeBorder(Color.white.opacity(0.15), lineWidth: 1))
                        )
                    HStack(spacing: 7) {
                        Button(action: { store.saveEdit() }) {
                            Text("Save")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundColor(.white)
                                .padding(.horizontal, 13)
                                .padding(.vertical, 4)
                                .background(RoundedRectangle(cornerRadius: 6).fill(comment.isAI ? Th.ai : Th.accent))
                        }
                        .buttonStyle(.plain)
                        Button(action: { store.cancelEdit() }) {
                            Text("Cancel")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundColor(Th.text2)
                                .padding(.horizontal, 13)
                                .padding(.vertical, 4)
                                .background(
                                    RoundedRectangle(cornerRadius: 6)
                                        .fill(Color.white.opacity(0.06))
                                        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Th.borderStrong, lineWidth: 1))
                                )
                        }
                        .buttonStyle(.plain)
                    }
                }
            } else {
                Text(comment.body)
                    .font(.system(size: 13))
                    .lineSpacing(3)
                    .foregroundColor(Color(hex: 0xc8c8cc))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 9)
        .background(
            RoundedRectangle(cornerRadius: 9)
                .fill(Th.card)
                .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(Color.white.opacity(0.08), lineWidth: 1))
        )
        .onHover { hovered = $0 }
    }

    private var displayName: String {
        if comment.isMine || comment.author == "You" { return "You" }
        if comment.isAI { return comment.author + " · AI reviewer" }
        return comment.author
    }
}

struct ComposerView: View {
    @EnvironmentObject var store: AppStore

    private var textBinding: Binding<String> {
        Binding(
            get: { store.composer?.text ?? "" },
            set: { newValue in
                if var c = store.composer {
                    c.text = newValue
                    store.composer = c
                }
            }
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            // Saved replies
            VStack(alignment: .leading, spacing: 6) {
                FlowLayout(spacing: 6) {
                    Text("SAVED REPLIES")
                        .font(.system(size: 10, weight: .semibold))
                        .kerning(0.4)
                        .foregroundColor(Color(hex: 0x7a7a80))
                        .padding(.vertical, 4)
                    ForEach(store.savedReplies, id: \.self) { reply in
                        SavedReplyChip(reply: reply)
                    }
                    Button(action: { store.saveCurrentReply() }) {
                        Text("＋ Save current")
                            .font(.system(size: 11.5, weight: .semibold))
                            .foregroundColor(Th.accent)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 3)
                            .background(
                                Capsule().strokeBorder(Color.white.opacity(0.2), style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
                            )
                    }
                    .buttonStyle(.plain)
                    .help("Save the current draft as a reusable reply")
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.white.opacity(0.03))
            .overlay(alignment: .bottom) { Rectangle().fill(Th.border).frame(height: 1) }

            ZStack(alignment: .topLeading) {
                if (store.composer?.text ?? "").isEmpty {
                    Text("Write a reply, or click a saved one above…")
                        .font(.system(size: 13))
                        .foregroundColor(Th.dimmer)
                        .padding(.horizontal, 12)
                        .padding(.top, 10)
                        .allowsHitTesting(false)
                }
                TextEditor(text: textBinding)
                    .font(.system(size: 13))
                    .foregroundColor(Th.text)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2)
                    .frame(minHeight: 56)
            }

            HStack {
                Text("Posts as You" + lineSuffix)
                    .font(.system(size: 11))
                    .foregroundColor(Color(hex: 0x7a7a80))
                Spacer()
                Button(action: { store.closeComposer() }) {
                    Text("Cancel")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(Th.text2)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 5)
                }
                .buttonStyle(.plain)
                Button(action: { store.addComment() }) {
                    Text("Add comment")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 15)
                        .padding(.vertical, 5)
                        .background(RoundedRectangle(cornerRadius: 7).fill(Th.accent))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(Color.white.opacity(0.03))
            .overlay(alignment: .top) { Rectangle().fill(Th.border).frame(height: 1) }
        }
        .background(RoundedRectangle(cornerRadius: 11).fill(Th.card))
        .clipShape(RoundedRectangle(cornerRadius: 11))
        .overlay(RoundedRectangle(cornerRadius: 11).strokeBorder(Color.white.opacity(0.1), lineWidth: 1))
    }

    private var lineSuffix: String {
        if let line = store.composer?.line { return " · line \(line)" }
        return ""
    }
}

private struct SavedReplyChip: View {
    @EnvironmentObject var store: AppStore
    let reply: String

    var body: some View {
        HStack(spacing: 0) {
            Button(action: { store.insertReply(reply) }) {
                Text(reply)
                    .font(.system(size: 11.5))
                    .foregroundColor(Th.codeText)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .frame(maxWidth: 220, alignment: .leading)
                    .fixedSize(horizontal: true, vertical: false)
                    .padding(.leading, 11)
                    .padding(.trailing, 4)
                    .padding(.vertical, 3)
            }
            .buttonStyle(.plain)
            .help(reply)
            Button(action: { store.removeReply(reply) }) {
                Text("✕")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(Th.dim)
                    .padding(.trailing, 8)
                    .padding(.leading, 3)
                    .padding(.vertical, 3)
            }
            .buttonStyle(.plain)
            .help("Remove saved reply")
        }
        .background(
            Capsule()
                .fill(Color.white.opacity(0.07))
                .overlay(Capsule().strokeBorder(Th.borderStrong, lineWidth: 1))
        )
    }
}
