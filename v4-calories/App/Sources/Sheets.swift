import SwiftUI
import CalorieModel

/// Numeric keypad shared by both entry sheets. Keys: "0"–"9", ".", "del", "clear".
struct Keypad: View {
    let decimal: Bool
    let onKey: (String) -> Void

    private let cols = Array(repeating: GridItem(.flexible(), spacing: 8), count: 3)

    var body: some View {
        LazyVGrid(columns: cols, spacing: 8) {
            ForEach(1...9, id: \.self) { n in key("\(n)") }
            key(decimal ? "." : "clear", label: decimal ? "." : "CLR", filled: decimal)
            key("0")
            key("del", label: "⌫", filled: false)
        }
    }

    private func key(_ value: String, label: String? = nil, filled: Bool = true) -> some View {
        Button { onKey(value) } label: {
            Text(label ?? value)
                .font(value == "clear" ? .mono(13, .semibold) : .mono(value == "del" ? 22 : 24, .semibold))
                .foregroundStyle(filled ? Theme.text : Theme.textDim(0.7))
                .frame(maxWidth: .infinity).padding(.vertical, 17)
                .background(filled ? RoundedRectangle(cornerRadius: 14).fill(Theme.key) : nil)
        }
    }
}

/// Sheet chrome: grabber + sheet background.
private struct SheetShell<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(Color.white.opacity(0.18)).frame(width: 38, height: 5)
                .padding(.top, 10).padding(.bottom, 16)
            content
        }
        .padding(.horizontal, 18).padding(.bottom, 24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Theme.sheetBg.ignoresSafeArea())
    }
}

struct EntrySheet: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss

    let editing: CalorieEntry?
    @State private var draft: String
    @State private var name: String
    @State private var showSave = false
    @State private var saveLabel = ""

    init(editing: CalorieEntry? = nil) {
        self.editing = editing
        _draft = State(initialValue: editing.map { String(Int($0.kcal)) } ?? "")
        _name = State(initialValue: editing?.label ?? "")
    }

    private var value: Double { Double(draft) ?? 0 }
    private var hasDraft: Bool { value > 0 }
    private var isEditing: Bool { editing != nil }

    var body: some View {
        SheetShell {
            HStack {
                Text(isEditing ? "Edit entry" : "Add calories").font(.system(size: 15, weight: .semibold))
                Spacer()
                Text("\(Fmt.int(store.todayTotal)) / \(Fmt.int(store.analysis?.dailyBudgetKcal ?? 2000))")
                    .font(.mono(12)).foregroundStyle(Theme.textDim(0.5))
            }

            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(draft.isEmpty ? "0" : draft)
                    .font(.mono(60, .heavy)).tracking(-2)
                    .foregroundStyle(hasDraft ? Theme.text : Theme.textDim(0.28))
                Text("kcal").font(.mono(16)).foregroundStyle(Theme.textDim(0.4))
            }
            .padding(.top, 14).padding(.bottom, 6)

            TextField("", text: $name, prompt: Text("Name (optional)").foregroundColor(Theme.textDim(0.35)))
                .font(.system(size: 14)).foregroundStyle(Theme.text)
                .multilineTextAlignment(.center)
                .padding(.vertical, 9)
                .background(RoundedRectangle(cornerRadius: 10).fill(Theme.key))
                .padding(.bottom, 14)

            shortcuts.padding(.bottom, 16)

            Keypad(decimal: false) { handle($0) }

            Button { commit() } label: {
                Text(buttonLabel)
                    .font(.system(size: 16, weight: .bold)).foregroundStyle(Theme.onGreen)
                    .frame(maxWidth: .infinity).padding(17)
                    .background(RoundedRectangle(cornerRadius: 14).fill(Theme.green))
                    .opacity(hasDraft ? 1 : 0.4)
            }
            .disabled(!hasDraft)
            .padding(.top, 12)

            if isEditing {
                Button(role: .destructive) {
                    if let e = editing { store.deleteEntry(e.id) }
                    dismiss()
                } label: {
                    Text("Delete entry").font(.system(size: 14, weight: .medium)).foregroundStyle(Theme.amber)
                        .frame(maxWidth: .infinity).padding(.top, 12)
                }
            }
        }
        .alert("Save shortcut", isPresented: $showSave) {
            TextField("Name (e.g. Usual lunch)", text: $saveLabel)
            Button("Save") {
                store.addShortcut(label: saveLabel.isEmpty ? (name.isEmpty ? "Shortcut" : name) : saveLabel, kcal: value)
                saveLabel = ""
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Save \(Fmt.int(value)) kcal as a one-tap shortcut.")
        }
    }

    private var buttonLabel: String {
        if isEditing { return "Save changes" }
        return hasDraft ? "Log \(Fmt.int(value)) kcal" : "Log calories"
    }

    private var shortcuts: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(store.state.shortcuts) { s in
                    Button { addShortcut(s.kcal) } label: {
                        VStack(spacing: 2) {
                            Text("+\(Fmt.int(s.kcal))").font(.mono(15, .bold)).foregroundStyle(Theme.green)
                            Text(s.label).font(.system(size: 10)).foregroundStyle(Theme.textDim(0.55)).lineLimit(1)
                        }
                        .padding(.horizontal, 14).padding(.vertical, 9)
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.1), lineWidth: 0.5))
                    }
                }
                Button { if hasDraft { saveLabel = name; showSave = true } } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "bookmark").font(.system(size: 11))
                        Text("Save shortcut").font(.system(size: 12))
                    }
                    .foregroundStyle(hasDraft ? Theme.green : Theme.textDim(0.4))
                    .padding(.horizontal, 14).padding(.vertical, 9)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke((hasDraft ? Theme.green : Color.white).opacity(0.3), style: StrokeStyle(lineWidth: 0.5, dash: [3, 2])))
                }
                .disabled(!hasDraft)
            }
        }
    }

    private func handle(_ k: String) {
        switch k {
        case "del": if !draft.isEmpty { draft.removeLast() }
        case "clear": draft = ""
        default:
            if draft.count < 5 { draft = (draft == "0" ? "" : draft) + k }
        }
    }

    private func addShortcut(_ kcal: Double) {
        draft = String(Int((Double(draft) ?? 0) + kcal))
    }

    private func commit() {
        guard hasDraft else { return }
        let label = name.trimmingCharacters(in: .whitespaces)
        if let e = editing {
            store.updateEntry(e.id, kcal: value, label: label.isEmpty ? nil : label)
        } else {
            store.addEntry(kcal: value, label: label.isEmpty ? nil : label)
        }
        dismiss()
    }
}

struct WeighInSheet: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var draft = ""

    private var value: Double { Double(draft) ?? 0 }
    private var hasDraft: Bool { value > 0 }

    var body: some View {
        SheetShell {
            Text("Log weight").font(.system(size: 15, weight: .semibold))
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(draft.isEmpty ? "0" : draft)
                    .font(.mono(60, .heavy)).tracking(-2)
                    .foregroundStyle(hasDraft ? Theme.text : Theme.textDim(0.28))
                Text("lb").font(.mono(16)).foregroundStyle(Theme.textDim(0.4))
            }
            .padding(.vertical, 14)

            Text("Water swings are expected — the trend line filters them out.")
                .font(.system(size: 12)).foregroundStyle(Theme.textDim(0.45))
                .multilineTextAlignment(.center).padding(.bottom, 16)

            Keypad(decimal: true) { handle($0) }

            Button { log() } label: {
                Text("Save weigh-in")
                    .font(.system(size: 16, weight: .bold)).foregroundStyle(Theme.onGreen)
                    .frame(maxWidth: .infinity).padding(17)
                    .background(RoundedRectangle(cornerRadius: 14).fill(Theme.green))
                    .opacity(hasDraft ? 1 : 0.4)
            }
            .disabled(!hasDraft)
            .padding(.top, 12)
        }
    }

    private func handle(_ k: String) {
        switch k {
        case "del": if !draft.isEmpty { draft.removeLast() }
        case ".": if !draft.contains(".") { draft = (draft.isEmpty ? "0" : draft) + "." }
        default:
            // allow up to 3 integer + 1 decimal digit
            if draft.replacingOccurrences(of: ".", with: "").count < 4 {
                draft = (draft == "0" ? "" : draft) + k
            }
        }
    }

    private func log() {
        guard hasDraft else { return }
        store.addWeighIn(value)
        dismiss()
    }
}
