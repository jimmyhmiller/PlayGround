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
    @State private var showAI = false

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
        .sheet(isPresented: $showAI) {
            AskAISheet { total, label in
                draft = String(total)
                if name.trimmingCharacters(in: .whitespaces).isEmpty { name = label }
            }
            .presentationDetents([.large])
        }
    }

    private var buttonLabel: String {
        if isEditing { return "Save changes" }
        return hasDraft ? "Log \(Fmt.int(value)) kcal" : "Log calories"
    }

    private var shortcuts: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                Button { showAI = true } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "sparkles").font(.system(size: 12))
                        Text("Ask AI").font(.system(size: 12, weight: .semibold))
                    }
                    .foregroundStyle(Theme.green)
                    .padding(.horizontal, 14).padding(.vertical, 9)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Theme.green.opacity(0.13)))
                }
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

/// Describe a meal in plain language; DeepSeek estimates the calories and you confirm.
struct AskAISheet: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    /// Called with the chosen total and a short label when the user accepts an estimate.
    var onUse: (Int, String) -> Void

    @State private var text = ""
    @State private var busy = false
    @State private var error: String?
    @State private var result: CalorieEstimate?
    @FocusState private var focused: Bool

    private let examples = ["two eggs, toast & butter, latte",
                            "chipotle chicken bowl w/ guac",
                            "handful of almonds"]

    var body: some View {
        SheetShell {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: 6) {
                        Image(systemName: "sparkles").foregroundStyle(Theme.green)
                        Text("Ask AI").font(.system(size: 15, weight: .semibold))
                        Spacer()
                    }
                    Text("Describe what you ate. You'll confirm before it's logged.")
                        .font(.system(size: 13)).foregroundStyle(Theme.textDim(0.5))
                        .padding(.top, 4)

                    TextField("", text: $text, prompt: Text("e.g. two eggs, toast with butter, and a latte")
                        .foregroundColor(Theme.textDim(0.32)), axis: .vertical)
                        .font(.system(size: 16)).foregroundStyle(Theme.text)
                        .lineLimit(2...5)
                        .focused($focused)
                        .submitLabel(.go)
                        .padding(12)
                        .background(RoundedRectangle(cornerRadius: 12).fill(Theme.key))
                        .padding(.top, 14)

                    if result == nil && !busy {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(examples, id: \.self) { ex in
                                    Button { text = ex } label: {
                                        Text(ex).font(.system(size: 12)).foregroundStyle(Theme.textDim(0.6))
                                            .padding(.horizontal, 12).padding(.vertical, 7)
                                            .overlay(RoundedRectangle(cornerRadius: 10)
                                                .stroke(Color.white.opacity(0.1), lineWidth: 0.5))
                                    }
                                }
                            }
                        }
                        .padding(.top, 10)
                    }

                    Button { run() } label: {
                        HStack(spacing: 8) {
                            if busy { ProgressView().tint(Theme.onGreen) }
                            Text(busy ? "Estimating…" : "Estimate calories")
                                .font(.system(size: 16, weight: .bold)).foregroundStyle(Theme.onGreen)
                        }
                        .frame(maxWidth: .infinity).padding(16)
                        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.green))
                        .opacity(canRun ? 1 : 0.4)
                    }
                    .disabled(!canRun)
                    .padding(.top, 14)

                    if let error {
                        Text(error).font(.system(size: 13)).foregroundStyle(Theme.amber)
                            .padding(.top, 12)
                    }

                    if let result { resultCard(result).padding(.top, 18) }
                }
            }
            .scrollIndicators(.hidden)
        }
        .onAppear { focused = true }
    }

    private var canRun: Bool {
        !busy && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    @ViewBuilder
    private func resultCard(_ est: CalorieEstimate) -> some View {
        if est.isFood {
            VStack(alignment: .leading, spacing: 0) {
                Text(est.label.uppercased()).font(.mono(10)).tracking(1).foregroundStyle(Theme.textDim(0.45))
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("\(est.total)").font(.mono(52, .heavy)).tracking(-2)
                    Text("kcal").font(.mono(15)).foregroundStyle(Theme.textDim(0.4))
                }
                .padding(.top, 2)
                Text("estimated range \(est.low)–\(est.high) kcal")
                    .font(.system(size: 12)).foregroundStyle(Theme.textDim(0.45))

                if !est.items.isEmpty {
                    VStack(spacing: 0) {
                        ForEach(est.items) { item in
                            HStack {
                                Text(item.name).font(.system(size: 13)).foregroundStyle(Theme.textDim(0.7))
                                Spacer()
                                Text("\(item.kcal)").font(.mono(13)).foregroundStyle(Theme.textDim(0.55))
                            }
                            .padding(.vertical, 9)
                            .overlay(Rectangle().fill(Theme.hair).frame(height: 0.5), alignment: .top)
                        }
                    }
                    .padding(.top, 12)
                }

                Button { onUse(est.total, est.label); dismiss() } label: {
                    Text("Use \(est.total) kcal")
                        .font(.system(size: 16, weight: .bold)).foregroundStyle(Theme.onGreen)
                        .frame(maxWidth: .infinity).padding(16)
                        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.green))
                }
                .padding(.top, 16)
                Text("It's an estimate — you can tweak the number after.")
                    .font(.system(size: 11)).foregroundStyle(Theme.textDim(0.4))
                    .frame(maxWidth: .infinity, alignment: .center).padding(.top, 8)
            }
            .padding(16)
            .background(RoundedRectangle(cornerRadius: 16).fill(Theme.key))
        } else {
            HStack(spacing: 8) {
                Image(systemName: "questionmark.circle").foregroundStyle(Theme.amber)
                Text("That doesn't look like food. Try describing a meal or snack.")
                    .font(.system(size: 13)).foregroundStyle(Theme.textDim(0.6))
            }
            .padding(14)
            .background(RoundedRectangle(cornerRadius: 12).fill(Theme.key))
        }
    }

    private func run() {
        guard canRun else { return }
        guard let key = store.aiKey else {
            error = CalorieAIError.noKey.errorDescription
            return
        }
        focused = false
        error = nil
        result = nil
        busy = true
        let query = text
        Task {
            do {
                result = try await CalorieAI.estimate(query, apiKey: key)
            } catch {
                self.error = (error as? CalorieAIError)?.errorDescription ?? error.localizedDescription
            }
            busy = false
        }
    }
}
