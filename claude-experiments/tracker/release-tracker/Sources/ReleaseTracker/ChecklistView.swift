import SwiftUI
import Shared

@MainActor
final class ChecklistViewModel: ObservableObject {
    @Published private(set) var checklist: Checklist
    private let store = ChecklistStore()

    init() {
        self.checklist = store.load()
    }

    func toggle(_ item: ChecklistItem) {
        guard let idx = checklist.items.firstIndex(where: { $0.id == item.id }) else { return }
        checklist.items[idx].isDone.toggle()
        try? store.save(checklist)
    }

    func reload() {
        checklist = store.load()
    }

    func resetToSeed() {
        try? store.resetToSeed()
        reload()
    }
}

struct ChecklistView: View {
    @StateObject private var vm = ChecklistViewModel()
    @State private var showResetConfirmation = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 24) {
                    ForEach(vm.checklist.categories, id: \.self) { category in
                        categorySection(category)
                    }
                }
                .padding(20)
            }
        }
        .frame(minWidth: 480, minHeight: 600)
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button {
                    showResetConfirmation = true
                } label: {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                }
                .help("Re-seed checklist (clears all checks)")
            }
        }
        .confirmationDialog(
            "Reset checklist?",
            isPresented: $showResetConfirmation,
            titleVisibility: .visible
        ) {
            Button("Reset to seed", role: .destructive) {
                vm.resetToSeed()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This clears all your check marks and reloads the original list.")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("Ease Release")
                    .font(.title2.weight(.semibold))
                Spacer()
                Text("\(vm.checklist.doneCount) / \(vm.checklist.totalCount)")
                    .font(.system(.body, design: .rounded).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: vm.checklist.fraction)
                .progressViewStyle(.linear)
                .tint(.accentColor)
        }
        .padding(20)
    }

    private func categorySection(_ category: String) -> some View {
        let items = vm.checklist.items(in: category)
        let done = items.filter(\.isDone).count
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(category)
                    .font(.headline)
                Spacer()
                Text("\(done) / \(items.count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            ForEach(items) { item in
                ChecklistRow(item: item) {
                    vm.toggle(item)
                }
            }
        }
    }
}

struct ChecklistRow: View {
    let item: ChecklistItem
    let onToggle: () -> Void

    var body: some View {
        Button(action: onToggle) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: item.isDone ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 16))
                    .foregroundStyle(item.isDone ? .green : .secondary)
                    .padding(.top, 1)
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.title)
                        .strikethrough(item.isDone, color: .secondary)
                        .foregroundStyle(item.isDone ? .secondary : .primary)
                    if let note = item.note, !note.isEmpty {
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
