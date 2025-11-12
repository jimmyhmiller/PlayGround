import Foundation

@MainActor
class CategoryManager: ObservableObject {
    @Published var categories: [String] = []
    @Published var showingAddCategory = false
    @Published var newCategoryName = ""

    private var workingDirectory: URL?
    private let stateFileName = ".pdf-categorizer-state.json"

    struct State: Codable {
        let categories: [String]
    }

    func setWorkingDirectory(_ url: URL?) {
        workingDirectory = url
        if let url = url {
            loadState(from: url)
        }
    }

    func addCategory(_ name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !categories.contains(trimmed) else { return }

        categories.append(trimmed)
        categories.sort()
        saveState()
    }

    func removeCategory(_ name: String) {
        categories.removeAll { $0 == name }
        saveState()
    }

    private func loadState(from directory: URL) {
        let stateURL = directory.appendingPathComponent(stateFileName)

        guard FileManager.default.fileExists(atPath: stateURL.path),
              let data = try? Data(contentsOf: stateURL),
              let state = try? JSONDecoder().decode(State.self, from: data) else {
            return
        }

        categories = state.categories
    }

    private func saveState() {
        guard let directory = workingDirectory else { return }

        let state = State(categories: categories)
        let stateURL = directory.appendingPathComponent(stateFileName)

        if let data = try? JSONEncoder().encode(state) {
            try? data.write(to: stateURL)
        }
    }
}
