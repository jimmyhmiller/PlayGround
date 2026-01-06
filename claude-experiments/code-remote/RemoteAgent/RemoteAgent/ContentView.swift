import SwiftUI
import SwiftData

struct ContentView: View {
    var body: some View {
        NavigationStack {
            ServerListView()
        }
    }
}

#Preview {
    ContentView()
        .modelContainer(for: [Server.self, Project.self], inMemory: true)
}
