import SwiftUI

struct VirtualLogContentView: View {
    @StateObject private var virtualStore: VirtualLogStore
    
    private let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()
    
    init(url: URL) {
        // Create a fallback store if the real one fails to initialize
        if let store = VirtualLogStore(url: url) {
            self._virtualStore = StateObject(wrappedValue: store)
        } else {
            // Create a dummy store with empty data
            let dummyURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("empty.log")
            try? "".write(to: dummyURL, atomically: true, encoding: .utf8)
            let fallbackStore = VirtualLogStore(url: dummyURL)!
            self._virtualStore = StateObject(wrappedValue: fallbackStore)
        }
    }
    
    var body: some View {
        HexFiendVirtualScrollView(
            itemCount: virtualStore.totalLines,
            itemHeight: 22,
            selectedIndex: $virtualStore.selectedIndex,
            content: { lineNumber in
            if let entry = virtualStore.entry(at: lineNumber) {
                LogEntryRow(entry: entry, timeFormatter: timeFormatter)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .background(
                        lineNumber == virtualStore.selectedIndex ?
                        Color.accentColor.opacity(0.2) :
                        Color.clear
                    )
            } else {
                HStack {
                    Text("Error loading line \(lineNumber)")
                        .foregroundColor(.red)
                        .font(.system(size: 11).monospaced())
                    Spacer()
                }
                .padding(.horizontal, 12)
            }
        })
        .onReceive(NotificationCenter.default.publisher(for: .jumpToLogLine)) { notification in
            if let lineNumber = notification.object as? Int {
                virtualStore.selectedIndex = lineNumber
                // TODO: Implement scroll to line in HexFiendVirtualScrollView
            }
        }
    }
}