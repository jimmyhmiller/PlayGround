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
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(0..<virtualStore.totalLines, id: \.self) { lineNumber in
                        LogLineView(
                            virtualStore: virtualStore,
                            lineNumber: lineNumber,
                            timeFormatter: timeFormatter
                        )
                        .id(lineNumber)
                    }
                }
                .padding(.horizontal, 8)
            }
            .background(Color(red: 0.15, green: 0.15, blue: 0.16))
            .onReceive(NotificationCenter.default.publisher(for: .jumpToLogLine)) { notification in
                if let lineNumber = notification.object as? Int {
                    virtualStore.selectedIndex = lineNumber
                    withAnimation(.easeInOut(duration: 0.3)) {
                        proxy.scrollTo(lineNumber, anchor: .center)
                    }
                }
            }
        }
    }
}

struct LogLineView: View {
    let virtualStore: VirtualLogStore
    let lineNumber: Int
    let timeFormatter: DateFormatter
    
    var body: some View {
        Group {
            if let entry = virtualStore.entry(at: lineNumber) {
                HStack(alignment: .top, spacing: 8) {
                    Text(timeFormatter.string(from: entry.timestamp))
                        .font(.system(size: 11).monospaced())
                        .foregroundColor(.secondary)
                        .frame(width: 180, alignment: .leading)
                    
                    Text(entry.level.rawValue)
                        .font(.system(size: 11, weight: .medium).monospaced())
                        .foregroundColor(entry.level.color)
                        .frame(width: 50, alignment: .leading)
                    
                    Text(entry.message)
                        .font(.system(size: 11).monospaced())
                        .foregroundColor(.primary)
                        .multilineTextAlignment(.leading)
                    
                    Spacer()
                    
                    if let source = entry.source {
                        Text(source)
                            .font(.system(size: 11).monospaced())
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.vertical, 4)
                .padding(.horizontal, 8)
                .background(
                    lineNumber == virtualStore.selectedIndex ?
                    Color.accentColor.opacity(0.2) :
                    Color.clear
                )
            } else {
                HStack {
                    Text("Loading line \(lineNumber)...")
                        .foregroundColor(.secondary)
                        .font(.system(size: 11).monospaced())
                    Spacer()
                }
                .padding(.vertical, 4)
                .padding(.horizontal, 8)
            }
        }
    }
}