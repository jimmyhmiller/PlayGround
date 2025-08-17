import SwiftUI

struct LogContentView: View {
    @EnvironmentObject var logStore: LogStore
    
    private let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()
    
    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(logStore.entries.enumerated()), id: \.element.id) { index, entry in
                        LogEntryRow(entry: entry, timeFormatter: timeFormatter)
                            .id(entry.id)
                            .background(
                                index == logStore.selectedIndex ?
                                Color.accentColor.opacity(0.2) :
                                Color.clear
                            )
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }
            .background(Color(red: 0.15, green: 0.15, blue: 0.16))
            .onAppear {
                logStore.loadSampleData()
            }
            .onChange(of: logStore.selectedIndex) { newIndex in
                if let index = newIndex,
                   index < logStore.entries.count {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        proxy.scrollTo(logStore.entries[index].id, anchor: .center)
                    }
                }
            }
        }
    }
}

struct LogEntryRow: View {
    let entry: LogEntry
    let timeFormatter: DateFormatter
    
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            // Timestamp
            Text(timeFormatter.string(from: entry.timestamp))
                .font(.system(size: 11).monospaced())
                .foregroundColor(.secondary)
                .frame(width: 180, alignment: .leading)
            
            // Log level
            Text(entry.level.rawValue)
                .font(.system(size: 11, weight: .medium).monospaced())
                .foregroundColor(entry.level.color)
                .frame(width: 50, alignment: .leading)
            
            // Message
            Text(entry.message)
                .font(.system(size: 11).monospaced())
                .foregroundColor(.primary)
                .multilineTextAlignment(.leading)
            
            Spacer()
            
            // Source (right-aligned like in mockup)
            if let source = entry.source {
                Text(source)
                    .font(.system(size: 11).monospaced())
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 2)
        .padding(.horizontal, 4)
        .background(
            Rectangle()
                .fill(Color.clear)
                .background(.ultraThinMaterial, in: Rectangle())
                .opacity(0)
        )
        .onHover { isHovered in
            // Add subtle hover effect
        }
    }
}