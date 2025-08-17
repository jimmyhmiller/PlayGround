import SwiftUI

struct TimelineSidebar: View {
    @EnvironmentObject var logStore: LogStore
    
    private let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter
    }()
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color(red: 0.12, green: 0.12, blue: 0.13)
                
                if !logStore.entries.isEmpty,
                   let first = logStore.entries.first?.timestamp,
                   let last = logStore.entries.last?.timestamp {
                    
                    HStack(spacing: 0) {
                        // Time labels
                        timeLabels(height: geometry.size.height, first: first, last: last)
                            .frame(width: 35)
                        
                        // Timeline visualization
                        TimelineCanvas(entries: logStore.entries, height: geometry.size.height)
                            .frame(width: 45)
                    }
                }
            }
        }
        .onAppear {
            logStore.loadSampleData()
        }
    }
    
    @ViewBuilder
    private func timeLabels(height: CGFloat, first: Date, last: Date) -> some View {
        let labelCount = 10
        let totalDuration = last.timeIntervalSince(first)
        
        VStack(alignment: .trailing, spacing: 0) {
            ForEach(0..<labelCount, id: \.self) { i in
                let progress = Double(i) / Double(labelCount - 1)
                let time = first.addingTimeInterval(totalDuration * progress)
                
                Text(timeFormatter.string(from: time))
                    .font(.system(size: 7).monospaced())
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .frame(height: height / CGFloat(labelCount - 1), alignment: .top)
            }
        }
    }
}

struct TimelineCanvas: View {
    let entries: [LogEntry]
    let height: CGFloat
    @EnvironmentObject var logStore: LogStore
    
    var body: some View {
        Canvas { context, size in
            guard let first = entries.first?.timestamp,
                  let last = entries.last?.timestamp else { return }
            
            let totalDuration = last.timeIntervalSince(first)
            let bucketHeight: CGFloat = 2 // Each bucket is 2 pixels tall
            let bucketCount = Int(size.height / bucketHeight)
            
            // Calculate density for each bucket
            for bucketIndex in 0..<bucketCount {
                let y = CGFloat(bucketIndex) * bucketHeight
                let startProgress = Double(bucketIndex) / Double(bucketCount)
                let endProgress = Double(bucketIndex + 1) / Double(bucketCount)
                
                let startTime = first.addingTimeInterval(totalDuration * startProgress)
                let endTime = first.addingTimeInterval(totalDuration * endProgress)
                
                // Count events in this time range
                let eventsInBucket = entries.filter { entry in
                    entry.timestamp >= startTime && entry.timestamp < endTime
                }
                
                if !eventsInBucket.isEmpty {
                    // Determine color based on log levels in bucket
                    let hasError = eventsInBucket.contains { $0.level == .error }
                    let hasWarning = eventsInBucket.contains { $0.level == .warning }
                    
                    let color: Color
                    if hasError {
                        color = .red
                    } else if hasWarning {
                        color = .orange
                    } else {
                        color = .blue
                    }
                    
                    // Calculate intensity (width and opacity)
                    let intensity = min(1.0, Double(eventsInBucket.count) / 3.0)
                    let width = 5 + (intensity * 35) // 5 to 40 pixels wide
                    let opacity = 0.4 + (intensity * 0.6) // 0.4 to 1.0 opacity
                    
                    // Draw the bar
                    let rect = CGRect(x: 0, y: y, width: width, height: bucketHeight)
                    context.fill(Path(rect), with: .color(color.opacity(opacity)))
                }
            }
            
            // Draw thin vertical line at the left edge
            context.stroke(
                Path { path in
                    path.move(to: CGPoint(x: 0, y: 0))
                    path.addLine(to: CGPoint(x: 0, y: size.height))
                },
                with: .color(.secondary.opacity(0.2))
            )
        }
        .onTapGesture { location in
            guard let first = entries.first?.timestamp,
                  let last = entries.last?.timestamp else { return }
            
            let totalDuration = last.timeIntervalSince(first)
            let progress = location.y / height
            let targetTime = first.addingTimeInterval(totalDuration * progress)
            
            logStore.jumpToTime(targetTime)
        }
    }
}