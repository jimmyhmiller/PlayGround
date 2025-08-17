import SwiftUI

struct VirtualTimelineSidebar: View {
    @StateObject private var virtualStore: VirtualLogStore
    
    private let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
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
        GeometryReader { geometry in
            ZStack {
                Color(red: 0.12, green: 0.12, blue: 0.13)
                
                if virtualStore.totalLines > 0,
                   let timeRange = virtualStore.getTimeRange() {
                    
                    HStack(spacing: 0) {
                        // Time labels
                        timeLabels(height: geometry.size.height, timeRange: timeRange)
                            .frame(width: 35)
                        
                        // Timeline visualization
                        VirtualTimelineCanvas(
                            virtualStore: virtualStore,
                            timeRange: timeRange,
                            height: geometry.size.height
                        )
                        .frame(width: 45)
                    }
                }
            }
        }
    }
    
    @ViewBuilder
    private func timeLabels(height: CGFloat, timeRange: (start: Date, end: Date)) -> some View {
        let labelCount = 10
        let totalDuration = timeRange.end.timeIntervalSince(timeRange.start)
        
        VStack(alignment: .trailing, spacing: 0) {
            ForEach(0..<labelCount, id: \.self) { i in
                let progress = Double(i) / Double(labelCount - 1)
                let time = timeRange.start.addingTimeInterval(totalDuration * progress)
                
                Text(timeFormatter.string(from: time))
                    .font(.system(size: 7).monospaced())
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .frame(height: height / CGFloat(labelCount - 1), alignment: .top)
            }
        }
    }
}

struct VirtualTimelineCanvas: View {
    let virtualStore: VirtualLogStore
    let timeRange: (start: Date, end: Date)
    let height: CGFloat
    
    var body: some View {
        Canvas { context, size in
            let totalDuration = timeRange.end.timeIntervalSince(timeRange.start)
            let bucketHeight: CGFloat = 2
            let bucketCount = Int(size.height / bucketHeight)
            
            // Sample entries to determine density and colors
            for bucketIndex in 0..<bucketCount {
                let y = CGFloat(bucketIndex) * bucketHeight
                let startProgress = Double(bucketIndex) / Double(bucketCount)
                let endProgress = Double(bucketIndex + 1) / Double(bucketCount)
                
                let startTime = timeRange.start.addingTimeInterval(totalDuration * startProgress)
                let endTime = timeRange.start.addingTimeInterval(totalDuration * endProgress)
                
                // Sample a few entries in this time range to determine color and density
                let sampleCount = sampleEntriesInTimeRange(startTime..<endTime)
                
                if sampleCount.total > 0 {
                    let color: Color
                    if sampleCount.errors > 0 {
                        color = .red
                    } else if sampleCount.warnings > 0 {
                        color = .orange
                    } else {
                        color = .blue
                    }
                    
                    let intensity = min(1.0, Double(sampleCount.total) / 10.0)
                    let width = 5 + (intensity * 35)
                    let opacity = 0.4 + (intensity * 0.6)
                    
                    let rect = CGRect(x: 0, y: y, width: width, height: bucketHeight)
                    context.fill(Path(rect), with: .color(color.opacity(opacity)))
                }
            }
            
            // Draw reference line
            context.stroke(
                Path { path in
                    path.move(to: CGPoint(x: 0, y: 0))
                    path.addLine(to: CGPoint(x: 0, y: size.height))
                },
                with: .color(.secondary.opacity(0.2))
            )
        }
        .onTapGesture { location in
            let progress = location.y / height
            let targetTime = timeRange.start.addingTimeInterval(
                timeRange.end.timeIntervalSince(timeRange.start) * progress
            )
            
            if let lineNumber = virtualStore.findEntryNear(timestamp: targetTime) {
                NotificationCenter.default.post(
                    name: .jumpToLogLine,
                    object: lineNumber
                )
            }
        }
    }
    
    private func sampleEntriesInTimeRange(_ range: Range<Date>) -> (total: Int, errors: Int, warnings: Int) {
        // Convert time range to approximate line range
        let totalDuration = timeRange.end.timeIntervalSince(timeRange.start)
        let startProgress = range.lowerBound.timeIntervalSince(timeRange.start) / totalDuration
        let endProgress = range.upperBound.timeIntervalSince(timeRange.start) / totalDuration
        
        let totalLines = virtualStore.totalLines
        let startLine = Int(Double(totalLines) * startProgress)
        let endLine = min(totalLines, Int(Double(totalLines) * endProgress))
        
        // Sample a few entries in this range
        var total = 0
        var errors = 0
        var warnings = 0
        
        let sampleSize = min(10, endLine - startLine)
        if sampleSize > 0 {
            let step = max(1, (endLine - startLine) / sampleSize)
            
            for i in stride(from: startLine, to: endLine, by: step) {
                if let entry = virtualStore.entry(at: i) {
                    total += 1
                    switch entry.level {
                    case .error: errors += 1
                    case .warning: warnings += 1
                    default: break
                    }
                }
            }
        }
        
        return (total: total, errors: errors, warnings: warnings)
    }
}