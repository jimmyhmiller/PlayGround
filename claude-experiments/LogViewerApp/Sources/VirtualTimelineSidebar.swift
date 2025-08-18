import SwiftUI

struct VirtualTimelineSidebar: View {
    @StateObject private var virtualStore: VirtualLogStore
    
    private let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter
    }()
    
    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM dd"
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
                
                if !virtualStore.isInitializing && virtualStore.totalLines > 0,
                   let timeRange = virtualStore.getTimeRange() {
                    
                    VStack(spacing: 0) {
                        // Date header
                        Text(dateFormatter.string(from: timeRange.start))
                            .font(.system(size: 10, weight: .medium).monospaced())
                            .foregroundColor(.white)
                            .padding(.top, 8)
                            .padding(.bottom, 4)
                        
                        HStack(spacing: 2) {
                            // Time labels
                            timeLabels(height: geometry.size.height - 40, timeRange: timeRange)
                                .frame(width: 32)
                            
                            // Timeline visualization
                            VirtualTimelineCanvas(
                                virtualStore: virtualStore,
                                timeRange: timeRange,
                                height: geometry.size.height - 40
                            )
                            .frame(width: 44)
                        }
                        
                        Spacer()
                    }
                } else if virtualStore.isInitializing {
                    // Show loading state
                    VStack(spacing: 8) {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .scaleEffect(0.8)
                        
                        Text("Loading...")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.white.opacity(0.7))
                        
                        if virtualStore.estimatedLineCount > 0 {
                            Text("~\(formatLineCount(virtualStore.estimatedLineCount)) lines")
                                .font(.system(size: 9))
                                .foregroundColor(.white.opacity(0.5))
                        }
                    }
                }
            }
        }
    }
    
    @ViewBuilder
    private func timeLabels(height: CGFloat, timeRange: (start: Date, end: Date)) -> some View {
        let labelCount = 8
        let totalDuration = timeRange.end.timeIntervalSince(timeRange.start)
        
        VStack(alignment: .trailing, spacing: 0) {
            ForEach(0..<labelCount, id: \.self) { i in
                let progress = Double(i) / Double(labelCount - 1)
                let time = timeRange.start.addingTimeInterval(totalDuration * progress)
                
                VStack(alignment: .trailing, spacing: 1) {
                    Text(timeFormatter.string(from: time))
                        .font(.system(size: 9, weight: .medium).monospaced())
                        .foregroundColor(.white.opacity(0.9))
                    
                    // Add a subtle tick mark
                    Rectangle()
                        .fill(Color.white.opacity(0.3))
                        .frame(width: 8, height: 1)
                }
                .frame(maxWidth: .infinity, alignment: .trailing)
                .frame(height: height / CGFloat(labelCount - 1), alignment: i == 0 ? .top : (i == labelCount - 1 ? .bottom : .center))
            }
        }
    }
    
    private func formatLineCount(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1fM", Double(count) / 1_000_000)
        } else if count >= 1_000 {
            return String(format: "%.1fK", Double(count) / 1_000)
        } else {
            return "\(count)"
        }
    }
}

struct VirtualTimelineCanvas: View {
    let virtualStore: VirtualLogStore
    let timeRange: (start: Date, end: Date)
    let height: CGFloat
    
    var body: some View {
        Canvas { context, size in
            // Draw background with subtle grid
            drawBackground(context: context, size: size)
            
            let totalDuration = timeRange.end.timeIntervalSince(timeRange.start)
            let bucketHeight: CGFloat = 1
            let bucketCount = Int(size.height / bucketHeight)
            
            // First pass: collect all intensities to normalize
            var maxIntensity: Double = 1.0
            var bucketData: [(intensity: Double, color: Color)] = []
            
            for bucketIndex in 0..<bucketCount {
                let startProgress = Double(bucketIndex) / Double(bucketCount)
                let endProgress = Double(bucketIndex + 1) / Double(bucketCount)
                
                let startTime = timeRange.start.addingTimeInterval(totalDuration * startProgress)
                let endTime = timeRange.start.addingTimeInterval(totalDuration * endProgress)
                
                let sampleCount = sampleEntriesInTimeRange(startTime..<endTime)
                
                if sampleCount.total > 0 {
                    let color: Color
                    if sampleCount.errors > 0 {
                        color = .red
                    } else if sampleCount.warnings > 0 {
                        color = .orange
                    } else {
                        color = Color(red: 0.3, green: 0.7, blue: 1.0) // Brighter blue
                    }
                    
                    let intensity = Double(sampleCount.total)
                    maxIntensity = max(maxIntensity, intensity)
                    bucketData.append((intensity: intensity, color: color))
                } else {
                    bucketData.append((intensity: 0, color: .clear))
                }
            }
            
            // Second pass: draw normalized bars
            for (bucketIndex, data) in bucketData.enumerated() {
                if data.intensity > 0 {
                    let y = CGFloat(bucketIndex) * bucketHeight
                    let normalizedIntensity = data.intensity / maxIntensity
                    
                    // Much more visible sizing
                    let width = 8 + (normalizedIntensity * 32) // Minimum 8px, max 40px
                    let opacity = 0.7 + (normalizedIntensity * 0.3) // 0.7 to 1.0 opacity
                    
                    let rect = CGRect(x: 2, y: y, width: width, height: bucketHeight)
                    context.fill(Path(rect), with: .color(data.color.opacity(opacity)))
                    
                    // Add subtle glow for high intensity
                    if normalizedIntensity > 0.7 {
                        let glowRect = CGRect(x: 1, y: y, width: width + 2, height: bucketHeight + 1)
                        context.fill(Path(glowRect), with: .color(data.color.opacity(0.3)))
                    }
                }
            }
            
            // Draw enhanced border and grid
            drawBorderAndGrid(context: context, size: size)
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
    
    private func drawBackground(context: GraphicsContext, size: CGSize) {
        // Draw subtle background grid
        let gridSpacing: CGFloat = 20
        let gridColor = Color.white.opacity(0.05)
        
        // Horizontal grid lines
        for y in stride(from: 0, through: size.height, by: gridSpacing) {
            let path = Path { path in
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
            }
            context.stroke(path, with: .color(gridColor), lineWidth: 0.5)
        }
    }
    
    private func drawBorderAndGrid(context: GraphicsContext, size: CGSize) {
        // Draw left border
        let borderPath = Path { path in
            path.move(to: CGPoint(x: 0, y: 0))
            path.addLine(to: CGPoint(x: 0, y: size.height))
        }
        context.stroke(borderPath, with: .color(.white.opacity(0.4)), lineWidth: 1)
        
        // Draw right border 
        let rightBorderPath = Path { path in
            path.move(to: CGPoint(x: size.width, y: 0))
            path.addLine(to: CGPoint(x: size.width, y: size.height))
        }
        context.stroke(rightBorderPath, with: .color(.white.opacity(0.2)), lineWidth: 1)
    }
    
    private func sampleEntriesInTimeRange(_ range: Range<Date>) -> (total: Int, errors: Int, warnings: Int) {
        // Convert time range to approximate line range
        let totalDuration = timeRange.end.timeIntervalSince(timeRange.start)
        let startProgress = range.lowerBound.timeIntervalSince(timeRange.start) / totalDuration
        let endProgress = range.upperBound.timeIntervalSince(timeRange.start) / totalDuration
        
        let totalLines = virtualStore.totalLines
        let startLine = Int(Double(totalLines) * startProgress)
        let endLine = min(totalLines, Int(Double(totalLines) * endProgress))
        
        // Much lighter sampling - just estimate based on line count for now
        let lineCount = max(0, endLine - startLine)
        
        // Simple heuristic: assume some activity if there are lines
        // In a real implementation, this could be improved with a separate
        // density index that doesn't require parsing individual entries
        if lineCount > 0 {
            // Rough estimates without actually parsing entries
            let total = min(lineCount / 10, 50) // Scale down for visualization
            let errors = total / 20 // Assume ~5% errors
            let warnings = total / 10 // Assume ~10% warnings
            
            return (total: total, errors: errors, warnings: warnings)
        }
        
        return (total: 0, errors: 0, warnings: 0)
    }
}