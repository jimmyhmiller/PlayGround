import SwiftUI

struct VirtualTimelineSidebar: View {
    @StateObject private var virtualStore: VirtualLogStore
    @State private var stableTimeRange: (start: Date, end: Date)?
    @State private var hasShownTimeline = false
    @State private var timeMarkers: [(position: CGFloat, time: Date, lineNumber: Int)] = []
    
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
                
                if let timeRange = stableTimeRange ?? virtualStore.getTimeRange(),
                   virtualStore.totalLines > 0 {
                    
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
                                height: geometry.size.height - 40,
                                onTimeMarkersGenerated: { markers in
                                    self.timeMarkers = markers
                                }
                            )
                            .frame(width: 44)
                        }
                        
                        Spacer()
                    }
                    .onAppear {
                        // Cache the time range once we have it to prevent disappearing
                        if stableTimeRange == nil {
                            stableTimeRange = timeRange
                            hasShownTimeline = true
                        }
                    }
                } else if hasShownTimeline && stableTimeRange != nil {
                    // Show timeline with stable cached data even if other conditions fail
                    VStack(spacing: 0) {
                        Text(dateFormatter.string(from: stableTimeRange!.start))
                            .font(.system(size: 10, weight: .medium).monospaced())
                            .foregroundColor(.white)
                            .padding(.top, 8)
                            .padding(.bottom, 4)
                        
                        HStack(spacing: 2) {
                            timeLabels(height: geometry.size.height - 40, timeRange: stableTimeRange!)
                                .frame(width: 32)
                            
                            VirtualTimelineCanvas(
                                virtualStore: virtualStore,
                                timeRange: stableTimeRange!,
                                height: geometry.size.height - 40,
                                onTimeMarkersGenerated: { markers in
                                    self.timeMarkers = markers
                                }
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
        if !timeMarkers.isEmpty {
            // Use actual time markers from the timeline data
            ZStack(alignment: .topTrailing) {
                ForEach(Array(timeMarkers.enumerated()), id: \.offset) { index, marker in
                    VStack(alignment: .trailing, spacing: 1) {
                        Text(timeFormatter.string(from: marker.time))
                            .font(.system(size: 9, weight: .medium).monospaced())
                            .foregroundColor(.white.opacity(0.9))
                        
                        Rectangle()
                            .fill(Color.white.opacity(0.3))
                            .frame(width: 8, height: 1)
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .position(x: 16, y: marker.position - 4.5) // Move up by half font height (9pt ≈ 9px)
                    .onTapGesture {
                        // Find the earliest entry in this minute
                        if let lineNumber = virtualStore.findEarliestEntryInMinute(for: marker.time) {
                            NotificationCenter.default.post(
                                name: .jumpToLogLine,
                                object: lineNumber
                            )
                        }
                    }
                }
            }
            .frame(height: height)
        } else {
            // Fallback to evenly spaced labels if no markers available
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
                        
                        Rectangle()
                            .fill(Color.white.opacity(0.3))
                            .frame(width: 8, height: 1)
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .frame(height: height / CGFloat(labelCount - 1), alignment: i == 0 ? .top : (i == labelCount - 1 ? .bottom : .center))
                    .onTapGesture {
                        // Find the earliest entry in this minute
                        if let lineNumber = virtualStore.findEarliestEntryInMinute(for: time) {
                            NotificationCenter.default.post(
                                name: .jumpToLogLine,
                                object: lineNumber
                            )
                        }
                    }
                }
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
    let onTimeMarkersGenerated: ([(position: CGFloat, time: Date, lineNumber: Int)]) -> Void
    
    @State private var clickMap: [(yPos: CGFloat, lineNumber: Int)] = [] // Sorted Y positions -> line numbers
    @State private var timelineData: TimelineData? = nil
    @State private var lastHeight: CGFloat = 0
    
    struct TimelineData {
        let buckets: [(color: Color, intensity: Double, lineNumber: Int, timestamp: Date?)]
        let timeMarkers: [(bucketIndex: Int, time: Date)]  // Store bucket index instead of yPos
        let maxIntensity: Double
    }
    
    var body: some View {
        Canvas { context, size in
            // Draw background with subtle grid
            drawBackground(context: context, size: size)
            
            // Invalidate cache if height changed significantly
            let heightChanged = abs(size.height - lastHeight) > 1
            
            // Use cached timeline data if available and size hasn't changed, otherwise generate it
            let data: TimelineData
            if let cached = timelineData, !heightChanged {
                data = cached
            } else {
                guard let generated = generateTimelineData(height: size.height) else {
                    drawBorderAndGrid(context: context, size: size)
                    return
                }
                data = generated
                // Cache it asynchronously and notify about time markers
                DispatchQueue.main.async {
                    self.timelineData = generated
                    self.lastHeight = size.height
                    // Convert bucket indices to actual Y positions based on canvas size
                    let bucketHeight = size.height / CGFloat(generated.buckets.count)
                    let markers = generated.timeMarkers.map { marker in
                        let bucket = generated.buckets[marker.bucketIndex]
                        return (position: CGFloat(marker.bucketIndex) * bucketHeight + bucketHeight / 2, 
                                time: marker.time,
                                lineNumber: bucket.lineNumber)
                    }
                    self.onTimeMarkersGenerated(markers)
                }
            }
            
            // Always recalculate time marker positions based on current size
            if heightChanged && timelineData != nil {
                DispatchQueue.main.async {
                    let bucketHeight = size.height / CGFloat(data.buckets.count)
                    let markers = data.timeMarkers.map { marker in
                        let bucket = data.buckets[marker.bucketIndex]
                        return (position: CGFloat(marker.bucketIndex) * bucketHeight + bucketHeight / 2, 
                                time: marker.time,
                                lineNumber: bucket.lineNumber)
                    }
                    self.onTimeMarkersGenerated(markers)
                }
            }
            
            // Calculate bucket height to fill the entire canvas
            let bucketHeight = size.height / CGFloat(data.buckets.count)
            
            // Draw the timeline bars from cached data
            for (index, bucket) in data.buckets.enumerated() {
                if bucket.intensity > 0 {
                    let y = CGFloat(index) * bucketHeight
                    let safeMaxIntensity = max(1.0, data.maxIntensity)
                    let normalizedIntensity = min(1.0, bucket.intensity / safeMaxIntensity)
                    
                    let width = max(2, min(40, 8 + (normalizedIntensity * 32)))
                    let opacity = max(0.1, min(1.0, 0.7 + (normalizedIntensity * 0.3)))
                    
                    let rect = CGRect(x: 2, y: y, width: width, height: bucketHeight)
                    context.fill(Path(rect), with: .color(bucket.color.opacity(opacity)))
                    
                    // Add subtle glow for high intensity
                    if normalizedIntensity > 0.7 {
                        let glowRect = CGRect(x: 1, y: y, width: width + 2, height: bucketHeight + 1)
                        context.fill(Path(glowRect), with: .color(bucket.color.opacity(0.3)))
                    }
                }
            }
            
            // Draw enhanced border and grid
            drawBorderAndGrid(context: context, size: size)
            
            // Update click map from timeline data with proper Y positions
            DispatchQueue.main.async {
                let bucketHeight = size.height / CGFloat(data.buckets.count)
                self.clickMap = data.buckets.enumerated().map { index, bucket in
                    (yPos: CGFloat(index) * bucketHeight, lineNumber: bucket.lineNumber)
                }
            }
        }
        .onTapGesture { location in
            // Use pre-computed click map for instant lookup
            let clickY = location.y
            
            let targetLineNumber: Int
            if clickMap.isEmpty {
                // Fallback to calculation if map is empty
                let progress = location.y / height
                let totalLines = virtualStore.totalLines
                targetLineNumber = max(0, min(totalLines - 1, Int(Double(totalLines) * progress)))
            } else {
                // Special case: if clicking at the very bottom, go to last line
                if clickY >= height - 5 { // Within 5 pixels of bottom
                    targetLineNumber = virtualStore.totalLines - 1
                } else {
                    // Find the bucket that contains this Y position
                    var closestIndex = 0
                    var closestDistance = CGFloat.infinity
                    
                    for (index, entry) in clickMap.enumerated() {
                        let distance = abs(entry.yPos - clickY)
                        if distance < closestDistance {
                            closestDistance = distance
                            closestIndex = index
                        }
                    }
                    
                    targetLineNumber = clickMap[closestIndex].lineNumber
                }
            }
            
            // No need to pre-index - data should already be cached from timeline generation!
            
            NotificationCenter.default.post(
                name: .jumpToLogLine,
                object: targetLineNumber
            )
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
        // Convert time range to approximate line range with safety checks
        let totalDuration = timeRange.end.timeIntervalSince(timeRange.start)
        
        // Safety check for invalid duration
        guard totalDuration > 0 && totalDuration.isFinite else {
            return (total: 0, errors: 0, warnings: 0)
        }
        
        let startInterval = range.lowerBound.timeIntervalSince(timeRange.start)
        let endInterval = range.upperBound.timeIntervalSince(timeRange.start)
        
        // Safety checks for time intervals
        guard startInterval.isFinite && endInterval.isFinite else {
            return (total: 0, errors: 0, warnings: 0)
        }
        
        let startProgress = max(0.0, min(1.0, startInterval / totalDuration))
        let endProgress = max(0.0, min(1.0, endInterval / totalDuration))
        
        let totalLines = virtualStore.totalLines
        
        // Safety checks for line calculations
        let startLineDouble = Double(totalLines) * startProgress
        let endLineDouble = Double(totalLines) * endProgress
        
        guard startLineDouble.isFinite && endLineDouble.isFinite else {
            return (total: 0, errors: 0, warnings: 0)
        }
        
        let startLine = max(0, min(totalLines - 1, Int(startLineDouble)))
        let endLine = max(0, min(totalLines, Int(endLineDouble)))
        
        let lineCount = max(0, endLine - startLine)
        
        if lineCount > 0 {
            // Try limited non-blocking sampling for real colors
            var total = 0
            var errors = 0  
            var warnings = 0
            
            // Sample at most 2 entries per bucket to minimize performance impact
            let sampleCount = min(2, lineCount)
            if sampleCount > 0 {
                let step = max(1, lineCount / sampleCount)
                
                for i in 0..<sampleCount {
                    let lineIndex = startLine + (i * step)
                    // Use non-blocking entry access to get real log levels for colors
                    if let entry = virtualStore.entry(at: lineIndex) {
                        total += 1
                        switch entry.level {
                        case .error: 
                            errors += 1
                        case .warning: 
                            warnings += 1
                        default: 
                            break
                        }
                    }
                }
            }
            
            // If we got real data, use it with scaling
            if total > 0 {
                let scaleFactor = max(1, lineCount / 20) // Represent density
                let safeTotal = min(total * scaleFactor, 1000)
                let safeErrors = min(errors * scaleFactor, 1000)
                let safeWarnings = min(warnings * scaleFactor, 1000)
                
                return (total: safeTotal, errors: safeErrors, warnings: safeWarnings)
            } else {
                // Fallback to smart heuristics if no entries were cached
                let intensity = max(5, min(lineCount / 10, 30))
                let safeProgress = max(0.0, min(1.0, startProgress))
                let timeVariation = abs(sin(safeProgress * 6 + startProgress * 2))
                let errorRate = timeVariation > 0.8 ? 0.15 : 0.03
                let warningRate = timeVariation > 0.6 ? 0.2 : 0.05
                
                let estimatedErrors = Int(Double(intensity) * errorRate)
                let estimatedWarnings = Int(Double(intensity) * warningRate)
                
                return (total: intensity, errors: estimatedErrors, warnings: estimatedWarnings)
            }
        }
        
        return (total: 0, errors: 0, warnings: 0)
    }
    
    /// Generate timeline data by sampling log entries
    private func generateTimelineData(height: CGFloat) -> TimelineData? {
        // Use a reasonable number of buckets that scales with height but has limits
        let targetBucketHeight: CGFloat = 2.0 // Target 2 pixels per bucket for good granularity
        let idealBucketCount = Int(height / targetBucketHeight)
        let bucketCount = min(max(50, idealBucketCount), 500) // Clamp between 50 and 500 buckets
        
        guard bucketCount > 0 else { return nil }
        
        let totalLines = virtualStore.totalLines
        guard totalLines > 0 else { return nil }
        
        var buckets: [(color: Color, intensity: Double, lineNumber: Int, timestamp: Date?)] = []
        var timeMarkers: [(bucketIndex: Int, time: Date)] = []
        var maxIntensity: Double = 1.0
        
        // Sample strategy: divide lines into buckets and sample each bucket
        let linesPerBucket = max(1, totalLines / bucketCount)
        
        for bucketIndex in 0..<bucketCount {
            let startLine = bucketIndex * linesPerBucket
            let endLine = min((bucketIndex + 1) * linesPerBucket, totalLines)
            
            // Sample a few entries from this bucket to determine color and get actual timestamp
            var bucketErrors = 0
            var bucketWarnings = 0
            var bucketTotal = 0
            var bucketTimestamp: Date?
            
            // Sample up to 5 entries per bucket for color determination
            let sampleStep = max(1, (endLine - startLine) / 5)
            for lineNum in stride(from: startLine, to: endLine, by: sampleStep) {
                if let entry = virtualStore.entry(at: lineNum) {
                    bucketTotal += 1
                    if bucketTimestamp == nil {
                        bucketTimestamp = entry.timestamp
                    }
                    switch entry.level {
                    case .error: bucketErrors += 1
                    case .warning: bucketWarnings += 1
                    default: break
                    }
                }
            }
            
            // Determine color based on sampled entries
            let color: Color
            let intensity: Double
            
            if bucketTotal > 0 {
                let errorRatio = Double(bucketErrors) / Double(bucketTotal)
                let warningRatio = Double(bucketWarnings) / Double(bucketTotal)
                
                if errorRatio > 0.3 {
                    color = Color(red: 1.0, green: 0.2, blue: 0.2)
                } else if errorRatio > 0.1 {
                    color = Color(red: 1.0, green: 0.4, blue: 0.3)
                } else if warningRatio > 0.4 {
                    color = Color(red: 1.0, green: 0.6, blue: 0.1)
                } else if warningRatio > 0.1 {
                    color = Color(red: 1.0, green: 0.8, blue: 0.2)
                } else if endLine - startLine > 20 {
                    color = Color(red: 0.2, green: 0.8, blue: 0.3)
                } else {
                    color = Color(red: 0.3, green: 0.7, blue: 1.0)
                }
                
                // Intensity based on line density
                intensity = Double(endLine - startLine)
                maxIntensity = max(maxIntensity, intensity)
            } else {
                color = .clear
                intensity = 0
            }
            
            buckets.append((color: color, intensity: intensity, lineNumber: startLine, timestamp: bucketTimestamp))
            
            // Add time marker for this bucket (we'll filter later for display)
            if let timestamp = bucketTimestamp {
                timeMarkers.append((bucketIndex: bucketIndex, time: timestamp))
            }
        }
        
        // Filter time markers to show only ~8 well-distributed ones
        let filteredMarkers = filterTimeMarkers(timeMarkers, targetCount: 8)
        
        return TimelineData(buckets: buckets, timeMarkers: filteredMarkers, maxIntensity: maxIntensity)
    }
    
    /// Filter time markers to show a reasonable number with good distribution
    private func filterTimeMarkers(_ markers: [(bucketIndex: Int, time: Date)], targetCount: Int) -> [(bucketIndex: Int, time: Date)] {
        guard markers.count > targetCount else { return markers }
        
        var filtered: [(bucketIndex: Int, time: Date)] = []
        let step = markers.count / targetCount
        
        for i in 0..<targetCount {
            let index = min(i * step, markers.count - 1)
            filtered.append(markers[index])
        }
        
        return filtered
    }
}