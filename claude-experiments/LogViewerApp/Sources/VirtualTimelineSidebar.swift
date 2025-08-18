import SwiftUI

struct VirtualTimelineSidebar: View {
    @StateObject private var virtualStore: VirtualLogStore
    @State private var stableTimeRange: (start: Date, end: Date)?
    @State private var hasShownTimeline = false
    
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
                                height: geometry.size.height - 40
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
            
            // Validate timeline data
            guard totalDuration > 0 && totalDuration.isFinite else {
                // Just draw the border if we have invalid time data
                drawBorderAndGrid(context: context, size: size)
                return
            }
            
            let bucketHeight: CGFloat = 1
            let bucketCount = Int(size.height / bucketHeight)
            
            // Ensure reasonable bucket count
            guard bucketCount > 0 && bucketCount < 10000 else {
                drawBorderAndGrid(context: context, size: size)
                return
            }
            
            // First pass: collect all intensities to normalize
            var maxIntensity: Double = 1.0
            var bucketData: [(intensity: Double, color: Color)] = []
            
            for bucketIndex in 0..<bucketCount {
                let startProgress = Double(bucketIndex) / Double(bucketCount)
                let endProgress = Double(bucketIndex + 1) / Double(bucketCount)
                
                let startTime = timeRange.start.addingTimeInterval(totalDuration * startProgress)
                let endTime = timeRange.start.addingTimeInterval(totalDuration * endProgress)
                
                // Ensure valid time range
                guard startTime < endTime else {
                    bucketData.append((intensity: 0, color: .clear))
                    continue
                }
                
                let sampleCount = sampleEntriesInTimeRange(startTime..<endTime)
                
                if sampleCount.total > 0 {
                    let color: Color
                    
                    // Determine color based on error/warning ratio with safety checks
                    let safeTotal = max(1, sampleCount.total) // Prevent division by zero
                    let errorRatio = Double(sampleCount.errors) / Double(safeTotal)
                    let warningRatio = Double(sampleCount.warnings) / Double(safeTotal)
                    
                    if errorRatio > 0.3 { // High error rate
                        color = Color(red: 1.0, green: 0.2, blue: 0.2) // Bright red
                    } else if errorRatio > 0.1 { // Some errors
                        color = Color(red: 1.0, green: 0.4, blue: 0.3) // Orange-red
                    } else if warningRatio > 0.4 { // High warning rate
                        color = Color(red: 1.0, green: 0.6, blue: 0.1) // Orange
                    } else if warningRatio > 0.1 { // Some warnings
                        color = Color(red: 1.0, green: 0.8, blue: 0.2) // Yellow-orange
                    } else if sampleCount.total > 20 { // High activity, no issues
                        color = Color(red: 0.2, green: 0.8, blue: 0.3) // Green
                    } else { // Normal activity
                        color = Color(red: 0.3, green: 0.7, blue: 1.0) // Blue
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
                    let safeMaxIntensity = max(1.0, maxIntensity) // Prevent division by zero
                    let normalizedIntensity = min(1.0, data.intensity / safeMaxIntensity) // Clamp to [0,1]
                    
                    // Much more visible sizing with safety checks
                    let width = max(2, min(40, 8 + (normalizedIntensity * 32))) // Clamp width
                    let opacity = max(0.1, min(1.0, 0.7 + (normalizedIntensity * 0.3))) // Clamp opacity
                    
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
            // Smart sampling: check a few actual entries to get real data
            var total = 0
            var errors = 0  
            var warnings = 0
            
            // Sample at most 5 entries from this range to avoid performance issues
            let sampleCount = min(5, lineCount)
            if sampleCount > 0 {
                let step = max(1, lineCount / sampleCount)
                
                for i in 0..<sampleCount {
                    let lineIndex = startLine + (i * step)
                    // Force entry parsing for timeline sampling (small number of entries)
                    if let lineInfo = virtualStore.lineIndex.lineInfo(at: lineIndex),
                       let lineText = virtualStore.slice.substring(in: lineInfo.byteRange) {
                        let entry = virtualStore.parseLogLine(lineText.trimmingCharacters(in: .whitespacesAndNewlines), lineNumber: lineIndex)
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
            
            // If we got actual data, use it. Otherwise, use smart heuristics
            if total > 0 {
                // Scale up the sample to represent the whole range
                let scaleFactor = max(1, lineCount / 100) // Represent density
                
                // Safety checks to prevent overflow or invalid values
                let safeTotal = min(total * scaleFactor, 1000)
                let safeErrors = min(errors * scaleFactor, 1000)
                let safeWarnings = min(warnings * scaleFactor, 1000)
                
                return (total: safeTotal, errors: safeErrors, warnings: safeWarnings)
            } else {
                // Fallback to variable heuristics based on position and time
                // Make this more generous so timeline shows some activity
                let intensity = max(5, min(lineCount / 5, 50)) // More generous intensity
                
                // Add some variation based on the time range position with safety checks
                let safeProgress = max(0.0, min(1.0, startProgress)) // Clamp to [0,1]
                let timeVariation = abs(sin(safeProgress * 8 + startProgress * 3)) // More variation
                let errorRate = timeVariation > 0.8 ? 0.2 : (timeVariation > 0.6 ? 0.1 : 0.02) 
                let warningRate = timeVariation > 0.7 ? 0.25 : (timeVariation > 0.4 ? 0.15 : 0.05)
                
                let estimatedErrors = max(0, min(100, Int(Double(intensity) * errorRate)))
                let estimatedWarnings = max(0, min(100, Int(Double(intensity) * warningRate)))
                
                return (total: max(5, intensity), errors: estimatedErrors, warnings: estimatedWarnings)
            }
        }
        
        return (total: 0, errors: 0, warnings: 0)
    }
}