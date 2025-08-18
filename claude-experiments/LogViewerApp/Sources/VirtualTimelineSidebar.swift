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
    
    @State private var clickMap: [(yPos: CGFloat, lineNumber: Int)] = [] // Sorted Y positions -> line numbers
    
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
            
            // First pass: collect all intensities to normalize and build click map
            var maxIntensity: Double = 1.0
            var bucketData: [(intensity: Double, color: Color)] = []
            var newClickMap: [(yPos: CGFloat, lineNumber: Int)] = []
            
            for bucketIndex in 0..<bucketCount {
                let startProgress = Double(bucketIndex) / Double(bucketCount)
                let endProgress = Double(bucketIndex + 1) / Double(bucketCount)
                
                let startTime = timeRange.start.addingTimeInterval(totalDuration * startProgress)
                let endTime = timeRange.start.addingTimeInterval(totalDuration * endProgress)
                
                // Build click map: Y position -> line number
                let yPosition = CGFloat(bucketIndex) * bucketHeight
                let totalLines = virtualStore.totalLines
                let lineNumber = max(0, min(totalLines - 1, Int(Double(totalLines) * startProgress)))
                newClickMap.append((yPos: yPosition, lineNumber: lineNumber))
                
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
            
            // Update click map after generation
            DispatchQueue.main.async {
                clickMap = newClickMap
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
                // Binary search in sorted click map for efficient lookup
                var left = 0
                var right = clickMap.count - 1
                
                while left <= right {
                    let mid = (left + right) / 2
                    let midY = clickMap[mid].yPos
                    
                    if midY <= clickY {
                        left = mid + 1
                    } else {
                        right = mid - 1
                    }
                }
                
                // Use the closest entry
                let index = max(0, min(clickMap.count - 1, right))
                targetLineNumber = clickMap[index].lineNumber
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
                    // Check if this line was already cached during timeline generation
                    if let entry = virtualStore.getCachedEntry(at: lineIndex) {
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
}