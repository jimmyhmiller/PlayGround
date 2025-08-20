import SwiftUI
import AppKit
import Combine

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
        VirtualLogScrollView(
            virtualStore: virtualStore,
            timeFormatter: timeFormatter
        )
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

// MARK: - UIScrollView Implementation

class ScrollCoordinator: ObservableObject {
    weak var scrollView: NSScrollView?
    
    func scrollToLine(_ lineNumber: Int, lineHeight: CGFloat = 25.0) {
        guard let scrollView = scrollView else { return }
        
        let targetY = CGFloat(lineNumber) * lineHeight
        let visibleHeight = scrollView.contentView.bounds.height
        let maxY = scrollView.documentView?.bounds.height ?? 0
        
        // Adjust scroll position so the line is centered in the view
        // For the last lines, make sure we don't scroll past the end
        let centeredY = targetY - (visibleHeight / 2)
        let clampedY = max(0, min(centeredY, maxY - visibleHeight))
        
        let point = NSPoint(x: 0, y: clampedY)
        
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.3
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            scrollView.contentView.animator().setBoundsOrigin(point)
        }
    }
}

struct VirtualLogScrollView: NSViewRepresentable {
    let virtualStore: VirtualLogStore
    let timeFormatter: DateFormatter
    @ObservedObject var coordinator = ScrollCoordinator()
    
    private let lineHeight: CGFloat = 25.0
    
    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        
        // Configure scroll view
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = false
        scrollView.backgroundColor = NSColor(red: 0.15, green: 0.15, blue: 0.16, alpha: 1.0)
        
        // Create and configure the content view
        let contentView = VirtualLogNSView(
            virtualStore: virtualStore,
            timeFormatter: timeFormatter,
            lineHeight: lineHeight
        )
        
        // Set content size
        let totalHeight = CGFloat(virtualStore.totalLines) * lineHeight
        contentView.frame = NSRect(x: 0, y: 0, width: 800, height: totalHeight)
        
        scrollView.documentView = contentView
        coordinator.scrollView = scrollView
        
        // Listen for scroll notifications
        NotificationCenter.default.addObserver(
            forName: .jumpToLogLine,
            object: nil,
            queue: .main
        ) { notification in
            if let lineNumber = notification.object as? Int {
                self.virtualStore.selectedIndex = lineNumber
                self.coordinator.scrollToLine(lineNumber, lineHeight: self.lineHeight)
            }
        }
        
        return scrollView
    }
    
    func updateNSView(_ nsView: NSScrollView, context: Context) {
        // Update content size if total lines changed
        let totalHeight = CGFloat(virtualStore.totalLines) * lineHeight
        if let contentView = nsView.documentView {
            let newFrame = NSRect(x: 0, y: 0, width: max(800, nsView.frame.width), height: totalHeight)
            if contentView.frame != newFrame {
                contentView.frame = newFrame
                contentView.needsDisplay = true
            }
        }
    }
}

class VirtualLogNSView: NSView {
    let virtualStore: VirtualLogStore
    let timeFormatter: DateFormatter
    let lineHeight: CGFloat
    private var cancellables: Set<AnyCancellable> = []
    
    init(virtualStore: VirtualLogStore, timeFormatter: DateFormatter, lineHeight: CGFloat) {
        self.virtualStore = virtualStore
        self.timeFormatter = timeFormatter
        self.lineHeight = lineHeight
        super.init(frame: .zero)
        
        self.wantsLayer = true
        self.layer?.backgroundColor = NSColor(red: 0.15, green: 0.15, blue: 0.16, alpha: 1.0).cgColor
        
        // Observe changes to trigger redraws
        virtualStore.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.needsDisplay = true
            }
            .store(in: &cancellables)
    }
    
    // Flip coordinates to have origin at top-left like a text view
    override var isFlipped: Bool {
        return true
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        
        // Calculate which lines are visible
        let startLine = max(0, Int(dirtyRect.minY / lineHeight))
        let endLine = min(virtualStore.totalLines - 1, Int(dirtyRect.maxY / lineHeight) + 1)
        
        // Safety check to prevent invalid range
        guard startLine <= endLine && virtualStore.totalLines > 0 else {
            return
        }
        
        // Draw visible lines
        for lineNumber in startLine...endLine {
            let lineRect = NSRect(
                x: 8,
                y: CGFloat(lineNumber) * lineHeight,
                width: dirtyRect.width - 16,
                height: lineHeight
            )
            
            drawLine(lineNumber: lineNumber, in: lineRect)
        }
    }
    
    private func drawLine(lineNumber: Int, in rect: NSRect) {
        guard let entry = virtualStore.entry(at: lineNumber) else {
            // Draw loading placeholder
            let loadingText = "Loading line \(lineNumber)..."
            let attributes: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
                .foregroundColor: NSColor.secondaryLabelColor
            ]
            
            let attributedString = NSAttributedString(string: loadingText, attributes: attributes)
            let size = attributedString.size()
            let drawRect = NSRect(
                x: rect.minX,
                y: rect.minY + (rect.height - size.height) / 2,
                width: size.width,
                height: size.height
            )
            attributedString.draw(in: drawRect)
            return
        }
        
        // Draw selection background if this line is selected
        if lineNumber == virtualStore.selectedIndex {
            NSColor.controlAccentColor.withAlphaComponent(0.2).setFill()
            rect.fill()
        }
        
        // Draw timestamp
        let timestampStr = timeFormatter.string(from: entry.timestamp)
        let timestampAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
            .foregroundColor: NSColor.secondaryLabelColor
        ]
        
        let timestampRect = NSRect(x: rect.minX, y: rect.minY + 4, width: 180, height: rect.height - 8)
        NSAttributedString(string: timestampStr, attributes: timestampAttributes).draw(in: timestampRect)
        
        // Draw level
        let levelColor: NSColor = {
            switch entry.level {
            case .error: return NSColor.systemRed
            case .warning: return NSColor.systemOrange
            case .debug: return NSColor.systemGray
            default: return NSColor.labelColor
            }
        }()
        
        let levelAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .medium),
            .foregroundColor: levelColor
        ]
        
        let levelRect = NSRect(x: rect.minX + 190, y: rect.minY + 4, width: 50, height: rect.height - 8)
        NSAttributedString(string: entry.level.rawValue, attributes: levelAttributes).draw(in: levelRect)
        
        // Draw message
        let messageAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
            .foregroundColor: NSColor.labelColor
        ]
        
        let messageRect = NSRect(x: rect.minX + 250, y: rect.minY + 4, width: rect.width - 250, height: rect.height - 8)
        NSAttributedString(string: entry.message, attributes: messageAttributes).draw(in: messageRect)
    }
}