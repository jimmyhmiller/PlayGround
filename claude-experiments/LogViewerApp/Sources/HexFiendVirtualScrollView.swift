// Virtual scroll view inspired by HexFiend's approach
// Based on HexFiend's BSD 2-Clause License

import SwiftUI
import AppKit

struct HexFiendVirtualScrollView<Content: View>: NSViewRepresentable {
    let itemCount: Int
    let itemHeight: CGFloat
    @Binding var selectedIndex: Int?
    @ViewBuilder let content: (Int) -> Content
    
    func makeNSView(context: Context) -> VirtualScrollNSView {
        let scrollView = VirtualScrollNSView()
        scrollView.contentBuilder = { index in
            AnyView(content(index))
        }
        scrollView.itemCount = itemCount
        scrollView.itemHeight = itemHeight
        scrollView.selectedIndex = selectedIndex
        return scrollView
    }
    
    func updateNSView(_ scrollView: VirtualScrollNSView, context: Context) {
        scrollView.itemCount = itemCount
        scrollView.selectedIndex = selectedIndex
        scrollView.virtualContentView.updateVisibleItems()
    }
}

class VirtualScrollNSView: NSScrollView {
    var itemCount: Int = 0 {
        didSet { updateContentSize() }
    }
    var itemHeight: CGFloat = 22 {
        didSet { updateContentSize() }
    }
    var selectedIndex: Int?
    var contentBuilder: ((Int) -> AnyView)?
    
    var virtualContentView: VirtualContentNSView!
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupScrollView()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupScrollView()
    }
    
    private func setupScrollView() {
        virtualContentView = VirtualContentNSView()
        virtualContentView.parentScrollView = self
        
        hasVerticalScroller = true
        hasHorizontalScroller = false
        autohidesScrollers = false
        documentView = virtualContentView
        
        backgroundColor = NSColor(red: 0.15, green: 0.15, blue: 0.16, alpha: 1.0)
    }
    
    private func updateContentSize() {
        let totalHeight = CGFloat(itemCount) * itemHeight
        virtualContentView.frame = NSRect(x: 0, y: 0, width: frame.width, height: totalHeight)
        virtualContentView.updateVisibleItems()
        virtualContentView.needsDisplay = true
    }
    
    func visibleRange() -> Range<Int> {
        let visibleRect = documentVisibleRect
        let startOffset = visibleRect.minY
        let endOffset = visibleRect.maxY
        
        let startIndex = max(0, Int(startOffset / itemHeight))
        let endIndex = min(itemCount, Int(endOffset / itemHeight) + 1)
        
        return startIndex..<endIndex
    }
    
    override func reflectScrolledClipView(_ clipView: NSClipView) {
        super.reflectScrolledClipView(clipView)
        virtualContentView.needsDisplay = true
    }
}

class VirtualContentNSView: NSView {
    weak var parentScrollView: VirtualScrollNSView?
    
    override var isFlipped: Bool { true }
    
    private var hostingViews: [Int: NSHostingView<AnyView>] = [:]
    
    override func draw(_ dirtyRect: NSRect) {
        // Clear background
        NSColor(red: 0.15, green: 0.15, blue: 0.16, alpha: 1.0).setFill()
        dirtyRect.fill()
        
        // Update visible items
        updateVisibleItems()
    }
    
    func updateVisibleItems() {
        guard let parent = parentScrollView,
              let contentBuilder = parent.contentBuilder else { return }
        
        let visibleRange = parent.visibleRange()
        
        // Remove hosting views that are no longer visible
        let indicesToRemove = hostingViews.keys.filter { !visibleRange.contains($0) }
        for index in indicesToRemove {
            hostingViews[index]?.removeFromSuperview()
            hostingViews.removeValue(forKey: index)
        }
        
        // Create or update hosting views for visible items
        for index in visibleRange {
            let itemRect = NSRect(
                x: 0,
                y: CGFloat(index) * parent.itemHeight,
                width: frame.width,
                height: parent.itemHeight
            )
            
            if hostingViews[index] == nil {
                let hostingView = NSHostingView(rootView: contentBuilder(index))
                hostingView.frame = itemRect
                hostingViews[index] = hostingView
                addSubview(hostingView)
            } else {
                // Update frame if needed
                hostingViews[index]?.frame = itemRect
            }
        }
    }
    
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil {
            // Force initial update when added to window
            updateVisibleItems()
        }
    }
}