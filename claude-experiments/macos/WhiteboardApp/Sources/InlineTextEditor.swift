import AppKit

class InlineTextEditor: NSTextField {
    var onTextChanged: ((String) -> Void)?
    var onEditingFinished: (() -> Void)?
    var originalBounds: CGRect = .zero
    var bubbleColor: NSColor = .blue
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupEditor()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupEditor()
    }
    
    private func setupEditor() {
        isBordered = false
        backgroundColor = NSColor.clear
        textColor = NSColor.black
        font = NSFont.systemFont(ofSize: 16.0)
        alignment = .left
        focusRingType = .none  // Remove the light blue focus ring
        
        // Ensure this editor appears above other content
        wantsLayer = true
        
        // Important: Use baseline alignment to match our custom text drawing
        usesSingleLineMode = true
        cell?.sendsActionOnEndEditing = true
        cell?.wraps = false  // Prevent text wrapping
        cell?.isScrollable = true  // Allow text to scroll horizontally
        
        // Center text vertically
        if let cell = cell as? NSTextFieldCell {
            cell.controlSize = .regular
            cell.font = NSFont.systemFont(ofSize: 16.0)
        }
        
        // Set delegate to self to handle text changes
        delegate = self
        
        // Automatically select all text when editing starts
        selectText(nil)
    }
    
    func startEditing(with text: String, in bounds: CGRect, color: NSColor) {
        stringValue = text
        textColor = NSColor.black  // Always use black text for readability
        bubbleColor = color
        originalBounds = bounds
        frame = bounds
        
        // Keep editor completely invisible - no background, no border
        backgroundColor = NSColor.clear
        layer?.borderWidth = 0
        layer?.cornerRadius = 0
        
        // Make sure the field is focused and text is selected
        window?.makeFirstResponder(self)
        selectText(nil)
    }
}

extension InlineTextEditor: NSTextFieldDelegate {
    func controlTextDidChange(_ obj: Notification) {
        // Resize the editor to fit the new text
        resizeToFitText()
        onTextChanged?(stringValue)
    }
    
    private func resizeToFitText() {
        guard let font = font else { return }
        
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        let textSize = (stringValue as NSString).size(withAttributes: attributes)
        
        // Add some padding for expansion, but keep it reasonable
        let padding: CGFloat = 20
        let newWidth = max(textSize.width + padding, 60) // Minimum 60px
        let newHeight = max(textSize.height + 8, 24) // Minimum 24px
        
        // Keep the left edge fixed from where the editor started
        let originalTextStartX = originalBounds.origin.x
        
        let newFrame = CGRect(
            x: originalTextStartX,
            y: originalBounds.midY - newHeight / 2,
            width: newWidth,
            height: newHeight
        )
        
        frame = newFrame
    }
    
    func controlTextDidEndEditing(_ obj: Notification) {
        // Reset appearance when editing finishes
        backgroundColor = NSColor.clear
        layer?.borderWidth = 0
        
        onEditingFinished?()
    }
    
    // Handle Return key to finish editing
    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            // Return key pressed - finish editing
            window?.makeFirstResponder(superview)
            return true
        } else if commandSelector == #selector(NSResponder.cancelOperation(_:)) {
            // Escape key pressed - cancel editing
            window?.makeFirstResponder(superview)
            return true
        }
        return false
    }
}