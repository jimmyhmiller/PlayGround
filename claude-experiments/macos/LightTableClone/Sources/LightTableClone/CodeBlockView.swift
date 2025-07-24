import Cocoa

enum CodeLanguage {
    case python
    case html
    case javascript
}

class CodeBlockView: NSView {
    private let textView: NSTextView
    let code: String
    private let language: CodeLanguage
    
    init(code: String, language: CodeLanguage) {
        self.code = code
        self.language = language
        
        textView = NSTextView()
        
        super.init(frame: .zero)
        
        setupView()
        setupTextView()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupView() {
        wantsLayer = true
        layer?.backgroundColor = NSColor(red: 0.15, green: 0.15, blue: 0.2, alpha: 0.95).cgColor
        layer?.cornerRadius = 8
        layer?.borderWidth = 1
        layer?.borderColor = NSColor(red: 0.3, green: 0.3, blue: 0.4, alpha: 1.0).cgColor
        
        textView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(textView)
        
        NSLayoutConstraint.activate([
            textView.topAnchor.constraint(equalTo: topAnchor, constant: 8),
            textView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            textView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            textView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -8)
        ])
    }
    
    private func setupTextView() {
        // Set up text view to auto-size with content
        textView.string = code
        textView.backgroundColor = .clear
        textView.isEditable = true
        textView.isSelectable = true
        textView.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.textColor = .white
        textView.isRichText = true
        textView.usesFontPanel = false
        textView.isVerticallyResizable = false
        textView.isHorizontallyResizable = false
        
        // Apply syntax highlighting
        applyCode()
        
        // Set up delegate to reapply syntax highlighting when text changes
        textView.delegate = self
    }
    
    private func applyCode() {
        let currentText = textView.string.isEmpty ? code : textView.string
        
        // Save current selection/cursor position
        let selectedRange = textView.selectedRange()
        
        let attributedString = NSMutableAttributedString(string: currentText)
        
        let baseAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 13, weight: .regular),
            .foregroundColor: NSColor.white
        ]
        
        attributedString.addAttributes(baseAttributes, range: NSRange(location: 0, length: currentText.count))
        
        applySyntaxHighlighting(to: attributedString)
        
        textView.textStorage?.setAttributedString(attributedString)
        
        // Restore cursor position
        if selectedRange.location <= currentText.count {
            textView.setSelectedRange(selectedRange)
        }
    }
    
    
    private func applySyntaxHighlighting(to attributedString: NSMutableAttributedString) {
        let text = attributedString.string
        
        let keywordColor = NSColor(red: 0.8, green: 0.4, blue: 0.8, alpha: 1.0)
        let stringColor = NSColor(red: 0.4, green: 0.8, blue: 0.4, alpha: 1.0)
        let commentColor = NSColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)
        let functionColor = NSColor(red: 0.4, green: 0.8, blue: 0.9, alpha: 1.0)
        
        switch language {
        case .python:
            let keywords = ["def", "return", "import", "from", "global", "for", "in", "if"]
            for keyword in keywords {
                highlightPattern("\\b\(keyword)\\b", in: text, with: keywordColor, in: attributedString)
            }
            
            highlightPattern("\"\"\".*?\"\"\"", in: text, with: stringColor, in: attributedString)
            highlightPattern("\".*?\"", in: text, with: stringColor, in: attributedString)
            highlightPattern("'.*?'", in: text, with: stringColor, in: attributedString)
            
            highlightPattern("def\\s+(\\w+)", in: text, with: functionColor, in: attributedString, group: 1)
            
        case .html:
            highlightPattern("<[^>]+>", in: text, with: keywordColor, in: attributedString)
            highlightPattern("\\{%.*?%\\}", in: text, with: functionColor, in: attributedString)
            highlightPattern("\\{\\{.*?\\}\\}", in: text, with: functionColor, in: attributedString)
            
        case .javascript:
            let keywords = ["function", "var", "let", "const", "return", "if", "else", "for"]
            for keyword in keywords {
                highlightPattern("\\b\(keyword)\\b", in: text, with: keywordColor, in: attributedString)
            }
        }
    }
    
    private func highlightPattern(_ pattern: String, in text: String, with color: NSColor, in attributedString: NSMutableAttributedString, group: Int = 0) {
        do {
            let regex = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators])
            let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
            
            for match in matches {
                let range = group < match.numberOfRanges ? match.range(at: group) : match.range
                if range.location != NSNotFound {
                    attributedString.addAttribute(.foregroundColor, value: color, range: range)
                }
            }
        } catch {
            print("Regex error: \(error)")
        }
    }
}

extension CodeBlockView: NSTextViewDelegate {
    func textDidChange(_ notification: Notification) {
        // Delay syntax highlighting to avoid cursor jumping
        DispatchQueue.main.async {
            self.applyCode()
        }
    }
}