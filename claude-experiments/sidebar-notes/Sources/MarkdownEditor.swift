import SwiftUI
import AppKit

struct MarkdownEditor: NSViewRepresentable {
    @Binding var text: String
    @Binding var rawMode: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text, rawMode: $rawMode)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        scroll.autohidesScrollers = true
        scroll.scrollerStyle = .overlay

        let textView = MarkdownTextView()
        textView.delegate = context.coordinator
        textView.isRichText = false
        textView.allowsUndo = true
        textView.drawsBackground = false
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isAutomaticLinkDetectionEnabled = false
        textView.isContinuousSpellCheckingEnabled = false
        textView.smartInsertDeleteEnabled = false
        textView.usesFontPanel = false
        textView.usesFindPanel = false
        textView.font = NSFont.systemFont(ofSize: 14)
        textView.textColor = .labelColor
        textView.insertionPointColor = .labelColor
        textView.textContainerInset = NSSize(width: 4, height: 4)

        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude,
                                  height: CGFloat.greatestFiniteMagnitude)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize =
            NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)

        textView.onToggleRawMode = { [weak coordinator = context.coordinator] in
            coordinator?.toggleRawMode()
        }

        textView.string = text
        scroll.documentView = textView

        context.coordinator.textView = textView
        context.coordinator.applyStyling()
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let textView = scroll.documentView as? MarkdownTextView else { return }
        if textView.string != text {
            let selection = textView.selectedRanges
            textView.string = text
            textView.selectedRanges = selection
        }
        if context.coordinator.rawMode != rawMode {
            context.coordinator.rawMode = rawMode
            context.coordinator.applyStyling()
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        @Binding var text: String
        @Binding private var rawModeBinding: Bool
        var rawMode: Bool
        weak var textView: NSTextView?
        private var isApplyingStyling = false

        init(text: Binding<String>, rawMode: Binding<Bool>) {
            self._text = text
            self._rawModeBinding = rawMode
            self.rawMode = rawMode.wrappedValue
        }

        func toggleRawMode() {
            rawModeBinding.toggle()
            rawMode = rawModeBinding
            applyStyling()
        }

        func textDidChange(_ notification: Notification) {
            guard let tv = notification.object as? NSTextView else { return }
            text = tv.string
            applyStyling()
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            guard !isApplyingStyling else { return }
            applyStyling()
        }

        func applyStyling() {
            guard let tv = textView, let storage = tv.textStorage else { return }
            isApplyingStyling = true
            storage.beginEditing()
            MarkdownHighlighter.apply(
                to: storage,
                rawMode: rawMode,
                activeLineRange: currentLineRange()
            )
            storage.endEditing()
            isApplyingStyling = false
        }

        private func currentLineRange() -> NSRange? {
            guard let tv = textView else { return nil }
            let str = tv.string as NSString
            let sel = tv.selectedRange()
            // Clamp so a selection past EOF (after deletes) doesn't blow up lineRange.
            let safe = NSRange(
                location: min(sel.location, str.length),
                length: max(0, min(sel.length, str.length - min(sel.location, str.length)))
            )
            return str.lineRange(for: safe)
        }
    }
}

final class MarkdownTextView: NSTextView {
    var onToggleRawMode: (() -> Void)?

    override func keyDown(with event: NSEvent) {
        if event.modifierFlags.contains(.command),
           event.charactersIgnoringModifiers == "/" {
            onToggleRawMode?()
            return
        }
        super.keyDown(with: event)
    }
}
