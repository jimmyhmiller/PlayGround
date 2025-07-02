import AppKit

class TextInputDialog {
    static func show(title: String = "Enter Text", message: String = "Enter text for the bubble:", defaultText: String = "Text") -> String? {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        
        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 24))
        textField.stringValue = defaultText
        textField.selectText(nil)
        alert.accessoryView = textField
        
        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            let text = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? defaultText : text
        }
        return nil
    }
}