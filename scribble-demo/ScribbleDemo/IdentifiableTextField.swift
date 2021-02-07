/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A custom text field used to show a sticker added to the background image.
*/

import UIKit

/*
 StickerTextField is a UITextField subclass that can adjust the look and feel
 of the text to look like a sticker over the back of a laptop. It adjusts its
 own frame as the content changes. It also demonstrates adding buttons to the
 shortcuts bar and the Scribble palette through UITextInputAssistantItem.
 */
class IdentifiableTextField: UITextField {
    
    var fontSize: CGFloat = 28.0
    
    let identifier = UUID()
    
    var writableFrame: CGRect {
        frame.inset(by: UIEdgeInsets(top: -20, left: -20, bottom: -20, right: -70))
    }
    
    required init?(coder: NSCoder) {
        fatalError("Not implemented")
    }
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        text = ""
        borderStyle = .roundedRect
        font = UIFont(name: "Futura-Bold", size: 70)
        layer.borderWidth = 1
        layer.cornerRadius = 3
        layer.borderColor = #colorLiteral(red: 0.6000000238, green: 0.6000000238, blue: 0.6000000238, alpha: 1)
        backgroundColor = #colorLiteral(red: 0.8039215803, green: 0.8039215803, blue: 0.8039215803, alpha: 1)
        textColor = UIColor(red: 0, green: 0, blue: 0, alpha: 1)

    }
    
    convenience init(origin: CGPoint) {
        self.init(frame: CGRect(origin: origin, size: CGSize(width: 12, height: 20)))
    }
    
    
    func updateSize(centerResize: Bool = false) {
        let oldSize = frame.size
        let size = sizeThatFits(CGSize(width: 1024, height: fontSize))
        let oldOrigin = frame.origin
        
        let deltaX = size.width - oldSize.width
        let deltaY = size.height - oldSize.height
        
        // Adjust the size of the field to fit the current content.
        let origin = centerResize ? CGPoint(x: oldOrigin.x - deltaX / 2, y: oldOrigin.y - deltaY / 2) : oldOrigin
        frame = CGRect(origin: origin, size: size)
    }
    

}
