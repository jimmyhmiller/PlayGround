/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The main view controller for the Scribble Demo.
*/

import UIKit

/*
 This view controller demonstrates how to customize the behavior of Scribble
 and how to enable writing in a view that is not normally a text input.
 Specifically, it installs:
 
 * A UIIndirectScribbleInteraction to enable writing with Scribble on the
 background image to create new texts.
 * A UIScribbleInteraction to disable writing on a specific area on the image
 where the logo is.
 
 It also installs an EngravingFakeField view, which allows adding engraved text
 to the back of the laptop. The class implementing this view contains another
 example of using UIIndirectScribbleInteraction, to enable writing on a
 text-field-lookalike without requiring to tap on it first.
 */
class ViewController: UIViewController, UIIndirectScribbleInteractionDelegate, UIScribbleInteractionDelegate, UITextFieldDelegate {
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        setNeedsStatusBarAppearanceUpdate()
    }
    override var preferredStatusBarStyle: UIStatusBarStyle {
        .darkContent
    }

    var textTextFields: [IdentifiableTextField] = []
    
    var textContainerView = UIView()

    // Used to identify the Scribble Element representing the background view.
    let rootViewElementID = UUID()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        textContainerView.backgroundColor = #colorLiteral(red: 0.8039215803, green: 0.8039215803, blue: 0.8039215803, alpha: 1)
        // The text container view provides the writing area to add new
        // texts over the background, and has the Scribble interactions.
        textContainerView.frame = view.bounds
        textContainerView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(textContainerView)
        
        // Install a UIScribbleInteraction, which we'll use to disable Scribble
        // when we want to let the Pencil draw instead of write.
        let scribbleInteraction = UIScribbleInteraction(delegate: self)
        textContainerView.addInteraction(scribbleInteraction)

        // Install a UIIndirectScribbleInteraction, which will provide the
        // "elements" that represent virtual writing areas.
        let indirectScribbleInteraction = UIIndirectScribbleInteraction(delegate: self)
        textContainerView.addInteraction(indirectScribbleInteraction)

        
        // Background tap recognizer.
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTapGesture))
        view.addGestureRecognizer(tapGesture)
    }
    
    // MARK: - UIScribbleInteractionDelegate
    
    func scribbleInteraction(_ interaction: UIScribbleInteraction, shouldBeginAt location: CGPoint) -> Bool {

        return true
    }
    
    // MARK: - UIIndirectScribbleInteractionDelegate
    
    func indirectScribbleInteraction(_ interaction: UIInteraction, shouldDelayFocusForElement elementIdentifier: UUID) -> Bool {
        // When writing on a blank area, wait until the user stops writing
        // before triggering element focus, to avoid writing distractions.
        return elementIdentifier == rootViewElementID
    }
    
    func indirectScribbleInteraction(_ interaction: UIInteraction, requestElementsIn rect: CGRect,
                                     completion: @escaping ([ElementIdentifier]) -> Void) {

        var availableElementIDs: [UUID] = []

        // Include the identifier of the root view. It must be at the start of
        // the array, so it doesn't cover all the other fields.
        availableElementIDs.append(rootViewElementID)
        
        // Include the text fields that intersect the requested rect.
        // Even though these are real text fields, Scribble can't find them
        // because it doesn't traverse subviews of a view that has a
        // UIIndirectScribbleInteraction.
        for textField in textTextFields {
            if textField.writableFrame.intersects(rect) {
                availableElementIDs.append(textField.identifier)
            }
        }

        // Call the completion handler with the array of element identifiers.
        completion(availableElementIDs)
    }
    
    func indirectScribbleInteraction(_ interaction: UIInteraction, isElementFocused elementIdentifier: UUID) -> Bool {
        if elementIdentifier == rootViewElementID {
            // The root element represents the background view, so it never
            // becomes focused itself.
            return false
        } else {
            // For text elements, indicate if the corresponding text field
            // is first responder.
            return textFieldForIdentifier(elementIdentifier)?.isFirstResponder ?? false
        }
    }
    
    func indirectScribbleInteraction(_ interaction: UIInteraction, frameForElement elementIdentifier: UUID) -> CGRect {
        var elementRect = CGRect.null
        
        if let textField = textFieldForIdentifier(elementIdentifier) {
            // Scribble is asking about the frame for one of the text frames.
            // Return a frame larger than the field itself to make it easier to
            // append text without creating another field.
            elementRect = textField.writableFrame
        } else if elementIdentifier == rootViewElementID {
            // Scribble is asking about the background writing area. Return the
            // frame for the whole view.
            elementRect = textContainerView.frame
        }
        
        return elementRect
    }

    func indirectScribbleInteraction(_ interaction: UIInteraction, focusElementIfNeeded elementIdentifier: UUID,
                                     referencePoint focusReferencePoint: CGPoint, completion: @escaping ((UIResponder & UITextInput)?) -> Void) {

        // Get an existing field at this location, or create a new one if
        // writing in the background.
        let textField: IdentifiableTextField?

        if elementIdentifier == rootViewElementID {
            textField = addStickerFieldAtLocation(focusReferencePoint)
        } else {
            textField = textFieldForIdentifier(elementIdentifier)
        }
        let tapGesture = UIPanGestureRecognizer(target: self, action: #selector(panPiece))
        textField?.addGestureRecognizer(tapGesture)

        // Focus the field. It should have no effect if it was focused already.
        textField?.becomeFirstResponder()
        
        // Call the completion handler as expected by the caller.
        // It could be called asynchronously if, for example, there was an
        // animation to insert a new text field.
        completion(textField)
    }
    
    var initialCenter = CGPoint()  // The initial center point of the view.
    
    @IBAction func panPiece(_ gestureRecognizer : UIPanGestureRecognizer) {
       guard gestureRecognizer.view != nil else {return}
       let piece = gestureRecognizer.view!
       // Get the changes in the X and Y directions relative to
       // the superview's coordinate space.
       let translation = gestureRecognizer.translation(in: piece.superview)
       if gestureRecognizer.state == .began {
          // Save the view's original position.
          self.initialCenter = piece.center
       }
        if gestureRecognizer.state == .ended {
            print(gestureRecognizer.velocity(in: textContainerView).x)
            if gestureRecognizer.velocity(in: textContainerView).x >= 4000 {
                remove(textField: piece as! IdentifiableTextField)
            }

        }
          // Update the position for the .began, .changed, and .ended states
       if gestureRecognizer.state != .cancelled {
          // Add the X and Y translation to the view's original position.
          let newCenter = CGPoint(x: initialCenter.x + translation.x, y: initialCenter.y + translation.y)
          piece.center = newCenter
       }
       else {
          // On cancellation, return the piece to its original location.
          piece.center = initialCenter
       }
    }
        
    
    // MARK: - Text Field Event Handling
            
    @objc
    func handleTextFieldDidChange(_ textField: UITextField) {

        guard let textField = textField as? IdentifiableTextField else {
            return
        }
 
        // When erasing the entire text of a text, remove the corresponding
        // text field.
        if !removeIfEmpty(textField) {
            // The size updates to accommodate the current content.
            textField.updateSize()
        }
    }
    
    func textFieldDidEndEditing(_ textField: UITextField) {
        guard let textField = textField as? IdentifiableTextField else {
            return
        }
        
        removeIfEmpty(textField)
    }
    
    // MARK: - Gesture Handling
    
    @objc
    func handleTapGesture() {
        // Unfocus our text fields.
        for textField in textTextFields where textField.isFirstResponder {
            textField.resignFirstResponder()
            break
        }

    }
    
    // MARK: - Sticker Text Field Handling
    
    func textFieldForIdentifier(_ identifier: UUID) -> IdentifiableTextField? {
        for textField in textTextFields where textField.identifier == identifier {
            return textField
        }
        return nil
    }
    
    func addStickerFieldAtLocation(_ location: CGPoint) -> IdentifiableTextField {

        let textField = IdentifiableTextField(origin: location)
        textField.delegate = self
        textField.addTarget(self, action: #selector(handleTextFieldDidChange(_:)), for: .editingChanged)
        textTextFields.append(textField)

        textContainerView.addSubview(textField)

        return textField
    }

    func remove(textField: IdentifiableTextField) {
        if let index = textTextFields.firstIndex(of: textField) {
            textTextFields.remove(at: index)
        }
        textField.resignFirstResponder()
        textField.removeFromSuperview()
    }
    
    @discardableResult
    func removeIfEmpty(_ textField: IdentifiableTextField) -> Bool {
        let textLength = textField.text?.count ?? 0
        if textLength == 0 {
            remove(textField: textField)
            return true
        }
        return false
    }
    
}

