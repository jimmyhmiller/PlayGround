import AppKit

struct VisualEffects {
    static func applyShadow(to layer: CALayer,
                           color: NSColor = .black,
                           opacity: Float = 0.15,
                           offset: CGSize = CGSize(width: 0, height: -2),
                           radius: CGFloat = 8,
                           path: CGPath? = nil) {
        layer.shadowColor = color.cgColor
        layer.shadowOpacity = opacity
        layer.shadowOffset = offset
        layer.shadowRadius = radius
        if let path = path {
            layer.shadowPath = path
        }
    }
    
    static func applyGlow(to layer: CALayer,
                         color: NSColor,
                         radius: CGFloat = 10,
                         opacity: Float = 0.5) {
        layer.shadowColor = color.cgColor
        layer.shadowOpacity = opacity
        layer.shadowOffset = .zero
        layer.shadowRadius = radius
    }
    
    static func createRoundedRectPath(in rect: CGRect, cornerRadius: CGFloat) -> CGPath {
        CGPath(roundedRect: rect, cornerWidth: cornerRadius, cornerHeight: cornerRadius, transform: nil)
    }
    
    static func animateLayer(_ layer: CALayer,
                           keyPath: String,
                           from: Any?,
                           to: Any?,
                           duration: CFTimeInterval = 0.3,
                           timingFunction: CAMediaTimingFunction = .init(name: .easeInEaseOut)) {
        let animation = CABasicAnimation(keyPath: keyPath)
        animation.fromValue = from
        animation.toValue = to
        animation.duration = duration
        animation.timingFunction = timingFunction
        animation.fillMode = .forwards
        animation.isRemovedOnCompletion = false
        
        layer.add(animation, forKey: keyPath)
        layer.setValue(to, forKeyPath: keyPath)
    }
    
    static func springAnimation(keyPath: String,
                              from: Any?,
                              to: Any?,
                              damping: CGFloat = 10,
                              stiffness: CGFloat = 100,
                              mass: CGFloat = 1) -> CASpringAnimation {
        let animation = CASpringAnimation(keyPath: keyPath)
        animation.fromValue = from
        animation.toValue = to
        animation.damping = damping
        animation.stiffness = stiffness
        animation.mass = mass
        animation.duration = animation.settlingDuration
        animation.fillMode = .forwards
        animation.isRemovedOnCompletion = false
        return animation
    }
    
    static func fadeTransition(duration: CFTimeInterval = 0.2) -> CATransition {
        let transition = CATransition()
        transition.type = .fade
        transition.duration = duration
        return transition
    }
}

class BlurView: NSView {
    private var blurView: NSVisualEffectView!
    
    var material: NSVisualEffectView.Material = .hudWindow {
        didSet {
            blurView.material = material
        }
    }
    
    var blendingMode: NSVisualEffectView.BlendingMode = .behindWindow {
        didSet {
            blurView.blendingMode = blendingMode
        }
    }
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupBlur()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupBlur()
    }
    
    private func setupBlur() {
        blurView = NSVisualEffectView(frame: bounds)
        blurView.autoresizingMask = [.width, .height]
        blurView.material = material
        blurView.blendingMode = blendingMode
        blurView.state = .active
        addSubview(blurView)
    }
}

extension CALayer {
    func addSublayerWithConstraints(_ sublayer: CALayer, insets: NSEdgeInsets = .init()) {
        sublayer.frame = bounds.inset(by: insets)
        sublayer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        addSublayer(sublayer)
    }
}

extension CGRect {
    func inset(by insets: NSEdgeInsets) -> CGRect {
        return CGRect(
            x: origin.x + insets.left,
            y: origin.y + insets.bottom,
            width: width - insets.left - insets.right,
            height: height - insets.top - insets.bottom
        )
    }
}