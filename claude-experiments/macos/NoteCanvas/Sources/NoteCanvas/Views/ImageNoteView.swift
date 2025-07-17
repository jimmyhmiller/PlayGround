import AppKit

class ImageNoteView: BaseNoteView, NoteViewProtocol {
    var note: ImageNote {
        didSet {
            updateAppearance()
        }
    }
    
    private var imageLayer: CALayer!
    private var imageView: NSImageView!
    
    init(note: ImageNote) {
        self.note = note
        super.init(frame: CGRect(origin: note.position, size: note.size))
        setupImageView()
        updateAppearance()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupImageView() {
        imageView = NSImageView(frame: bounds.insetBy(dx: 1, dy: 1))
        imageView.autoresizingMask = [.width, .height]
        imageView.imageFrameStyle = .none
        imageView.imageScaling = .scaleProportionallyUpOrDown
        imageView.wantsLayer = true
        
        imageLayer = imageView.layer!
        imageLayer.cornerRadius = note.cornerRadius
        imageLayer.masksToBounds = true
        
        addSubview(imageView)
    }
    
    func updateAppearance() {
        contentLayer.cornerRadius = note.cornerRadius
        imageLayer.cornerRadius = note.cornerRadius
        
        if let image = NSImage(contentsOf: note.imagePath) {
            imageView.image = image
            
            switch note.aspectRatioMode {
            case .fit:
                imageView.imageScaling = .scaleProportionallyUpOrDown
            case .fill:
                imageView.imageScaling = .scaleProportionallyUpOrDown
                adjustImageForFillMode(image)
            case .stretch:
                imageView.imageScaling = .scaleAxesIndependently
            }
        }
    }
    
    private func adjustImageForFillMode(_ image: NSImage) {
        let imageAspect = image.size.width / image.size.height
        let viewAspect = bounds.width / bounds.height
        
        if imageAspect > viewAspect {
            imageView.imageScaling = .scaleProportionallyUpOrDown
        } else {
            imageView.imageScaling = .scaleProportionallyUpOrDown
        }
    }
    
    func updateSelection() {
        animateSelection(note.isSelected)
    }
    
    func startDragging(at point: NSPoint) {
    }
    
    func handleResize(edge: ResizeEdge, delta: CGSize) {
    }
    
    override func getNoteItem() -> (any NoteItem)? {
        return note
    }
}