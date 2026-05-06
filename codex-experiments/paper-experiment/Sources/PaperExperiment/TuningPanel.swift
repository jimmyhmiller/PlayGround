import AppKit

@MainActor
final class TuningPanel: NSView {
    private static let persistenceKey = "PaperExperiment.RenderTuning.v1"
    private weak var renderer: Renderer?
    private var sliderTargets: [SliderTarget] = []

    init(renderer: Renderer) {
        self.renderer = renderer
        renderer.tuning = Self.loadTuning()
        super.init(frame: NSRect(x: 0, y: 0, width: 280, height: 820))
        build()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func build() {
        wantsLayer = true
        layer?.backgroundColor = NSColor(calibratedWhite: 0.12, alpha: 1).cgColor

        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)

        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 18),
        ])

        let title = NSTextField(labelWithString: "Rendering")
        title.font = .systemFont(ofSize: 18, weight: .semibold)
        title.textColor = .white
        stack.addArrangedSubview(title)

        addSlider(stack, title: "Grain scale", range: 0.1...2.0, value: renderer?.tuning.grainScale ?? 0.75) { [weak renderer] value in
            renderer?.tuning.grainScale = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Grain amount", range: 0.0...0.12, value: renderer?.tuning.grainAmount ?? 0.035) { [weak renderer] value in
            renderer?.tuning.grainAmount = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Edge darkening", range: 0.0...0.22, value: renderer?.tuning.edgeDarkening ?? 0.07) { [weak renderer] value in
            renderer?.tuning.edgeDarkening = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Upper highlight", range: 0.0...0.18, value: renderer?.tuning.upperHighlight ?? 0.08) { [weak renderer] value in
            renderer?.tuning.upperHighlight = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Low brightness", range: 0.55...1.1, value: renderer?.tuning.lowLayerBrightness ?? 0.83) { [weak renderer] value in
            renderer?.tuning.lowLayerBrightness = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "High brightness", range: 0.8...1.3, value: renderer?.tuning.highLayerBrightness ?? 1.06) { [weak renderer] value in
            renderer?.tuning.highLayerBrightness = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "AO strength", range: 0.0...2.2, value: renderer?.tuning.ambientOcclusion ?? 1.0) { [weak renderer] value in
            renderer?.tuning.ambientOcclusion = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "AO radius", range: 4...80, value: renderer?.tuning.aoRadius ?? 28) { [weak renderer] value in
            renderer?.tuning.aoRadius = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Shadow strength", range: 0.0...2.4, value: renderer?.tuning.shadowStrength ?? 1.0) { [weak renderer] value in
            renderer?.tuning.shadowStrength = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Shadow radius", range: 6...100, value: renderer?.tuning.shadowRadius ?? 42) { [weak renderer] value in
            renderer?.tuning.shadowRadius = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Shadow X", range: -0.12...0.12, value: renderer?.tuning.shadowX ?? 0.035) { [weak renderer] value in
            renderer?.tuning.shadowX = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Shadow Y", range: -0.12...0.12, value: renderer?.tuning.shadowY ?? -0.025) { [weak renderer] value in
            renderer?.tuning.shadowY = value
            Self.saveTuning(renderer?.tuning)
        }
        addSlider(stack, title: "Recess boost", range: 0.0...2.5, value: renderer?.tuning.recessBoost ?? 1.0) { [weak renderer] value in
            renderer?.tuning.recessBoost = value
            Self.saveTuning(renderer?.tuning)
        }
    }

    private func addSlider(
        _ stack: NSStackView,
        title: String,
        range: ClosedRange<Float>,
        value: Float,
        onChange: @escaping (Float) -> Void
    ) {
        let row = NSStackView()
        row.orientation = .vertical
        row.spacing = 4
        row.translatesAutoresizingMaskIntoConstraints = false
        row.widthAnchor.constraint(equalToConstant: 248).isActive = true

        let label = NSTextField(labelWithString: "\(title): \(format(value))")
        label.font = .monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        label.textColor = NSColor(calibratedWhite: 0.88, alpha: 1)

        let slider = NSSlider(value: Double(value), minValue: Double(range.lowerBound), maxValue: Double(range.upperBound), target: nil, action: nil)
        slider.isContinuous = true
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.widthAnchor.constraint(equalToConstant: 248).isActive = true

        let target = SliderTarget { slider in
            let newValue = Float(slider.doubleValue)
            label.stringValue = "\(title): \(self.format(newValue))"
            onChange(newValue)
        }
        slider.target = target
        slider.action = #selector(SliderTarget.changed(_:))
        sliderTargets.append(target)

        row.addArrangedSubview(label)
        row.addArrangedSubview(slider)
        stack.addArrangedSubview(row)
    }

    private func format(_ value: Float) -> String {
        String(format: "%.3f", value)
    }

    private static func loadTuning() -> RenderTuning {
        guard let dictionary = UserDefaults.standard.dictionary(forKey: persistenceKey) as? [String: Double] else {
            return RenderTuning()
        }
        var tuning = RenderTuning()
        tuning.grainScale = Float(dictionary["grainScale"] ?? Double(tuning.grainScale))
        tuning.grainAmount = Float(dictionary["grainAmount"] ?? Double(tuning.grainAmount))
        tuning.edgeDarkening = Float(dictionary["edgeDarkening"] ?? Double(tuning.edgeDarkening))
        tuning.upperHighlight = Float(dictionary["upperHighlight"] ?? Double(tuning.upperHighlight))
        tuning.lowLayerBrightness = Float(dictionary["lowLayerBrightness"] ?? Double(tuning.lowLayerBrightness))
        tuning.highLayerBrightness = Float(dictionary["highLayerBrightness"] ?? Double(tuning.highLayerBrightness))
        tuning.upperEdgeMark = Float(dictionary["upperEdgeMark"] ?? Double(tuning.upperEdgeMark))
        tuning.ambientOcclusion = Float(dictionary["ambientOcclusion"] ?? Double(tuning.ambientOcclusion))
        tuning.aoRadius = Float(dictionary["aoRadius"] ?? Double(tuning.aoRadius))
        tuning.shadowStrength = Float(dictionary["shadowStrength"] ?? Double(tuning.shadowStrength))
        tuning.shadowRadius = Float(dictionary["shadowRadius"] ?? Double(tuning.shadowRadius))
        tuning.shadowX = Float(dictionary["shadowX"] ?? Double(tuning.shadowX))
        tuning.shadowY = Float(dictionary["shadowY"] ?? Double(tuning.shadowY))
        tuning.recessBoost = Float(dictionary["recessBoost"] ?? Double(tuning.recessBoost))
        return tuning
    }

    private static func saveTuning(_ tuning: RenderTuning?) {
        guard let tuning else { return }
        UserDefaults.standard.set([
            "grainScale": Double(tuning.grainScale),
            "grainAmount": Double(tuning.grainAmount),
            "edgeDarkening": Double(tuning.edgeDarkening),
            "upperHighlight": Double(tuning.upperHighlight),
            "lowLayerBrightness": Double(tuning.lowLayerBrightness),
            "highLayerBrightness": Double(tuning.highLayerBrightness),
            "upperEdgeMark": Double(tuning.upperEdgeMark),
            "ambientOcclusion": Double(tuning.ambientOcclusion),
            "aoRadius": Double(tuning.aoRadius),
            "shadowStrength": Double(tuning.shadowStrength),
            "shadowRadius": Double(tuning.shadowRadius),
            "shadowX": Double(tuning.shadowX),
            "shadowY": Double(tuning.shadowY),
            "recessBoost": Double(tuning.recessBoost),
        ], forKey: persistenceKey)
    }
}

@MainActor
private final class SliderTarget: NSObject {
    private let onChange: (NSSlider) -> Void

    init(onChange: @escaping (NSSlider) -> Void) {
        self.onChange = onChange
    }

    @objc func changed(_ sender: NSSlider) {
        onChange(sender)
    }
}
