import AppKit
import CoreGraphics
import simd

enum FieldFocus: CaseIterable {
    case fullName
    case email
    case password
}

enum HitTarget {
    case none
    case field(FieldFocus)
    case checkbox
    case button
}

struct RoundedRectCutout {
    var rect: CGRect
    var radius: Float
}

struct SurfaceSpec {
    var rect: CGRect
    var radius: Float
    var z: Float
    var color: SIMD4<Float>
    var cutouts: [RoundedRectCutout] = []
    var shadowSources: [RoundedRectCutout] = []
    var isOuterRect: Bool = false
    var isPressed: Bool = false
    var shadowDepth: Float = 0
    var inkMask: Float = 0
}

struct InkStamp {
    var id: String
    var text: String
    var rect: CGRect
    var font: NSFont
    var color: NSColor
    var alignment: CTTextAlignment

    func with(id: String? = nil, rect: CGRect? = nil, color: NSColor? = nil) -> InkStamp {
        InkStamp(
            id: id ?? self.id,
            text: text,
            rect: rect ?? self.rect,
            font: font,
            color: color ?? self.color,
            alignment: alignment
        )
    }
}

struct PaperPalette {
    static let ivory = SIMD4<Float>(0.95, 0.92, 0.86, 1.0)
    static let coral = SIMD4<Float>(0.86, 0.49, 0.45, 1.0)
    static let orange = SIMD4<Float>(0.90, 0.60, 0.33, 1.0)
    static let ochre = SIMD4<Float>(0.85, 0.72, 0.33, 1.0)
    static let sage = SIMD4<Float>(0.58, 0.70, 0.54, 1.0)
    static let teal = SIMD4<Float>(0.39, 0.62, 0.60, 1.0)
    static let blue = SIMD4<Float>(0.41, 0.54, 0.70, 1.0)
    static let violet = SIMD4<Float>(0.61, 0.53, 0.74, 1.0)
    static let charcoal = SIMD4<Float>(0.18, 0.19, 0.22, 1.0)
    static let ink = NSColor(calibratedRed: 0.12, green: 0.14, blue: 0.17, alpha: 0.92)
    static let softInk = NSColor(calibratedRed: 0.19, green: 0.22, blue: 0.25, alpha: 0.76)
}

struct SceneLayout {
    var viewport: CGSize
    var outerRect: CGRect
    var formRect: CGRect
    var titleRect: CGRect
    var subtitleRect: CGRect
    var nameField: CGRect
    var emailField: CGRect
    var passwordField: CGRect
    var rememberBox: CGRect
    var rememberLabel: CGRect
    var forgotLabel: CGRect
    var buttonRect: CGRect
}

final class SceneState {
    var fullName = ""
    var email = ""
    var password = ""
    var rememberMe = true
    var focus: FieldFocus = .fullName
    var buttonPressed = false
    var hoverTarget: HitTarget = .none
    var submitPulse: Float = 0

    func value(for field: FieldFocus) -> String {
        switch field {
        case .fullName: return fullName
        case .email: return email
        case .password: return String(repeating: "•", count: password.count)
        }
    }

    func editValue(for field: FieldFocus, transform: (String) -> String) {
        switch field {
        case .fullName: fullName = transform(fullName)
        case .email: email = transform(email)
        case .password: password = transform(password)
        }
    }
}

enum PaperScene {
    static let layerStep: Float = 18

    static func makeLayout(viewport: CGSize) -> SceneLayout {
        let width = viewport.width
        let height = viewport.height
        let blueOpening = CGRect(origin: .zero, size: viewport).insetBy(dx: 150, dy: 150)
        let formWidth = min(690, blueOpening.width - 72)
        let formHeight = min(560, blueOpening.height - 44)
        let formRect = CGRect(
            x: (width - formWidth) * 0.5,
            y: (height - formHeight) * 0.5,
            width: formWidth,
            height: formHeight
        )

        let insetX = formRect.minX + 64
        let fieldWidth = formRect.width - 128
        let fieldHeight: CGFloat = 60
        let fieldSpacing: CGFloat = 18
        let titleY = formRect.maxY - 92
        let subtitleY = titleY - 36
        let buttonY = formRect.minY + 42
        let rememberBoxY = buttonY + 84
        let passwordFieldY = rememberBoxY + 48
        let emailFieldY = passwordFieldY + fieldHeight + fieldSpacing
        let firstFieldY = emailFieldY + fieldHeight + fieldSpacing

        return SceneLayout(
            viewport: viewport,
            outerRect: CGRect(origin: .zero, size: viewport),
            formRect: formRect,
            titleRect: CGRect(x: formRect.minX + 190, y: titleY, width: formRect.width - 238, height: 46),
            subtitleRect: CGRect(x: formRect.minX + 190, y: subtitleY, width: formRect.width - 238, height: 28),
            nameField: CGRect(x: insetX, y: firstFieldY, width: fieldWidth, height: fieldHeight),
            emailField: CGRect(x: insetX, y: emailFieldY, width: fieldWidth, height: fieldHeight),
            passwordField: CGRect(x: insetX, y: passwordFieldY, width: fieldWidth, height: fieldHeight),
            rememberBox: CGRect(x: insetX, y: rememberBoxY, width: 22, height: 22),
            rememberLabel: CGRect(x: insetX + 34, y: rememberBoxY - 4, width: 130, height: 24),
            forgotLabel: CGRect(x: formRect.maxX - 168, y: rememberBoxY - 4, width: 114, height: 24),
            buttonRect: CGRect(x: insetX, y: buttonY, width: fieldWidth, height: 60)
        )
    }

    static func buildSurfaces(layout: SceneLayout, state: SceneState) -> [SurfaceSpec] {
        let form = layout.formRect
        let fieldRects = [layout.nameField, layout.emailField, layout.passwordField]
        let deepPurple = SIMD4<Float>(0.34, 0.19, 0.47, 1.0)
        let midPurple = SIMD4<Float>(0.48, 0.30, 0.64, 1.0)
        let topPurple = SIMD4<Float>(0.64, 0.48, 0.80, 1.0)

        var surfaces: [SurfaceSpec] = []
        let rainbowCuts: [(SIMD4<Float>, Float, CGFloat?, Float)] = [
            (PaperPalette.blue, 1, nil, 0.25),
            (PaperPalette.teal, 2, 150, 0.65),
            (PaperPalette.sage, 3, 120, 0.82),
            (PaperPalette.ochre, 4, 90, 1.0),
            (PaperPalette.orange, 5, 60, 1.18),
            (PaperPalette.coral, 6, 30, 1.35),
        ]

        for index in rainbowCuts.indices {
            let (color, level, inset, shadowDepth) = rainbowCuts[index]
            let cutouts = inset.map {
                [RoundedRectCutout(rect: layout.outerRect.insetBy(dx: $0, dy: $0), radius: 42)]
            } ?? []
            let shadowSources = index + 1 < rainbowCuts.count
                ? rainbowCuts[index + 1].2.map { [RoundedRectCutout(rect: layout.outerRect.insetBy(dx: $0, dy: $0), radius: 42)] } ?? []
                : []
            surfaces.append(
                SurfaceSpec(
                    rect: layout.outerRect,
                    radius: 0,
                    z: level * layerStep,
                    color: color,
                    cutouts: cutouts,
                    shadowSources: shadowSources,
                    isOuterRect: true,
                    shadowDepth: shadowDepth
                )
            )
        }

        let cardZ2 = Float(2) * layerStep
        let cardZ3 = Float(3) * layerStep
        let cardZ4 = Float(4) * layerStep
        let fieldCutouts = fieldRects.map { RoundedRectCutout(rect: $0.insetBy(dx: -1, dy: -1), radius: 18) }
        let buttonMoat = RoundedRectCutout(rect: layout.buttonRect.insetBy(dx: -1, dy: -1), radius: 20)
        let checkboxCut = RoundedRectCutout(rect: layout.rememberBox.insetBy(dx: -3, dy: -3), radius: 6)
        let avatarOuter = CGRect(x: form.minX + 54, y: form.maxY - 132, width: 104, height: 104)
        let avatarInner = CGRect(x: avatarOuter.midX - 17, y: avatarOuter.midY + 6, width: 34, height: 34)
        let avatarShoulders = CGRect(x: avatarOuter.midX - 31, y: avatarOuter.minY + 22, width: 62, height: 32)
        let avatarCut = RoundedRectCutout(rect: avatarOuter, radius: 52)
        let leftFoot = CGRect(x: form.minX - 22, y: form.minY + 16, width: 56, height: 58)
        let rightFoot = CGRect(x: form.maxX - 34, y: form.minY + 16, width: 56, height: 58)
        let leftShoulder = CGRect(x: form.minX - 20, y: form.maxY - 112, width: 54, height: 72)
        let rightShoulder = CGRect(x: form.maxX - 34, y: form.maxY - 112, width: 54, height: 72)

        let topCutouts = fieldCutouts + [buttonMoat, checkboxCut, avatarCut]
        surfaces.append(
            SurfaceSpec(
                rect: form.offsetBy(dx: 0, dy: -7),
                radius: 34,
                z: cardZ2,
                color: deepPurple,
                shadowDepth: 0.18
            )
        )

        for tab in [leftFoot, rightFoot, leftShoulder, rightShoulder] {
            surfaces.append(
                SurfaceSpec(
                    rect: tab.offsetBy(dx: 0, dy: -7),
                    radius: 26,
                    z: cardZ2,
                    color: deepPurple,
                    shadowDepth: 0.12
                )
            )
        }

        surfaces.append(
            SurfaceSpec(
                rect: form,
                radius: 34,
                z: cardZ4,
                color: topPurple,
                cutouts: topCutouts,
                shadowDepth: 0.18
            )
        )

        for tab in [leftFoot, rightFoot, leftShoulder, rightShoulder] {
            surfaces.append(
                SurfaceSpec(
                    rect: tab,
                    radius: 26,
                    z: cardZ4,
                    color: topPurple,
                    shadowDepth: 0.12
                )
            )
        }

        let recessZ = cardZ3
        for (idx, rect) in fieldRects.enumerated() {
            let active = state.focus == FieldFocus.allCases[idx]
            let wallRect = rect.insetBy(dx: -2, dy: -2)
            let floorRect = rect.insetBy(dx: 5, dy: 5)
            surfaces.append(
                SurfaceSpec(
                    rect: wallRect,
                    radius: 19,
                    z: cardZ4 - 1,
                    color: midPurple,
                    cutouts: [RoundedRectCutout(rect: floorRect, radius: 15)],
                    shadowDepth: 0.10
                )
            )
            surfaces.append(
                SurfaceSpec(
                    rect: floorRect,
                    radius: 15,
                    z: recessZ,
                    color: active ? SIMD4<Float>(0.42, 0.25, 0.56, 1.0) : SIMD4<Float>(0.31, 0.17, 0.43, 1.0),
                    shadowSources: [RoundedRectCutout(rect: wallRect, radius: 19)],
                    shadowDepth: active ? 0.30 : 0.26
                )
            )
        }

        surfaces.append(
            SurfaceSpec(
                rect: avatarCut.rect,
                radius: avatarCut.radius,
                z: cardZ4 - 1,
                color: midPurple,
                cutouts: [RoundedRectCutout(rect: avatarOuter.insetBy(dx: 6, dy: 6), radius: 42)],
                shadowDepth: 0.08
            )
        )

        surfaces.append(
            SurfaceSpec(
                rect: avatarOuter.insetBy(dx: 6, dy: 6),
                radius: 42,
                z: recessZ,
                color: midPurple,
                cutouts: [
                    RoundedRectCutout(rect: avatarInner, radius: 24),
                    RoundedRectCutout(rect: avatarShoulders, radius: 22),
                ],
                shadowSources: [avatarCut],
                shadowDepth: 0.42
            )
        )

        surfaces.append(
            SurfaceSpec(
                rect: checkboxCut.rect,
                radius: checkboxCut.radius,
                z: cardZ4 - 1,
                color: midPurple,
                cutouts: [RoundedRectCutout(rect: layout.rememberBox.insetBy(dx: 2, dy: 2), radius: 4)],
                shadowDepth: 0.06
            )
        )

        surfaces.append(
            SurfaceSpec(
                rect: layout.rememberBox.insetBy(dx: 2, dy: 2),
                radius: 4,
                z: recessZ,
                color: deepPurple,
                shadowSources: [checkboxCut],
                shadowDepth: 0.20
            )
        )

        surfaces.append(
            SurfaceSpec(
                rect: buttonMoat.rect,
                radius: buttonMoat.radius,
                z: cardZ4 - 1,
                color: midPurple,
                cutouts: [RoundedRectCutout(rect: layout.buttonRect.insetBy(dx: 6, dy: 6), radius: 15)],
                shadowDepth: 0.08
            )
        )

        surfaces.append(
            SurfaceSpec(
                rect: layout.buttonRect.insetBy(dx: 6, dy: 6),
                radius: 15,
                z: recessZ,
                color: SIMD4<Float>(0.31, 0.17, 0.43, 1.0),
                shadowSources: [buttonMoat],
                shadowDepth: 0.25
            )
        )

        if state.rememberMe {
            surfaces.append(
                SurfaceSpec(
                    rect: layout.rememberBox.insetBy(dx: 5, dy: 5),
                    radius: 3,
                    z: recessZ + 2,
                    color: topPurple,
                    shadowDepth: 0.25
                )
            )
        }

        return surfaces.sorted { lhs, rhs in
            if lhs.z == rhs.z {
                return lhs.rect.width * lhs.rect.height < rhs.rect.width * rhs.rect.height
            }
            return lhs.z < rhs.z
        }
    }

    static func buildInk(layout: SceneLayout, state: SceneState) -> [InkStamp] {
        let titleFont = NSFont.systemFont(ofSize: 30, weight: .bold)
        let subtitleFont = NSFont.systemFont(ofSize: 18, weight: .regular)
        let labelFont = NSFont.systemFont(ofSize: 16, weight: .medium)
        let valueFont = NSFont.systemFont(ofSize: 22, weight: .medium)
        let smallFont = NSFont.systemFont(ofSize: 17, weight: .medium)
        let buttonFont = NSFont.systemFont(ofSize: 24, weight: .semibold)

        let placeholders: [(String, CGRect, FieldFocus)] = [
            ("Full Name", layout.nameField, .fullName),
            ("Email Address", layout.emailField, .email),
            ("Password", layout.passwordField, .password),
        ]

        var stamps: [InkStamp] = [
            InkStamp(id: "title", text: "Create Account", rect: layout.titleRect, font: titleFont, color: PaperPalette.ink, alignment: .left),
            InkStamp(id: "subtitle", text: "Let's get you started", rect: layout.subtitleRect, font: subtitleFont, color: PaperPalette.softInk, alignment: .left),
            InkStamp(id: "remember", text: "Remember me", rect: layout.rememberLabel, font: smallFont, color: PaperPalette.ink, alignment: .left),
            InkStamp(id: "forgot", text: "Forgot password?", rect: layout.forgotLabel, font: smallFont, color: PaperPalette.softInk, alignment: .right),
            InkStamp(id: "signup", text: "Sign Up", rect: layout.buttonRect.insetBy(dx: 0, dy: 15), font: buttonFont, color: PaperPalette.ink, alignment: .center),
        ]

        for (index, entry) in placeholders.enumerated() {
            let labelRect = CGRect(x: entry.1.minX + 30, y: entry.1.minY + 10, width: entry.1.width - 60, height: 22)
            let valueRect = CGRect(x: entry.1.minX + 30, y: entry.1.minY + 34, width: entry.1.width - 60, height: 28)
            stamps.append(
                InkStamp(id: "placeholder_\(index)", text: entry.0, rect: labelRect, font: labelFont, color: PaperPalette.softInk, alignment: .left)
            )
            let value = state.value(for: entry.2)
            if !value.isEmpty {
                stamps.append(
                    InkStamp(id: "value_\(index)", text: value, rect: valueRect, font: valueFont, color: PaperPalette.ink, alignment: .left)
                )
            }
        }

        if state.rememberMe {
            stamps.append(
                InkStamp(
                    id: "check",
                    text: "✓",
                    rect: layout.rememberBox.insetBy(dx: 1, dy: -1),
                    font: NSFont.systemFont(ofSize: 16, weight: .black),
                    color: PaperPalette.ink,
                    alignment: .center
                )
            )
        }

        return stamps
    }

    static func hitTest(_ point: CGPoint, layout: SceneLayout) -> HitTarget {
        if layout.nameField.contains(point) { return .field(.fullName) }
        if layout.emailField.contains(point) { return .field(.email) }
        if layout.passwordField.contains(point) { return .field(.password) }
        if layout.rememberBox.insetBy(dx: -8, dy: -8).contains(point) { return .checkbox }
        if layout.buttonRect.insetBy(dx: -8, dy: -8).contains(point) { return .button }
        return .none
    }
}
