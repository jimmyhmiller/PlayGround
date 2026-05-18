import SwiftUI
import AppKit

struct NotesView: View {
    @AppStorage("sidebar.notes.text") private var text: String = ""
    @State private var rawMode: Bool = false

    var body: some View {
        ZStack {
            VisualEffectBackground(material: .hudWindow, blending: .behindWindow)

            MarkdownEditor(text: $text, rawMode: $rawMode)
                .padding(.horizontal, 12)
                .padding(.vertical, 16)
        }
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(Color.primary.opacity(0.10))
                .frame(width: 1)
        }
        .ignoresSafeArea()
    }
}

private struct VisualEffectBackground: NSViewRepresentable {
    let material: NSVisualEffectView.Material
    let blending: NSVisualEffectView.BlendingMode

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blending
        view.state = .active
        view.isEmphasized = true
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blending
    }
}
