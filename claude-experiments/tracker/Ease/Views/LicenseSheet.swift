#if !MAS

import SwiftUI

struct LicenseSheet: View {
    @ObservedObject var manager: LicenseManager = .shared
    @Environment(\.dismiss) private var dismiss

    @State private var licenseKey: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Enter your license")
                .font(.title2)
                .fontWeight(.semibold)

            Text("You should have received your license key by email after purchase.")
                .foregroundStyle(.secondary)
                .font(.callout)

            TextField("XXXX-XXXX-XXXX-XXXX-XXXX", text: $licenseKey)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
                .disableAutocorrection(true)

            statusView

            HStack {
                Button("Buy a license") {
                    if let url = URL(string: "https://your-lemon-squeezy-link.example.com") {
                        NSWorkspace.shared.open(url)
                    }
                }
                .buttonStyle(.link)

                Spacer()

                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Button("Activate") {
                    Task {
                        await manager.activate(licenseKey: trimmedKey)
                        if case .activated = manager.state {
                            dismiss()
                        }
                    }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(trimmedKey.isEmpty || manager.state == .checking)
            }
        }
        .padding(24)
        .frame(width: 440)
    }

    private var trimmedKey: String {
        licenseKey.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    @ViewBuilder
    private var statusView: some View {
        switch manager.state {
        case .checking:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Activating…").foregroundStyle(.secondary)
            }
        case .invalid(let reason):
            Text(reason)
                .font(.callout)
                .foregroundStyle(.red)
        case .activated:
            Text("Activated.")
                .font(.callout)
                .foregroundStyle(.green)
        case .unactivated:
            EmptyView()
        }
    }
}

#endif
