import SwiftUI

struct MonitorView: View {
    @ObservedObject var monitor: AudioMonitor

    var body: some View {
        VStack(spacing: 12) {
            header
            Divider()

            if monitor.permissionDenied {
                permissionView
            } else if monitor.isMonitoring {
                monitoringView
            } else {
                idleView
            }

            Divider()
            thresholdControl
            Divider()
            bottomControls
        }
        .padding()
        .frame(width: 280)
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Image(systemName: "waveform")
                .font(.title3)
            Text("Coffee Shop Monitor")
                .font(.headline)
            Spacer()
            statusBadge
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        if !monitor.isMonitoring {
            badge("Idle", color: .secondary)
        } else if monitor.isAboveThreshold {
            badge("LOUD", color: .red)
        } else if monitor.isNearThreshold {
            badge("Warm", color: .orange)
        } else {
            badge("OK", color: .green)
        }
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption.bold())
            .foregroundColor(.white)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color)
            .clipShape(Capsule())
    }

    // MARK: - Monitoring View

    private var monitoringView: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Level meter
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Input Level")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(Int(monitor.currentLevel)) dBFS")
                        .font(.caption.monospacedDigit())
                        .foregroundColor(.secondary)
                }

                LevelMeter(
                    level: monitor.currentLevel,
                    threshold: Float(monitor.thresholdDB)
                )
                .frame(height: 16)
                .animation(.linear(duration: 0.08), value: monitor.currentLevel)
            }

            // Status message
            statusMessage
        }
    }

    @ViewBuilder
    private var statusMessage: some View {
        if monitor.isAboveThreshold {
            Label("Lower your voice!", systemImage: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
                .font(.body.bold())
        } else if monitor.isNearThreshold {
            Label("Getting a bit loud...", systemImage: "speaker.wave.2.fill")
                .foregroundColor(.orange)
        } else {
            Label("You're at a good level", systemImage: "checkmark.circle.fill")
                .foregroundColor(.green)
        }
    }

    // MARK: - Idle / Permission

    private var idleView: some View {
        HStack(spacing: 8) {
            Image(systemName: "mic.slash")
                .foregroundColor(.secondary)
            Text("Waiting for mic activity...")
                .foregroundColor(.secondary)
                .font(.callout)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var permissionView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Microphone access denied", systemImage: "mic.slash.circle.fill")
                .foregroundColor(.red)

            Text("Grant access in System Settings > Privacy & Security > Microphone")
                .font(.caption)
                .foregroundColor(.secondary)

            Button("Open System Settings") {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                    NSWorkspace.shared.open(url)
                }
            }
            .buttonStyle(.link)
        }
    }

    // MARK: - Threshold

    private var thresholdControl: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Threshold")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("\(Int(monitor.thresholdDB)) dBFS")
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.secondary)
            }

            Slider(value: $monitor.thresholdDB, in: -45...(-5), step: 1)

            HStack {
                Text("Strict")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Relaxed")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Bottom

    private var bottomControls: some View {
        HStack {
            Toggle("Monitor now", isOn: Binding(
                get: { monitor.manualOverride },
                set: { _ in monitor.toggleManual() }
            ))
            .toggleStyle(.switch)
            .controlSize(.small)

            Spacer()

            Button("Quit") {
                NSApp.terminate(nil)
            }
        }
    }
}

// MARK: - Level Meter

struct LevelMeter: View {
    let level: Float
    let threshold: Float

    private let minDB: Float = -60
    private let maxDB: Float = 0

    private func normalized(_ value: Float) -> CGFloat {
        CGFloat(max(0, min(1, (value - minDB) / (maxDB - minDB))))
    }

    private var levelColor: Color {
        if level > threshold { return .red }
        if level > threshold - 6 { return .orange }
        return .green
    }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                // Background track
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.primary.opacity(0.08))

                // Level fill
                RoundedRectangle(cornerRadius: 3)
                    .fill(levelColor)
                    .frame(width: max(0, normalized(level) * geo.size.width))

                // Threshold marker
                Rectangle()
                    .fill(Color.primary.opacity(0.5))
                    .frame(width: 2, height: geo.size.height + 4)
                    .offset(x: normalized(threshold) * geo.size.width - 1)
            }
        }
    }
}
