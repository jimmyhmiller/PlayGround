import ActivityKit
import WidgetKit
import SwiftUI

struct ConnectionLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: ConnectionActivityAttributes.self) { context in
            // Lock Screen / Banner UI
            HStack(spacing: 12) {
                // Status indicator
                Image(systemName: "server.rack")
                    .font(.title2)
                    .foregroundStyle(statusColor(for: context.state.status))
                    .frame(width: 44)

                // Info
                VStack(alignment: .leading, spacing: 2) {
                    Text(context.attributes.projectName)
                        .font(.headline)
                        .lineLimit(1)

                    Text(context.attributes.serverName)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    HStack(spacing: 4) {
                        statusIndicator(for: context.state.status)
                        Text(context.state.currentOperation ?? "Ready")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                // Message count
                VStack {
                    Text("\(context.state.messagesExchanged)")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("msgs")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding()
            .activityBackgroundTint(.black.opacity(0.8))

        } dynamicIsland: { context in
            DynamicIsland {
                // Expanded view
                DynamicIslandExpandedRegion(.leading) {
                    HStack {
                        Image(systemName: "server.rack")
                            .foregroundStyle(.green)
                        Text(context.attributes.serverName)
                            .font(.caption)
                    }
                }

                DynamicIslandExpandedRegion(.trailing) {
                    Text("\(context.state.messagesExchanged) msgs")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                DynamicIslandExpandedRegion(.bottom) {
                    HStack {
                        statusIndicator(for: context.state.status)
                        Text(context.state.currentOperation ?? "Ready")
                            .font(.caption)
                    }
                }
            } compactLeading: {
                Image(systemName: "server.rack")
                    .foregroundStyle(statusColor(for: context.state.status))
            } compactTrailing: {
                statusIndicator(for: context.state.status)
            } minimal: {
                Image(systemName: "server.rack")
                    .foregroundStyle(statusColor(for: context.state.status))
            }
        }
    }

    @ViewBuilder
    private func statusIndicator(for status: ConnectionActivityAttributes.ContentState.ConnectionStatus) -> some View {
        switch status {
        case .connecting, .reconnecting:
            Image(systemName: "arrow.triangle.2.circlepath")
                .foregroundStyle(.yellow)
        case .connected, .idle:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .streaming:
            Image(systemName: "arrow.down.circle.fill")
                .foregroundStyle(.blue)
        case .toolRunning:
            Image(systemName: "gearshape.fill")
                .foregroundStyle(.orange)
        case .error:
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(.red)
        }
    }

    private func statusColor(for status: ConnectionActivityAttributes.ContentState.ConnectionStatus) -> Color {
        switch status {
        case .connecting, .reconnecting: return .yellow
        case .connected, .idle: return .green
        case .streaming: return .blue
        case .toolRunning: return .orange
        case .error: return .red
        }
    }
}
