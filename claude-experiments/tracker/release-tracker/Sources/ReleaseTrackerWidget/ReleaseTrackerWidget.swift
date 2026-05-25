import SwiftUI
import WidgetKit
import Shared

@main
struct ReleaseTrackerWidgetBundle: WidgetBundle {
    var body: some Widget {
        ReleaseTrackerWidget()
    }
}

struct ReleaseTrackerWidget: Widget {
    let kind = ChecklistStore.widgetKind

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: ChecklistProvider()) { entry in
            WidgetRootView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("Ease Release")
        .description("Track what's left before shipping Ease.")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge])
    }
}

struct ChecklistEntry: TimelineEntry {
    let date: Date
    let checklist: Checklist
}

struct ChecklistProvider: TimelineProvider {
    private let store = ChecklistStore()

    func placeholder(in context: Context) -> ChecklistEntry {
        ChecklistEntry(date: Date(), checklist: store.load())
    }

    func getSnapshot(in context: Context, completion: @escaping (ChecklistEntry) -> Void) {
        completion(ChecklistEntry(date: Date(), checklist: store.load()))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<ChecklistEntry>) -> Void) {
        // We don't need scheduled refreshes — WidgetCenter.reloadAllTimelines()
        // from the host app drives every refresh. A single never-refreshing
        // entry is what we want.
        let entry = ChecklistEntry(date: Date(), checklist: store.load())
        completion(Timeline(entries: [entry], policy: .never))
    }
}

// MARK: - Views

struct WidgetRootView: View {
    let entry: ChecklistEntry
    @Environment(\.widgetFamily) private var family

    var body: some View {
        switch family {
        case .systemSmall:
            SmallView(checklist: entry.checklist)
        case .systemMedium:
            MediumView(checklist: entry.checklist)
        case .systemLarge:
            LargeView(checklist: entry.checklist)
        default:
            MediumView(checklist: entry.checklist)
        }
    }
}

private struct ProgressHeader: View {
    let checklist: Checklist
    var compact: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Ease Release")
                    .font(compact ? .caption.weight(.medium) : .subheadline.weight(.semibold))
                Spacer()
                Text("\(checklist.doneCount)/\(checklist.totalCount)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: checklist.fraction)
                .progressViewStyle(.linear)
                .tint(.accentColor)
        }
    }
}

private struct SmallView: View {
    let checklist: Checklist

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ProgressHeader(checklist: checklist, compact: true)
            Spacer(minLength: 0)
            Text(percentString)
                .font(.system(.title, design: .rounded).weight(.semibold).monospacedDigit())
                .foregroundStyle(.primary)
            Text(remainingString)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .widgetURL(URL(string: "releasetracker://open"))
    }

    private var percentString: String {
        "\(Int((checklist.fraction * 100).rounded()))%"
    }

    private var remainingString: String {
        let remaining = checklist.totalCount - checklist.doneCount
        if remaining == 0 { return "Ready to ship 🎉" }
        return "\(remaining) left"
    }
}

private struct MediumView: View {
    let checklist: Checklist

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ProgressHeader(checklist: checklist)
            VStack(alignment: .leading, spacing: 4) {
                Text("Next up")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(Array(checklist.nextUp.prefix(3))) { item in
                    HStack(spacing: 6) {
                        Image(systemName: "circle")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Text(item.title)
                            .font(.caption)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
                if checklist.nextUp.isEmpty {
                    Text("All done — ready to ship.")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }
            Spacer(minLength: 0)
        }
        .widgetURL(URL(string: "releasetracker://open"))
    }
}

private struct LargeView: View {
    let checklist: Checklist

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ProgressHeader(checklist: checklist)
            VStack(alignment: .leading, spacing: 4) {
                Text("Next up")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(Array(checklist.nextUp.prefix(8))) { item in
                    HStack(spacing: 6) {
                        Image(systemName: "circle")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Text(item.title)
                            .font(.caption)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
                if checklist.nextUp.isEmpty {
                    Text("All done — ready to ship.")
                        .font(.callout)
                        .foregroundStyle(.green)
                }
            }
            Spacer(minLength: 0)
        }
        .widgetURL(URL(string: "releasetracker://open"))
    }
}
