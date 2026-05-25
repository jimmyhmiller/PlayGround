import SwiftUI
import Combine
import Shared

@MainActor
final class FloatingWidgetModel: ObservableObject {
    @Published private(set) var checklist: Checklist
    private let store = ChecklistStore()
    private var watcher: DispatchSourceFileSystemObject?

    init() {
        self.checklist = store.load()
        startWatching()
    }

    func reload() {
        checklist = store.load()
    }

    /// Watch the checklist file so the floating widget refreshes when the
    /// main window toggles items.
    private func startWatching() {
        let url = store.fileURL
        // The file may not exist yet — touch the parent directory instead.
        let watchURL = FileManager.default.fileExists(atPath: url.path)
            ? url
            : url.deletingLastPathComponent()
        let fd = open(watchURL.path, O_EVTONLY)
        guard fd >= 0 else { return }
        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: [.write, .extend, .rename],
            queue: .main
        )
        source.setEventHandler { [weak self] in
            self?.reload()
            // If we were watching the directory, switch to the file once it appears.
            if watchURL != url, FileManager.default.fileExists(atPath: url.path) {
                self?.startWatching()
            }
        }
        source.setCancelHandler { close(fd) }
        source.resume()
        watcher?.cancel()
        watcher = source
    }

    deinit {
        watcher?.cancel()
    }
}

struct FloatingWidgetView: View {
    @StateObject private var model = FloatingWidgetModel()

    var body: some View {
        ZStack {
            // Material background that visually echoes a system widget.
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(.regularMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(.white.opacity(0.08), lineWidth: 0.5)
                )

            VStack(alignment: .leading, spacing: 10) {
                header
                progress
                Divider().opacity(0.4)
                upcoming
                Spacer(minLength: 0)
            }
            .padding(14)
        }
        .frame(width: 280, height: 180)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Ease Release")
                .font(.subheadline.weight(.semibold))
            Spacer()
            Text("\(model.checklist.doneCount) / \(model.checklist.totalCount)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Button {
                openChecklistWindow()
            } label: {
                Image(systemName: "rectangle.expand.vertical")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Open full checklist")
        }
    }

    private var progress: some View {
        ProgressView(value: model.checklist.fraction)
            .progressViewStyle(.linear)
            .tint(.accentColor)
    }

    private var upcoming: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Next up")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(Array(model.checklist.nextUp.prefix(3))) { item in
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
            if model.checklist.nextUp.isEmpty {
                Text("All done — ready to ship.")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
        }
    }

    private func openChecklistWindow() {
        ChecklistWindowController.shared.show()
    }
}
