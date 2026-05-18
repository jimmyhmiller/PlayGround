import Foundation
#if !MAS
import Sparkle
#endif

enum UpdateChannel: String, CaseIterable {
    case production = "production"
    case local = "local"

    var appcastURL: URL {
        switch self {
        case .production:
            // TODO: Replace with your actual production appcast URL
            return URL(string: "https://github.com/jimmyhmiller/ease/releases/latest/download/appcast.xml")!
        case .local:
            // Local development server for testing updates
            return URL(string: "http://localhost:8080/appcast.xml")!
        }
    }

    var displayName: String {
        switch self {
        case .production:
            return "Stable"
        case .local:
            return "Local Dev"
        }
    }
}

#if MAS

// Mac App Store builds receive updates through the App Store itself.
// UpdateManager is reduced to a stub so the rest of the app can compile
// against the same API surface without linking Sparkle.
@MainActor
final class UpdateManager: NSObject, ObservableObject {
    static let shared = UpdateManager()

    @Published var currentChannel: UpdateChannel = .production

    let canCheckForUpdates: Bool = false
    var automaticallyChecksForUpdates: Bool = false
    let lastUpdateCheckDate: Date? = nil
    let isAvailable: Bool = false

    func checkForUpdates() {}
}

#else

@MainActor
class UpdateManager: NSObject, ObservableObject, SPUUpdaterDelegate {
    static let shared = UpdateManager()

    private(set) var updaterController: SPUStandardUpdaterController!

    @Published var currentChannel: UpdateChannel {
        didSet {
            UserDefaults.standard.set(currentChannel.rawValue, forKey: "updateChannel")
        }
    }

    let isAvailable: Bool = true

    override private init() {
        // Load saved channel preference
        let savedChannel = UserDefaults.standard.string(forKey: "updateChannel") ?? UpdateChannel.production.rawValue
        self.currentChannel = UpdateChannel(rawValue: savedChannel) ?? .production

        super.init()

        // Only start the updater when running inside a signed app bundle
        let inAppBundle = Bundle.main.bundlePath.hasSuffix(".app")
        self.updaterController = SPUStandardUpdaterController(
            startingUpdater: inAppBundle,
            updaterDelegate: self,
            userDriverDelegate: nil
        )
    }

    // MARK: - SPUUpdaterDelegate

    nonisolated func feedURLString(for updater: SPUUpdater) -> String? {
        // Access the current channel from UserDefaults directly to avoid actor isolation issues
        let savedChannel = UserDefaults.standard.string(forKey: "updateChannel") ?? UpdateChannel.production.rawValue
        let channel = UpdateChannel(rawValue: savedChannel) ?? .production
        return channel.appcastURL.absoluteString
    }

    // MARK: - Public API

    func checkForUpdates() {
        updaterController.checkForUpdates(nil)
    }

    var canCheckForUpdates: Bool {
        updaterController.updater.canCheckForUpdates
    }

    var automaticallyChecksForUpdates: Bool {
        get { updaterController.updater.automaticallyChecksForUpdates }
        set { updaterController.updater.automaticallyChecksForUpdates = newValue }
    }

    var lastUpdateCheckDate: Date? {
        updaterController.updater.lastUpdateCheckDate
    }
}

#endif
