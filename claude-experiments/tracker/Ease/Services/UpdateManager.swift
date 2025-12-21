import Foundation
import Sparkle

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

@MainActor
class UpdateManager: NSObject, ObservableObject, SPUUpdaterDelegate {
    static let shared = UpdateManager()

    private(set) var updaterController: SPUStandardUpdaterController!

    @Published var currentChannel: UpdateChannel {
        didSet {
            UserDefaults.standard.set(currentChannel.rawValue, forKey: "updateChannel")
        }
    }

    override private init() {
        // Load saved channel preference
        let savedChannel = UserDefaults.standard.string(forKey: "updateChannel") ?? UpdateChannel.production.rawValue
        self.currentChannel = UpdateChannel(rawValue: savedChannel) ?? .production

        super.init()

        // Initialize the updater controller with self as delegate
        self.updaterController = SPUStandardUpdaterController(
            startingUpdater: true,
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
