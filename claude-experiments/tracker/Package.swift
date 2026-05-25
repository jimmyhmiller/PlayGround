// swift-tools-version: 5.9
import PackageDescription
import Foundation

// MAS builds set EASE_MAS=1 in the environment before invoking `swift build`.
// When set, we drop the Sparkle dependency entirely so the Mac App Store
// build is free of any external auto-updater (which is App Store-prohibited).
let isMAS = ProcessInfo.processInfo.environment["EASE_MAS"] == "1"

let dependencies: [Package.Dependency] = isMAS
    ? []
    : [.package(url: "https://github.com/sparkle-project/Sparkle", from: "2.5.0")]

let targetDependencies: [Target.Dependency] = isMAS
    ? []
    : [.product(name: "Sparkle", package: "Sparkle")]

let swiftSettings: [SwiftSetting] = isMAS
    ? [.define("MAS")]
    : []

let package = Package(
    name: "Ease",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: dependencies,
    targets: [
        .executableTarget(
            name: "Ease",
            dependencies: targetDependencies,
            path: "Ease",
            resources: [
                .copy("AppIcon.icon")
            ],
            swiftSettings: swiftSettings
        )
    ]
)
