// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ReleaseTracker",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .target(
            name: "Shared",
            path: "Sources/Shared"
        ),
        .executableTarget(
            name: "ReleaseTracker",
            dependencies: ["Shared"],
            path: "Sources/ReleaseTracker"
        ),
        .executableTarget(
            name: "ReleaseTrackerWidget",
            dependencies: ["Shared"],
            path: "Sources/ReleaseTrackerWidget",
            // Widget extensions need:
            //   1. -application_extension so hardened-runtime symbols
            //      resolve to the extension-safe variants.
            //   2. The Info.plist embedded as a `__TEXT,__info_plist`
            //      segment. macOS's ExtensionKit / runningboardd reads
            //      that segment at process spawn to learn the extension
            //      type *before* the bundle's Info.plist is opened.
            //      Xcode does this automatically; SwiftPM doesn't.
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-application_extension",
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Resources/Info.widget.plist"
                ])
            ]
        )
    ]
)
