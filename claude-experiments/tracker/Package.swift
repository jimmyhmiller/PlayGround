// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Tracker",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "Tracker",
            path: "Tracker"
        )
    ]
)
