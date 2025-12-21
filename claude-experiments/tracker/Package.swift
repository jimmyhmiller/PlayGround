// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Ease",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "Ease",
            path: "Ease"
        )
    ]
)
