// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "VisualFontDemo",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "..")
    ],
    targets: [
        .executableTarget(
            name: "VisualFontDemo",
            dependencies: ["SwiftFontAtlas"]
        )
    ]
)