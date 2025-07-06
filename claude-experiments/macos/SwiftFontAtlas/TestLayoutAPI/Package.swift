// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TestLayoutAPI",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "..")
    ],
    targets: [
        .executableTarget(
            name: "TestLayoutAPI",
            dependencies: ["SwiftFontAtlas"]
        )
    ]
)