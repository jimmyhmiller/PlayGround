// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TestBaseline",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "..")
    ],
    targets: [
        .executableTarget(
            name: "TestBaseline",
            dependencies: ["SwiftFontAtlas"]
        )
    ]
)