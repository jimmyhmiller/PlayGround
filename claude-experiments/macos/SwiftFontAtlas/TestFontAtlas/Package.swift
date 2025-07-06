// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TestFontAtlas",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "..")
    ],
    targets: [
        .executableTarget(
            name: "TestFontAtlas",
            dependencies: ["SwiftFontAtlas"]
        )
    ]
)