// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "FontAtlasViewer",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "..")
    ],
    targets: [
        .executableTarget(
            name: "FontAtlasViewer",
            dependencies: ["SwiftFontAtlas"]
        )
    ]
)