// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TextRenderingApp",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "..")
    ],
    targets: [
        .executableTarget(
            name: "TextRenderingApp",
            dependencies: ["SwiftFontAtlas"],
            resources: [
                .process("Metal")
            ]
        )
    ]
)