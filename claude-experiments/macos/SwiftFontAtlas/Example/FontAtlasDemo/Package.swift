// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "FontAtlasDemo",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "FontAtlasDemo",
            targets: ["FontAtlasDemo"]),
    ],
    dependencies: [
        .package(path: "../..") // SwiftFontAtlas
    ],
    targets: [
        .executableTarget(
            name: "FontAtlasDemo",
            dependencies: ["SwiftFontAtlas"],
            resources: [
                .process("Resources")
            ]
        ),
    ]
)