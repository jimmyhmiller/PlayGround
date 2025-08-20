// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RenderThroughTest",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../")
    ],
    targets: [
        .executableTarget(
            name: "RenderThroughTest",
            dependencies: [
                .product(name: "SwiftFontAtlas", package: "SwiftFontAtlas")
            ]
        )
    ]
)