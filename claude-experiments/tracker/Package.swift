// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Ease",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.5.0")
    ],
    targets: [
        .executableTarget(
            name: "Ease",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle")
            ],
            path: "Ease"
        )
    ]
)
