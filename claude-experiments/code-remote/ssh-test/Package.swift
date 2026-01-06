// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ssh-test",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/orlandos-nl/Citadel.git", from: "0.8.0"),
    ],
    targets: [
        .executableTarget(
            name: "ssh-test",
            dependencies: [
                .product(name: "Citadel", package: "Citadel"),
            ]
        ),
    ]
)
