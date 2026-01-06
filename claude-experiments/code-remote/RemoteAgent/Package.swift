// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RemoteAgent",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .executable(name: "RemoteAgent", targets: ["RemoteAgent"]),
    ],
    dependencies: [
        .package(url: "https://github.com/orlandos-nl/Citadel.git", from: "0.7.0"),
        .package(path: "../acp-lib"),
    ],
    targets: [
        .executableTarget(
            name: "RemoteAgent",
            dependencies: [
                .product(name: "Citadel", package: "Citadel"),
                .product(name: "ACPLib", package: "acp-lib"),
            ],
            path: "RemoteAgent",
            exclude: ["Assets.xcassets", "Package.resolved", "ACP"],
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
    ]
)
