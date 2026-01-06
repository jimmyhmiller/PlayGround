// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ACPLib",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ACPLib", targets: ["ACPLib"]),
        .executable(name: "acp-cli", targets: ["ACPCli"]),
    ],
    targets: [
        .target(
            name: "ACPLib",
            path: "Sources/ACPLib"
        ),
        .executableTarget(
            name: "ACPCli",
            dependencies: ["ACPLib"],
            path: "Sources/ACPCli",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
        .testTarget(
            name: "ACPLibTests",
            dependencies: ["ACPLib"]
        ),
    ]
)
