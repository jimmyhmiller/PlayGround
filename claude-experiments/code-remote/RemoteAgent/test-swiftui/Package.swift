// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TestSwiftUI",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "TestSwiftUI",
            swiftSettings: [.unsafeFlags(["-parse-as-library"])]
        )
    ]
)
