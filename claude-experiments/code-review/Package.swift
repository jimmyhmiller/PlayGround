// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AgentReview",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "AgentReview", targets: ["AgentReview"])
    ],
    targets: [
        .executableTarget(
            name: "AgentReview",
            path: "Sources",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
        .testTarget(
            name: "AgentReviewTests",
            dependencies: ["AgentReview"],
            path: "Tests"
        )
    ]
)
