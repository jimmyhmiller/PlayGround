// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ConversationExplorer",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "ConversationExplorer",
            path: "Sources/ConversationExplorer"
        )
    ]
)
