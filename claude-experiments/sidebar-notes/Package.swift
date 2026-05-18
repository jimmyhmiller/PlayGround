// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SidebarNotes",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "SidebarNotes", targets: ["SidebarNotes"])
    ],
    targets: [
        .executableTarget(
            name: "SidebarNotes",
            path: "Sources",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        )
    ]
)
