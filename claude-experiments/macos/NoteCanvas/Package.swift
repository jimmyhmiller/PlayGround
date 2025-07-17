// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NoteCanvas",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "NoteCanvas",
            targets: ["NoteCanvas"]),
        .executable(
            name: "NoteCanvasApp",
            targets: ["NoteCanvasApp"])
    ],
    targets: [
        .target(
            name: "NoteCanvas",
            dependencies: []),
        .executableTarget(
            name: "NoteCanvasApp",
            dependencies: ["NoteCanvas"]),
        .testTarget(
            name: "NoteCanvasTests",
            dependencies: ["NoteCanvas"]),
        .testTarget(
            name: "NoteCanvasUITests",
            dependencies: ["NoteCanvas"])
    ]
)