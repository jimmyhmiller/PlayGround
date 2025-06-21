import Foundation
import FoundationModels

// Test what tool capabilities exist in FoundationModels
let session = LanguageModelSession()

// Check if there are any tool-related methods
print("Session type: \(type(of: session))")
print("Available methods:")

// Try to see what's available
let mirror = Mirror(reflecting: session)
for child in mirror.children {
    if let label = child.label {
        print("- \(label): \(type(of: child.value))")
    }
}

// Test if tools can be created
struct TestTool {
    let name = "test"
    let description = "A test tool"
}

// Check if LanguageModelSession has tool-related initializers
print("\nChecking LanguageModelSession capabilities...")

// Let me try a simple response to see if there are any tool-calling patterns
Task {
    do {
        let response = try await session.respond(to: "Call a function named 'getWeather' with argument 'Tokyo'")
        print("Response: \(response.content)")
    } catch {
        print("Error: \(error)")
    }
}