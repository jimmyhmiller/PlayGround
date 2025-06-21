import Foundation
import FoundationModels

// Simple test to verify FoundationModels is available
let session = LanguageModelSession()
print("FoundationModels session created successfully")

Task {
    do {
        let response = try await session.respond(to: "Hello world")
        print("Response: \(response.content)")
    } catch {
        print("Error: \(error)")
    }
}

// Keep the program running
RunLoop.main.run()