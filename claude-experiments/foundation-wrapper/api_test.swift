import Foundation
import FoundationModels

// Test to see what's available in the API
let session = LanguageModelSession()
print("Session created: \(session)")
print("Is responding: \(session.isResponding)")

let model = SystemLanguageModel.default
print("Model: \(model)")
print("Availability: \(model.availability)")

switch model.availability {
case .available:
    print("Model is available")
case .unavailable(let reason):
    print("Model unavailable, reason: \(reason)")
}

// Check what transcript returns
print("Transcript type: \(type(of: session.transcript))")

Task {
    do {
        let response = try await session.respond(to: "Hello")
        print("Response type: \(type(of: response))")
        print("Content type: \(type(of: response.content))")
        print("Content: \(response.content)")
        print("Transcript after response: \(session.transcript)")
    } catch {
        print("Error: \(error)")
    }
}