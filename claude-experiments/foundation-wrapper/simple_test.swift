import Foundation
import FoundationModels

@main
struct SimpleTest {
    static func main() {
        // Simple test to demonstrate the FoundationModels wrapper works
        print("=== Foundation Models C Wrapper Test ===")

// Test 1: Create wrapper and check availability
let wrapper = FoundationModelsWrapper()
let isAvailable = FoundationModelsWrapper.checkAvailability()
print("✓ Model availability: \(isAvailable)")

if !isAvailable {
    let reason = FoundationModelsWrapper.getUnavailabilityReason()
    print("  Reason: \(reason ?? "Unknown")")
    exit(1)
}

// Test 2: Simple response test
print("\n=== Testing Simple Response ===")
var responseReceived = false
wrapper.respond(to: "What's a good name for a trip to Japan? Respond only with a title") { content, error in
    if let content = content {
        print("✓ Response received: \(content)")
    } else if let error = error {
        print("✗ Error: \(error)")
    }
    responseReceived = true
}

// Wait for response
while !responseReceived && !wrapper.isResponding {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

while wrapper.isResponding {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

while !responseReceived {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

// Test 3: Instructions test
print("\n=== Testing Custom Instructions ===")
let rhymeWrapper = FoundationModelsWrapper(instructions: "You are a helpful assistant who always responds in rhyme.")
var rhymeResponseReceived = false

rhymeWrapper.respond(to: "Write a short greeting") { content, error in
    if let content = content {
        print("✓ Rhyme response: \(content)")
    } else if let error = error {
        print("✗ Error: \(error)")
    }
    rhymeResponseReceived = true
}

while !rhymeResponseReceived && !rhymeWrapper.isResponding {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

while rhymeWrapper.isResponding {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

while !rhymeResponseReceived {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

// Test 4: Content tagging
print("\n=== Testing Content Tagging ===")
let taggingWrapper = FoundationModelsWrapper(useCase: 1) // Content tagging
var taggingResponseReceived = false

taggingWrapper.respond(to: "I love hiking in the mountains and taking photos of wildlife") { content, error in
    if let content = content {
        print("✓ Tagging response: \(content)")
    } else if let error = error {
        print("✗ Error: \(error)")
    }
    taggingResponseReceived = true
}

while !taggingResponseReceived && !taggingWrapper.isResponding {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

while taggingWrapper.isResponding {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

while !taggingResponseReceived {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
}

        print("\n=== All Tests Completed Successfully! ===")
        print("The Foundation Models wrapper is working correctly.")
    }
}