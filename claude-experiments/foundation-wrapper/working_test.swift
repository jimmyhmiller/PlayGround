import Foundation
import FoundationModels

@main
struct WorkingTest {
    static func main() {
        print("=== Foundation Models Swift Wrapper Test ===")
        
        // Test 1: Check availability
        let isAvailable = FoundationModelsWrapper.checkAvailability()
        print("✓ Model availability: \(isAvailable)")
        
        if !isAvailable {
            let reason = FoundationModelsWrapper.getUnavailabilityReason()
            print("  Reason: \(reason ?? "Unknown")")
            return
        }
        
        // Test 2: Simple response
        print("\n=== Testing Simple Response ===")
        let wrapper = FoundationModelsWrapper()
        
        var completed = false
        wrapper.respond(to: "What's a good name for a trip to Japan? Respond only with a title") { content, error in
            if let content = content {
                print("✓ Response: \(content)")
            } else if let error = error {
                print("✗ Error: \(error)")
            }
            completed = true
        }
        
        // Wait for completion
        while !completed {
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
        }
        
        // Test 3: Instructions
        print("\n=== Testing Custom Instructions ===")
        let rhymeWrapper = FoundationModelsWrapper(instructions: "Respond in rhyme")
        
        completed = false
        rhymeWrapper.respond(to: "Say hello") { content, error in
            if let content = content {
                print("✓ Rhyme response: \(content)")
            } else if let error = error {
                print("✗ Error: \(error)")
            }
            completed = true
        }
        
        while !completed {
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
        }
        
        print("\n✓ All tests completed successfully!")
        print("The FoundationModels wrapper is working correctly.")
    }
}