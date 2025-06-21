import Foundation
import FoundationModels

// Simplest possible tool - just takes a string, returns a string
struct SimpleWeatherTool: Tool {
    let name = "getWeather"
    let description = "Get weather for a city"
    
    // Minimal @Generable struct - just one string field
    @Generable
    struct Arguments {
        let city: String
    }
    
    var parameters: GenerationSchema {
        return Arguments.generationSchema
    }
    
    func call(arguments: Arguments) async throws -> ToolOutput {
        // Simple weather response
        return ToolOutput("The weather in \(arguments.city) is 72°F and sunny.")
    }
}

// Test the minimal tool
print("=== Testing Minimal Tool ===")

let weatherTool = SimpleWeatherTool()
let session = LanguageModelSession(
    tools: [weatherTool],
    instructions: "You help with weather. Use the getWeather tool when asked about weather."
)

Task {
    do {
        let response = try await session.respond(to: "What's the weather in Paris?")
        print("Response: \(response.content)")
    } catch {
        print("Error: \(error)")
    }
    exit(0)
}

RunLoop.main.run()