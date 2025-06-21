import Foundation
import FoundationModels

// Let's examine the Tool protocol requirements
print("=== Tool Protocol Requirements ===")

// Try to implement a proper tool
struct WeatherTool: Tool {
    let name = "getWeather"
    let description = "Get weather information for a city"
    
    @Generable
    struct Arguments {
        @Guide(description: "The city to get weather for")
        let city: String
    }
    
    var parameters: GenerationSchema {
        return Arguments.generationSchema
    }
    
    func call(arguments: Arguments) async throws -> ToolOutput {
        let temperature = 72 // Mock temperature
        return ToolOutput("The temperature in \(arguments.city) is \(temperature)°F with clear skies.")
    }
}

let weatherTool = WeatherTool()
print("✓ Weather tool created: \(weatherTool.name)")

// Create session with tools
let session = LanguageModelSession(
    tools: [weatherTool],
    instructions: "You help users with weather information. Use the getWeather tool when users ask about weather."
)

print("✓ Session created with tools")

// Test tool usage
Task {
    do {
        print("\n=== Testing Tool Usage ===")
        let response = try await session.respond(to: "What's the weather like in Tokyo?")
        print("Response: \(response.content)")
    } catch {
        print("Error: \(error)")
    }
    
    exit(0)
}

RunLoop.main.run()