import Foundation
import FoundationModels

// Test the Tool API
print("=== Testing FoundationModels Tool API ===")

// Check what Tool protocol or class exists
print("Tool type: \(Tool.self)")

// Try to create a simple tool
struct WeatherTool: Tool {
    let name = "getWeather"
    let description = "Get weather information for a city"
    
    struct Arguments: Codable {
        let city: String
    }
    
    func call(arguments: Arguments) async throws -> ToolOutput {
        let temperature = 72 // Mock temperature
        return ToolOutput("The temperature in \(arguments.city) is \(temperature)°F")
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
print("Tool count: \(session.tools.count)")

// Test tool usage
Task {
    do {
        print("\n=== Testing Tool Usage ===")
        let response = try await session.respond(to: "What's the weather in Tokyo?")
        print("Response: \(response.content)")
    } catch {
        print("Error: \(error)")
    }
    
    exit(0)
}

RunLoop.main.run()