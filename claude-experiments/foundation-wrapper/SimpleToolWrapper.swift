import Foundation
import FoundationModels

// Simple tool for testing
public struct WeatherTool: Tool {
    public let name = "getWeather"
    public let description = "Get weather information for a city"
    
    @Generable
    public struct Arguments {
        @Guide(description: "The city to get weather for")
        public let city: String
    }
    
    public var parameters: GenerationSchema {
        return Arguments.generationSchema
    }
    
    public func call(arguments: Arguments) async throws -> ToolOutput {
        // Call the C function
        let temperature = 72 // Mock for now
        return ToolOutput("The temperature in \(arguments.city) is \(temperature)°F with clear skies.")
    }
}

// Test function
@objc public class SimpleToolTest: NSObject {
    @objc public static func testTool() {
        let weatherTool = WeatherTool()
        let session = LanguageModelSession(
            tools: [weatherTool],
            instructions: "You help users with weather information. Use the getWeather tool when users ask about weather."
        )
        
        Task {
            do {
                let response = try await session.respond(to: "What's the weather in Tokyo?")
                print("Tool Response: \(response.content)")
            } catch {
                print("Error: \(error)")
            }
        }
    }
}