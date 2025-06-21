import Foundation
import FoundationModels

// Create a working tool that can be called from C
struct WeatherTool: Tool {
    let name = "getWeather"
    let description = "Get weather for a city"
    
    // Simple @Generable struct
    @Generable
    struct Arguments {
        let city: String
    }
    
    var parameters: GenerationSchema {
        return Arguments.generationSchema
    }
    
    func call(arguments: Arguments) async throws -> ToolOutput {
        print("🔧 WEATHER TOOL CALLED! City: \(arguments.city)")
        return ToolOutput("The weather in \(arguments.city) is 75°F with clear skies from the Swift tool.")
    }
}

// Test the tool
print("=== Testing Weather Tool ===")

let weatherTool = WeatherTool()
let session = LanguageModelSession(
    tools: [weatherTool],
    instructions: "You help with weather. Use the getWeather tool when asked about weather."
)

Task {
    do {
        let response = try await session.respond(to: "What's the weather in Paris? Use the getWeather tool to find out.")
        print("Response: \(response.content)")
    } catch {
        print("Error: \(error)")
    }
    exit(0)
}

RunLoop.main.run()