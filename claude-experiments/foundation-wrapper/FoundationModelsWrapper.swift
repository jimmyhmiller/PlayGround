import Foundation
import FoundationModels

// Internal weather tool that can be called from C
struct CWeatherTool: Tool {
    let name = "getWeather"
    let description = "Get weather for a city"
    let callFunction: (String) -> String
    
    @Generable
    struct Arguments {
        let city: String
    }
    
    var parameters: GenerationSchema {
        return Arguments.generationSchema
    }
    
    func call(arguments: Arguments) async throws -> ToolOutput {
        let result = callFunction(arguments.city)
        return ToolOutput(result)
    }
}

@objc public class FoundationModelsWrapper: NSObject {
    private var session: LanguageModelSession?
    private var tools: [any Tool] = []
    
    @objc public override init() {
        super.init()
        self.session = LanguageModelSession()
    }
    
    @objc public init(instructions: String) {
        super.init()
        self.session = LanguageModelSession(instructions: instructions)
    }
    
    @objc public init(useCase: Int) {
        super.init()
        let systemModel: SystemLanguageModel
        switch useCase {
        case 1: // Content tagging
            systemModel = SystemLanguageModel(useCase: .contentTagging)
        default:
            systemModel = SystemLanguageModel.default
        }
        self.session = LanguageModelSession(model: systemModel)
    }
    
    @objc public init(weatherToolFunction: @escaping (String) -> String, instructions: String?) {
        super.init()
        
        let weatherTool = CWeatherTool(callFunction: weatherToolFunction)
        self.tools = [weatherTool]
        
        if let instructions = instructions {
            self.session = LanguageModelSession(
                tools: [weatherTool],
                instructions: instructions
            )
        } else {
            self.session = LanguageModelSession(tools: [weatherTool])
        }
    }
    
    @objc public func respond(to prompt: String, completion: @escaping (String?, String?) -> Void) {
        guard let session = session else {
            completion(nil, "No session available")
            return
        }
        
        Task {
            do {
                let response = try await session.respond(to: prompt)
                let content = response.content
                await MainActor.run {
                    completion(content, nil)
                }
            } catch {
                await MainActor.run {
                    completion(nil, error.localizedDescription)
                }
            }
        }
    }
    
    @objc public var isResponding: Bool {
        return session?.isResponding ?? false
    }
    
    @objc public var transcript: String {
        guard let session = session else { return "" }
        return "Transcript with \(session.transcript.entries.count) entries"
    }
    
    @objc public static func checkAvailability() -> Bool {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            return true
        case .unavailable(_):
            return false
        }
    }
    
    @objc public static func getUnavailabilityReason() -> String? {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            return nil
        case .unavailable(let reason):
            return String(describing: reason)
        }
    }
}