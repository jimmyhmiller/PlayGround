import Foundation
import FoundationModels

// Simple C-compatible tool that actually works
@objc public final class CToolWrapper: NSObject, Tool {
    public let name: String
    public let toolDescription: String
    private let callFunction: @Sendable (String) -> String
    
    // Override NSObject's description to return our tool description  
    public override var description: String {
        return toolDescription
    }
    
    // Simple @Generable struct - just one string field for the argument
    @Generable
    struct Arguments {
        let input: String  // Single string input from C
    }
    
    public var parameters: GenerationSchema {
        return Arguments.generationSchema
    }
    
    @objc public init(name: String, description: String, callFunction: @escaping @Sendable (String) -> String) {
        self.name = name
        self.toolDescription = description
        self.callFunction = callFunction
        super.init()
    }
    
    public func call(arguments: Arguments) async throws -> ToolOutput {
        let result = callFunction(arguments.input)
        return ToolOutput(result)
    }
}

@objc public class FoundationModelsWrapper: NSObject {
    private var session: LanguageModelSession?
    private var cTools: [CToolWrapper] = []
    
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
    
    @objc public init(tools toolArray: [CToolWrapper], instructions: String?) {
        super.init()
        self.cTools = toolArray
        
        if let instructions = instructions {
            self.session = LanguageModelSession(
                tools: toolArray,
                instructions: instructions
            )
        } else {
            self.session = LanguageModelSession(tools: toolArray)
        }
    }
    
    @objc public func addTool(_ tool: CToolWrapper) {
        cTools.append(tool)
        // Recreate session with updated tools
        if let currentSession = session {
            // Get current instructions if any
            let instructions = currentSession.transcript.entries.isEmpty ? nil : "Continue the conversation"
            self.session = LanguageModelSession(
                tools: cTools,
                instructions: instructions
            )
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