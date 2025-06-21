import Foundation
import FoundationModels

@objc public class FoundationModelsWrapper: NSObject {
    private var session: LanguageModelSession?
    
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