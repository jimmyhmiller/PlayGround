import Foundation

struct StackTrace: Codable {
    let program: String
    let totalCaptures: Int
    let captures: [Capture]
    
    enum CodingKeys: String, CodingKey {
        case program
        case totalCaptures = "total_captures"
        case captures
    }
}

struct Capture: Codable {
    let callNumber: Int
    let eventType: String
    let functionName: String
    let stackBase: String
    let stackDepth: Int
    let frames: [Frame]
    
    enum CodingKeys: String, CodingKey {
        case callNumber = "call_number"
        case eventType = "event_type"
        case functionName = "function_name"
        case stackBase = "stack_base"
        case stackDepth = "stack_depth"
        case frames
    }
}

struct Frame: Codable {
    let depth: Int
    let function: String
    let pc: String
    let sp: String
    let fp: String
}

class StackTraceParser {
    static func parse(from url: URL) throws -> StackTrace {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(StackTrace.self, from: data)
    }
    
    static func parse(from data: Data) throws -> StackTrace {
        let decoder = JSONDecoder()
        return try decoder.decode(StackTrace.self, from: data)
    }
}