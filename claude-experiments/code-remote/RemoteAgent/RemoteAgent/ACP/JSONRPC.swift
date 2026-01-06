import Foundation

// MARK: - JSON-RPC 2.0 Base Types

/// JSON-RPC version constant
let jsonrpcVersion = "2.0"

/// JSON-RPC Request
struct JSONRPCRequest: Codable, Sendable {
    let jsonrpc: String
    let id: String
    let method: String
    let params: AnyCodableValue?

    init(id: String, method: String, params: (any Encodable)? = nil) {
        self.jsonrpc = jsonrpcVersion
        self.id = id
        self.method = method
        if let params = params {
            self.params = AnyCodableValue(params)
        } else {
            self.params = nil
        }
    }
}

/// JSON-RPC Notification (no id = no response expected)
struct JSONRPCNotification: Codable, Sendable {
    let jsonrpc: String
    let method: String
    let params: AnyCodableValue?

    init(method: String, params: (any Encodable)? = nil) {
        self.jsonrpc = jsonrpcVersion
        self.method = method
        if let params = params {
            self.params = AnyCodableValue(params)
        } else {
            self.params = nil
        }
    }
}

/// JSON-RPC Success Response
struct JSONRPCSuccessResponse: Codable, Sendable {
    let jsonrpc: String
    let id: String
    let result: AnyCodableValue
}

/// JSON-RPC Error Object
struct JSONRPCError: Codable, Sendable, Error {
    let code: Int
    let message: String
    let data: AnyCodableValue?
}

/// JSON-RPC Error Response
struct JSONRPCErrorResponse: Codable, Sendable {
    let jsonrpc: String
    let id: String
    let error: JSONRPCError
}

/// JSON-RPC Response (success or error)
enum JSONRPCResponse: Sendable {
    case success(JSONRPCSuccessResponse)
    case error(JSONRPCErrorResponse)

    var id: String {
        switch self {
        case .success(let resp): return resp.id
        case .error(let resp): return resp.id
        }
    }
}

extension JSONRPCResponse: Decodable {
    enum CodingKeys: String, CodingKey {
        case jsonrpc, id, result, error
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let jsonrpc = try container.decode(String.self, forKey: .jsonrpc)
        let id = try container.decode(String.self, forKey: .id)

        if container.contains(.error) {
            let error = try container.decode(JSONRPCError.self, forKey: .error)
            self = .error(JSONRPCErrorResponse(jsonrpc: jsonrpc, id: id, error: error))
        } else {
            let result = try container.decode(AnyCodableValue.self, forKey: .result)
            self = .success(JSONRPCSuccessResponse(jsonrpc: jsonrpc, id: id, result: result))
        }
    }
}

/// Incoming message from the wire - could be request, response, or notification
enum JSONRPCMessage: Sendable {
    case request(JSONRPCRequest)
    case response(JSONRPCResponse)
    case notification(JSONRPCNotification)
}

extension JSONRPCMessage: Decodable {
    enum CodingKeys: String, CodingKey {
        case id, method, result, error
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let hasId = container.contains(.id)
        let hasMethod = container.contains(.method)
        let hasResult = container.contains(.result)
        let hasError = container.contains(.error)

        if hasMethod && hasId {
            // Request
            self = .request(try JSONRPCRequest(from: decoder))
        } else if hasMethod && !hasId {
            // Notification
            self = .notification(try JSONRPCNotification(from: decoder))
        } else if hasId && (hasResult || hasError) {
            // Response
            self = .response(try JSONRPCResponse(from: decoder))
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Cannot determine JSON-RPC message type"
                )
            )
        }
    }
}

// MARK: - AnyCodableValue for dynamic JSON

/// A type-erased Codable value for handling dynamic JSON
struct AnyCodableValue: Sendable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(_ encodable: any Encodable) {
        // Try to get the underlying value
        if let anyValue = encodable as? AnyCodableValue {
            self.value = anyValue.value
        } else {
            self.value = encodable
        }
    }

    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Encode self to data
        let data = try encoder.encode(self)
        // Decode as target type
        return try decoder.decode(type, from: data)
    }
}

extension AnyCodableValue: Decodable {
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self.value = NSNull()
        } else if let bool = try? container.decode(Bool.self) {
            self.value = bool
        } else if let int = try? container.decode(Int.self) {
            self.value = int
        } else if let double = try? container.decode(Double.self) {
            self.value = double
        } else if let string = try? container.decode(String.self) {
            self.value = string
        } else if let array = try? container.decode([AnyCodableValue].self) {
            self.value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodableValue].self) {
            self.value = dict.mapValues { $0.value }
        } else {
            self.value = NSNull()
        }
    }
}

extension AnyCodableValue: Encodable {
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch value {
        case is NSNull:
            try container.encodeNil()
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodableValue($0) })
        case let dict as [String: Any]:
            try container.encode(dict.mapValues { AnyCodableValue($0) })
        case let encodable as any Encodable:
            try encodable.encode(to: encoder)
        default:
            try container.encodeNil()
        }
    }
}
