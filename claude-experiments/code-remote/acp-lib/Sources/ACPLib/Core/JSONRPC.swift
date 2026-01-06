import Foundation

// MARK: - JSON-RPC 2.0 Base Types

/// JSON-RPC version constant
public let jsonrpcVersion = "2.0"

/// JSON-RPC ID can be string or number
public enum JSONRPCId: Codable, Sendable, Hashable, CustomStringConvertible {
    case string(String)
    case number(Int)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let stringValue = try? container.decode(String.self) {
            self = .string(stringValue)
        } else if let intValue = try? container.decode(Int.self) {
            self = .number(intValue)
        } else {
            throw DecodingError.typeMismatch(
                JSONRPCId.self,
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected string or number for JSON-RPC id")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .number(let n): try container.encode(n)
        }
    }

    public var description: String {
        switch self {
        case .string(let s): return s
        case .number(let n): return String(n)
        }
    }

    public var stringValue: String { description }
}

/// JSON-RPC Request
public struct JSONRPCRequest: Codable, Sendable {
    public let jsonrpc: String
    public let id: JSONRPCId
    public let method: String
    public let params: AnyCodableValue?

    public init(id: String, method: String, params: (any Encodable)? = nil) {
        self.jsonrpc = jsonrpcVersion
        self.id = .string(id)
        self.method = method
        if let params = params {
            self.params = AnyCodableValue(params)
        } else {
            self.params = nil
        }
    }

    public init(id: JSONRPCId, method: String, params: (any Encodable)? = nil) {
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
public struct JSONRPCNotification: Codable, Sendable {
    public let jsonrpc: String
    public let method: String
    public let params: AnyCodableValue?

    public init(method: String, params: (any Encodable)? = nil) {
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
public struct JSONRPCSuccessResponse: Codable, Sendable {
    public let jsonrpc: String
    public let id: JSONRPCId
    public let result: AnyCodableValue

    public init(jsonrpc: String = jsonrpcVersion, id: JSONRPCId, result: AnyCodableValue) {
        self.jsonrpc = jsonrpc
        self.id = id
        self.result = result
    }
}

/// JSON-RPC Error Object
public struct JSONRPCError: Codable, Sendable, Error {
    public let code: Int
    public let message: String
    public let data: AnyCodableValue?
}

/// JSON-RPC Error Response
public struct JSONRPCErrorResponse: Codable, Sendable {
    public let jsonrpc: String
    public let id: JSONRPCId
    public let error: JSONRPCError
}

/// JSON-RPC Response (success or error)
public enum JSONRPCResponse: Sendable {
    case success(JSONRPCSuccessResponse)
    case error(JSONRPCErrorResponse)

    public var id: JSONRPCId {
        switch self {
        case .success(let resp): return resp.id
        case .error(let resp): return resp.id
        }
    }
}

extension JSONRPCResponse: Codable {
    enum CodingKeys: String, CodingKey {
        case jsonrpc, id, result, error
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let jsonrpc = try container.decode(String.self, forKey: .jsonrpc)
        let id = try container.decode(JSONRPCId.self, forKey: .id)

        if container.contains(.error) {
            let error = try container.decode(JSONRPCError.self, forKey: .error)
            self = .error(JSONRPCErrorResponse(jsonrpc: jsonrpc, id: id, error: error))
        } else {
            let result = try container.decode(AnyCodableValue.self, forKey: .result)
            self = .success(JSONRPCSuccessResponse(jsonrpc: jsonrpc, id: id, result: result))
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .success(let response):
            try response.encode(to: encoder)
        case .error(let response):
            try response.encode(to: encoder)
        }
    }
}

/// Incoming message from the wire - could be request, response, or notification
public enum JSONRPCMessage: Sendable {
    case request(JSONRPCRequest)
    case response(JSONRPCResponse)
    case notification(JSONRPCNotification)
}

extension JSONRPCMessage: Decodable {
    enum CodingKeys: String, CodingKey {
        case id, method, result, error
    }

    public init(from decoder: Decoder) throws {
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
