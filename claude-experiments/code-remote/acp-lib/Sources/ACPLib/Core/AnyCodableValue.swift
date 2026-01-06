import Foundation

// MARK: - AnyCodableValue for dynamic JSON

/// A type-erased Codable value for handling dynamic JSON
public struct AnyCodableValue: Sendable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(_ encodable: any Encodable) {
        // Try to get the underlying value
        if let anyValue = encodable as? AnyCodableValue {
            self.value = anyValue.value
        } else {
            self.value = encodable
        }
    }

    public func decode<T: Decodable>(_ type: T.Type) throws -> T {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Encode self to data
        let data = try encoder.encode(self)
        // Decode as target type
        return try decoder.decode(type, from: data)
    }
}

extension AnyCodableValue: Decodable {
    public init(from decoder: Decoder) throws {
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
    public func encode(to encoder: Encoder) throws {
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
