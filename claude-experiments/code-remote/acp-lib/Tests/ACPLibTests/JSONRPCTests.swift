import XCTest
@testable import ACPLib

final class JSONRPCTests: XCTestCase {
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    // MARK: - Request Tests

    func testRequestEncoding() throws {
        let request = JSONRPCRequest(
            id: "1",
            method: "session/new",
            params: ACPSessionNewParams(cwd: "/test")
        )

        let data = try encoder.encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["jsonrpc"] as? String, "2.0")
        XCTAssertEqual(json["id"] as? String, "1")
        XCTAssertEqual(json["method"] as? String, "session/new")
        XCTAssertNotNil(json["params"])
    }

    func testRequestEncodingWithoutParams() throws {
        let request = JSONRPCRequest(id: "1", method: "ping", params: nil as String?)

        let data = try encoder.encode(request)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["jsonrpc"] as? String, "2.0")
        XCTAssertEqual(json["id"] as? String, "1")
        XCTAssertEqual(json["method"] as? String, "ping")
    }

    // MARK: - Response Tests

    func testSuccessResponseDecoding() throws {
        let json = """
        {"jsonrpc":"2.0","id":"1","result":{"sessionId":"test-123"}}
        """.data(using: .utf8)!

        let response = try decoder.decode(JSONRPCResponse.self, from: json)

        if case .success(let success) = response {
            let result = try success.result.decode(ACPSessionNewResult.self)
            XCTAssertEqual(result.sessionId, "test-123")
        } else {
            XCTFail("Expected success response")
        }
    }

    func testErrorResponseDecoding() throws {
        let json = """
        {"jsonrpc":"2.0","id":"1","error":{"code":-32001,"message":"Session not found"}}
        """.data(using: .utf8)!

        let response = try decoder.decode(JSONRPCResponse.self, from: json)

        if case .error(let error) = response {
            XCTAssertEqual(error.error.code, -32001)
            XCTAssertEqual(error.error.message, "Session not found")
        } else {
            XCTFail("Expected error response")
        }
    }

    // MARK: - Notification Tests

    func testNotificationEncoding() throws {
        let notification = JSONRPCNotification(
            method: "session/cancel",
            params: ACPSessionCancelParams(sessionId: "test-123")
        )

        let data = try encoder.encode(notification)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["jsonrpc"] as? String, "2.0")
        XCTAssertEqual(json["method"] as? String, "session/cancel")
        XCTAssertNil(json["id"]) // Notifications don't have id
    }

    func testNotificationDecoding() throws {
        let json = """
        {"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"test"}}
        """.data(using: .utf8)!

        let message = try decoder.decode(JSONRPCMessage.self, from: json)

        if case .notification(let notification) = message {
            XCTAssertEqual(notification.method, "session/update")
        } else {
            XCTFail("Expected notification")
        }
    }

    // MARK: - Message Type Detection

    func testMessageTypeDetectionRequest() throws {
        let json = """
        {"jsonrpc":"2.0","id":"1","method":"test"}
        """.data(using: .utf8)!

        let message = try decoder.decode(JSONRPCMessage.self, from: json)

        if case .request(let request) = message {
            XCTAssertEqual(request.id, .string("1"))
            XCTAssertEqual(request.method, "test")
        } else {
            XCTFail("Expected request")
        }
    }

    func testRequestWithNumericId() throws {
        // Permission requests from agent use numeric IDs
        let json = """
        {"jsonrpc":"2.0","id":0,"method":"session/request_permission","params":{}}
        """.data(using: .utf8)!

        let message = try decoder.decode(JSONRPCMessage.self, from: json)

        if case .request(let request) = message {
            XCTAssertEqual(request.id, .number(0))
            XCTAssertEqual(request.method, "session/request_permission")
        } else {
            XCTFail("Expected request with numeric id")
        }
    }

    func testResponseWithNumericId() throws {
        let json = """
        {"jsonrpc":"2.0","id":42,"result":{}}
        """.data(using: .utf8)!

        let response = try decoder.decode(JSONRPCResponse.self, from: json)

        if case .success(let success) = response {
            XCTAssertEqual(success.id, .number(42))
        } else {
            XCTFail("Expected success response")
        }
    }

    func testMessageTypeDetectionResponse() throws {
        let json = """
        {"jsonrpc":"2.0","id":"1","result":{}}
        """.data(using: .utf8)!

        let message = try decoder.decode(JSONRPCMessage.self, from: json)

        if case .response = message {
            // Success
        } else {
            XCTFail("Expected response")
        }
    }

    func testMessageTypeDetectionNotification() throws {
        let json = """
        {"jsonrpc":"2.0","method":"update"}
        """.data(using: .utf8)!

        let message = try decoder.decode(JSONRPCMessage.self, from: json)

        if case .notification(let notification) = message {
            XCTAssertEqual(notification.method, "update")
        } else {
            XCTFail("Expected notification")
        }
    }

    // MARK: - AnyCodableValue Tests

    func testAnyCodableValueString() throws {
        let value = AnyCodableValue("hello")
        let data = try encoder.encode(value)
        let decoded = try decoder.decode(AnyCodableValue.self, from: data)

        XCTAssertEqual(decoded.value as? String, "hello")
    }

    func testAnyCodableValueInt() throws {
        let value = AnyCodableValue(42)
        let data = try encoder.encode(value)
        let decoded = try decoder.decode(AnyCodableValue.self, from: data)

        XCTAssertEqual(decoded.value as? Int, 42)
    }

    func testAnyCodableValueBool() throws {
        let value = AnyCodableValue(true)
        let data = try encoder.encode(value)
        let decoded = try decoder.decode(AnyCodableValue.self, from: data)

        XCTAssertEqual(decoded.value as? Bool, true)
    }

    func testAnyCodableValueArray() throws {
        let json = "[1, 2, 3]".data(using: .utf8)!
        let decoded = try decoder.decode(AnyCodableValue.self, from: json)

        let array = decoded.value as? [Any]
        XCTAssertNotNil(array)
        XCTAssertEqual(array?.count, 3)
    }

    func testAnyCodableValueDictionary() throws {
        let json = """
        {"key": "value", "number": 42}
        """.data(using: .utf8)!

        let decoded = try decoder.decode(AnyCodableValue.self, from: json)

        let dict = decoded.value as? [String: Any]
        XCTAssertNotNil(dict)
        XCTAssertEqual(dict?["key"] as? String, "value")
        XCTAssertEqual(dict?["number"] as? Int, 42)
    }
}
