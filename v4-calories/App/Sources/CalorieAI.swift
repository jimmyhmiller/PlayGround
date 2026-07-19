import Foundation

/// A calorie estimate produced by the AI for a natural-language food description.
struct CalorieEstimate: Equatable {
    struct Item: Equatable, Identifiable {
        let name: String
        let kcal: Int
        var id: String { name }
    }
    let total: Int
    let low: Int
    let high: Int
    let label: String
    let items: [Item]

    /// The model returns total 0 / "Not food" when the text isn't something edible.
    var isFood: Bool { total > 0 }
}

enum CalorieAIError: LocalizedError {
    case noKey
    case empty
    case http(Int, String)
    case badResponse
    case network(String)

    var errorDescription: String? {
        switch self {
        case .noKey:        return "No DeepSeek API key set. Add one in Settings → Assistant."
        case .empty:        return "Describe what you ate first."
        case .http(let c, let m):
            return "AI service error (\(c))." + (m.isEmpty ? "" : " \(m)")
        case .badResponse:  return "Couldn't read the estimate. Try rephrasing."
        case .network(let m): return m
        }
    }
}

/// Thin DeepSeek (OpenAI-compatible) client that asks for a single structured tool call.
enum CalorieAI {
    private static let endpoint = URL(string: "https://api.deepseek.com/chat/completions")!

    private static let systemPrompt = """
    You are a nutrition estimator inside a calorie-tracking app. The user describes what \
    they ate in plain language. Estimate the total calories and call estimate_calories \
    exactly once. Assume typical home/restaurant portions when size is unspecified. Prefer \
    well-known brand or chain values when a brand is named. If the text is not food, set \
    total_kcal, low_kcal and high_kcal to 0 and label to "Not food".
    """

    private static var toolSpec: [String: Any] {
        [
            "type": "function",
            "function": [
                "name": "estimate_calories",
                "description": "Return the estimated calorie total for the described food.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "total_kcal": ["type": "integer", "description": "Best single estimate of total calories"],
                        "low_kcal":   ["type": "integer", "description": "Low end of a reasonable range"],
                        "high_kcal":  ["type": "integer", "description": "High end of a reasonable range"],
                        "label":      ["type": "string", "description": "Short label, max 4 words"],
                        "items": [
                            "type": "array",
                            "description": "Per-item breakdown",
                            "items": [
                                "type": "object",
                                "properties": [
                                    "name": ["type": "string"],
                                    "kcal": ["type": "integer"]
                                ],
                                "required": ["name", "kcal"]
                            ]
                        ]
                    ],
                    "required": ["total_kcal", "low_kcal", "high_kcal", "label", "items"]
                ]
            ]
        ]
    }

    static func estimate(_ description: String, apiKey: String) async throws -> CalorieEstimate {
        let text = description.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw CalorieAIError.empty }
        guard !apiKey.isEmpty else { throw CalorieAIError.noKey }

        let body: [String: Any] = [
            "model": "deepseek-chat",
            "messages": [
                ["role": "system", "content": systemPrompt],
                ["role": "user", "content": text]
            ],
            "tools": [toolSpec],
            "tool_choice": ["type": "function", "function": ["name": "estimate_calories"]],
            "temperature": 0.2
        ]

        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 30
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let data: Data, resp: URLResponse
        do {
            (data, resp) = try await URLSession.shared.data(for: req)
        } catch {
            throw CalorieAIError.network(error.localizedDescription)
        }

        guard let http = resp as? HTTPURLResponse else { throw CalorieAIError.badResponse }
        guard (200..<300).contains(http.statusCode) else {
            throw CalorieAIError.http(http.statusCode, apiMessage(data))
        }

        guard
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let choices = root["choices"] as? [[String: Any]],
            let message = choices.first?["message"] as? [String: Any],
            let calls = message["tool_calls"] as? [[String: Any]],
            let fn = calls.first?["function"] as? [String: Any],
            let argString = fn["arguments"] as? String,
            let argData = argString.data(using: .utf8),
            let args = try? JSONSerialization.jsonObject(with: argData) as? [String: Any]
        else {
            throw CalorieAIError.badResponse
        }

        let total = intVal(args["total_kcal"])
        let low = intVal(args["low_kcal"])
        let high = intVal(args["high_kcal"])
        let label = (args["label"] as? String)?.trimmingCharacters(in: .whitespaces) ?? "Estimate"
        let items: [CalorieEstimate.Item] = (args["items"] as? [[String: Any]] ?? []).compactMap {
            guard let name = ($0["name"] as? String)?.trimmingCharacters(in: .whitespaces), !name.isEmpty
            else { return nil }
            return CalorieEstimate.Item(name: name, kcal: intVal($0["kcal"]))
        }

        return CalorieEstimate(
            total: total,
            low: low > 0 ? min(low, total) : total,
            high: high > total ? high : total,
            label: label.isEmpty ? "Estimate" : label,
            items: items)
    }

    /// Pull a human-readable message out of an OpenAI-style error body, if present.
    private static func apiMessage(_ data: Data) -> String {
        if let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let err = root["error"] as? [String: Any],
           let msg = err["message"] as? String {
            return String(msg.prefix(160))
        }
        return ""
    }

    private static func intVal(_ v: Any?) -> Int {
        if let i = v as? Int { return i }
        if let d = v as? Double { return Int(d.rounded()) }
        if let s = v as? String, let d = Double(s) { return Int(d.rounded()) }
        return 0
    }
}
