import Foundation

// MARK: - App-level enums

enum Screen {
    case empty
    case main
    case submitted
}

enum DiffMode: String, CaseIterable {
    case unified = "Unified"
    case split = "Split"
}

enum Verdict: String, CaseIterable {
    case comment = "Comment"
    case approve = "Approve"
    case requestChanges = "Request changes"

    var ghFlag: String {
        switch self {
        case .comment: return "--comment"
        case .approve: return "--approve"
        case .requestChanges: return "--request-changes"
        }
    }
}

enum Selection: Equatable {
    case none
    case pr(projectID: String, number: Int)
    case workingTree(projectID: String)
}

// MARK: - Projects & pull requests

struct Project: Identifiable, Hashable {
    let name: String
    let path: URL
    let modified: Date

    var id: String { path.path }
}

struct ChecksSummary: Hashable, Codable {
    var passed = 0
    var failed = 0
    var pending = 0

    var total: Int { passed + failed + pending }
}

struct PullRequest: Identifiable, Hashable, Codable {
    let number: Int
    let title: String
    let author: String
    let authorIsBot: Bool
    let branch: String
    let updatedAt: Date?
    let state: String          // OPEN / MERGED / CLOSED
    let isDraft: Bool
    let reviewDecision: String // APPROVED / CHANGES_REQUESTED / REVIEW_REQUIRED / ""
    let additions: Int
    let deletions: Int
    let checks: ChecksSummary

    var id: Int { number }

    /// One-word status used for the colored dot + sidebar subtitle.
    var statusLabel: String {
        if state == "MERGED" { return "merged" }
        if state == "CLOSED" { return "closed" }
        if isDraft { return "draft" }
        switch reviewDecision {
        case "APPROVED": return "approved"
        case "CHANGES_REQUESTED": return "changes requested"
        default: return "open"
        }
    }
}

// MARK: - Diffs

enum LineKind {
    case ctx, add, del
}

struct DiffLine: Hashable {
    let kind: LineKind
    let oldNo: Int?
    let newNo: Int?
    let text: String
    /// The file ends here with no trailing newline (git's "\ No newline at end
    /// of file"). Must survive into rebuilt patches or applying one would
    /// silently add a newline the author didn't write.
    var noNewline: Bool = false
}

struct Hunk: Identifiable, Hashable {
    let id: String
    let header: String
    let lines: [DiffLine]
    /// The `diff --git a/x b/x` preamble, kept so patches can be rebuilt for
    /// split hunks and partial staging.
    let fileHeader: String
    /// Self-contained patch (file header + this hunk) usable with `git apply --cached`.
    let rawPatch: String
    /// Working-tree mode: whether this hunk came from the index (`git diff --cached`).
    let staged: Bool
    /// First line numbers on each side, from the @@ header.
    let oldStart: Int
    let newStart: Int
    /// Trailing text after the second @@ (the function context git shows).
    let contextSuffix: String
}

struct FileDiff: Identifiable, Hashable {
    let path: String
    let status: String // M / A / D / R
    let additions: Int
    let deletions: Int
    let hunks: [Hunk]
    let isBinary: Bool

    var id: String { path }
}

/// What the main pane renders: a file plus its (possibly merged staged+unstaged) hunks.
struct DisplayFile: Identifiable, Hashable {
    let path: String
    let status: String
    let additions: Int
    let deletions: Int
    let hunks: [Hunk]
    let isBinary: Bool
    /// Untracked files can only be staged whole, not per-hunk.
    let untracked: Bool

    var id: String { path }
    var name: String { (path as NSString).lastPathComponent }
    var dir: String {
        let d = (path as NSString).deletingLastPathComponent
        return d.isEmpty ? "" : d + "/"
    }
}

/// A row in the file list (left nav).
struct FileEntry: Identifiable, Hashable {
    let path: String
    let status: String
    let add: Int
    let del: Int
    let checked: Bool
    let inStagedSection: Bool
    var commentCount: Int = 0

    var id: String { (inStagedSection ? "s:" : "u:") + path }
    var name: String { (path as NSString).lastPathComponent }
    var dir: String {
        let d = (path as NSString).deletingLastPathComponent
        return d.isEmpty ? "" : d + "/"
    }
}

// MARK: - Comments

/// A review comment as fetched from GitHub (raw, before local curation).
struct RemoteComment: Identifiable, Hashable, Codable {
    let id: Int
    let author: String
    let isBot: Bool
    let body: String
    let path: String?
    let line: Int?
    let side: String
    let createdAt: String
}

/// A comment ready for display (remote with local rewrites applied, or local).
struct ReviewComment: Identifiable, Hashable {
    let id: String
    let author: String
    let isAI: Bool
    let isMine: Bool
    let category: String?
    let body: String
    let path: String?
    let line: Int?
    let side: String
}

struct ComposerState: Equatable {
    var path: String
    var line: Int?
    var text: String = ""
}

// MARK: - Category parsing

enum CategoryParser {
    static let known: Set<String> = [
        "bug", "blocker", "nit", "praise", "question", "consider",
        "suggestion", "style", "issue", "security", "perf", "performance",
    ]

    /// Splits "[Bug] body" or "nit: body" into (category, cleaned body).
    static func split(_ body: String) -> (String?, String) {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("[") , let close = trimmed.firstIndex(of: "]") {
            let cat = String(trimmed[trimmed.index(after: trimmed.startIndex)..<close])
            if known.contains(cat.lowercased()) {
                let rest = String(trimmed[trimmed.index(after: close)...]).trimmingCharacters(in: .whitespaces)
                return (cat.capitalized, rest.isEmpty ? trimmed : rest)
            }
        }
        if let colon = trimmed.firstIndex(of: ":") {
            let head = String(trimmed[..<colon])
            if head.count <= 12, known.contains(head.lowercased()) {
                let rest = String(trimmed[trimmed.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
                return (head.capitalized, rest.isEmpty ? trimmed : rest)
            }
        }
        return (nil, trimmed)
    }
}

// MARK: - Helpers

enum TimeFmt {
    static let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    static func shortAgo(_ date: Date?) -> String {
        guard let date else { return "" }
        let s = max(0, -date.timeIntervalSinceNow)
        if s < 60 { return "now" }
        let m = Int(s / 60)
        if m < 60 { return "\(m)m" }
        let h = m / 60
        if h < 24 { return "\(h)h" }
        let d = h / 24
        if d < 7 { return "\(d)d" }
        let w = d / 7
        if w < 5 { return "\(w)w" }
        let mo = d / 30
        if mo < 12 { return "\(mo)mo" }
        return "\(d / 365)y"
    }
}

func looksLikeBot(_ login: String) -> Bool {
    let l = login.lowercased()
    if l.hasSuffix("[bot]") || l.hasPrefix("app/") { return true }
    for needle in ["bot", "claude", "agent", "copilot", "devin", "codex", "cursor"] {
        if l.contains(needle) { return true }
    }
    return false
}

func initials(_ login: String) -> String {
    let parts = login.split(whereSeparator: { "-_. ".contains($0) })
    if parts.count >= 2, let a = parts[0].first, let b = parts[1].first {
        return String(a).uppercased() + String(b).uppercased()
    }
    return String(login.prefix(2)).uppercased()
}
