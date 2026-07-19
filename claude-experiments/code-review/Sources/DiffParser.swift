import Foundation

enum DiffParser {
    /// Parses unified diff output (git diff / gh pr diff) into FileDiffs.
    /// `staged` marks every produced hunk as coming from the index.
    static func parse(_ text: String, staged: Bool = false) -> [FileDiff] {
        var files: [FileDiff] = []
        let lines = text.components(separatedBy: "\n")
        var i = 0

        while i < lines.count {
            guard lines[i].hasPrefix("diff --git ") else { i += 1; continue }

            var headerLines = [lines[i]]
            var status = "M"
            var pathA: String?
            var pathB: String?
            var isBinary = false
            i += 1

            while i < lines.count, !lines[i].hasPrefix("@@"), !lines[i].hasPrefix("diff --git ") {
                let l = lines[i]
                headerLines.append(l)
                if l.hasPrefix("new file mode") { status = "A" }
                else if l.hasPrefix("deleted file mode") { status = "D" }
                else if l.hasPrefix("rename from") || l.hasPrefix("similarity index") { status = "R" }
                else if l.hasPrefix("Binary files ") || l.hasPrefix("GIT binary patch") { isBinary = true }
                else if l.hasPrefix("--- ") { pathA = stripDiffPath(String(l.dropFirst(4))) }
                else if l.hasPrefix("+++ ") { pathB = stripDiffPath(String(l.dropFirst(4))) }
                i += 1
            }

            let path = pathB ?? pathA ?? pathFromGitLine(headerLines[0]) ?? "unknown"
            let fileHeader = headerLines.joined(separator: "\n") + "\n"

            var hunks: [Hunk] = []
            var additions = 0
            var deletions = 0

            while i < lines.count, lines[i].hasPrefix("@@") {
                let header = lines[i]
                var raw = [header]
                var (o, n) = parseHunkHeader(header)
                let hunkOldStart = o
                let hunkNewStart = n
                var dls: [DiffLine] = []
                i += 1

                loop: while i < lines.count {
                    let l = lines[i]
                    guard let first = l.first else { break loop } // blank line = end of diff block
                    switch first {
                    case "\\":
                        raw.append(l)
                        // Belongs to the line just emitted; rebuilt patches
                        // must carry it or they'd add a trailing newline.
                        if !dls.isEmpty { dls[dls.count - 1].noNewline = true }
                    case "+":
                        raw.append(l)
                        dls.append(DiffLine(kind: .add, oldNo: nil, newNo: n, text: String(l.dropFirst())))
                        n += 1; additions += 1
                    case "-":
                        raw.append(l)
                        dls.append(DiffLine(kind: .del, oldNo: o, newNo: nil, text: String(l.dropFirst())))
                        o += 1; deletions += 1
                    case " ":
                        raw.append(l)
                        dls.append(DiffLine(kind: .ctx, oldNo: o, newNo: n, text: String(l.dropFirst())))
                        o += 1; n += 1
                    default:
                        break loop
                    }
                    i += 1
                }

                let rawPatch = fileHeader + raw.joined(separator: "\n") + "\n"
                hunks.append(Hunk(
                    id: "\(path)#\(hunks.count)#\(staged ? "s" : "u")",
                    header: header,
                    lines: dls,
                    fileHeader: fileHeader,
                    rawPatch: rawPatch,
                    staged: staged,
                    oldStart: hunkOldStart,
                    newStart: hunkNewStart,
                    contextSuffix: hunkContextSuffix(header)
                ))
            }

            files.append(FileDiff(
                path: path,
                status: status,
                additions: additions,
                deletions: deletions,
                hunks: hunks,
                isBinary: isBinary
            ))
        }
        return files
    }

    /// Builds a synthetic all-added diff for an untracked file (read straight from disk).
    static func untrackedFile(repo: URL, path: String) -> DisplayFile {
        let url = repo.appendingPathComponent(path)
        guard let data = try? Data(contentsOf: url) else {
            return DisplayFile(path: path, status: "A", additions: 0, deletions: 0,
                               hunks: [], isBinary: false, untracked: true)
        }
        if data.contains(0) || data.count > 2_000_000 {
            return DisplayFile(path: path, status: "A", additions: 0, deletions: 0,
                               hunks: [], isBinary: true, untracked: true)
        }
        let text = String(data: data, encoding: .utf8) ?? ""
        var rawLines = text.components(separatedBy: "\n")
        if rawLines.last == "" { rawLines.removeLast() }
        let dls = rawLines.enumerated().map { idx, t in
            DiffLine(kind: .add, oldNo: nil, newNo: idx + 1, text: t)
        }
        let hunk = Hunk(
            id: "\(path)#0#untracked",
            header: "@@ -0,0 +1,\(dls.count) @@",
            lines: dls,
            fileHeader: "",
            rawPatch: "",
            staged: false,
            oldStart: 0,
            newStart: 1,
            contextSuffix: ""
        )
        return DisplayFile(path: path, status: "A", additions: dls.count, deletions: 0,
                           hunks: dls.isEmpty ? [] : [hunk], isBinary: false, untracked: true)
    }

    // MARK: internals

    private static func stripDiffPath(_ s: String) -> String? {
        var p = s
        if let tab = p.firstIndex(of: "\t") { p = String(p[..<tab]) }
        if p == "/dev/null" { return nil }
        if p.hasPrefix("a/") || p.hasPrefix("b/") { p = String(p.dropFirst(2)) }
        if p.hasPrefix("\"") && p.hasSuffix("\"") && p.count >= 2 {
            p = String(p.dropFirst().dropLast())
        }
        return p
    }

    private static func pathFromGitLine(_ line: String) -> String? {
        // diff --git a/foo b/foo
        guard let range = line.range(of: " b/") else { return nil }
        return String(line[range.upperBound...])
    }

    /// The function-context text git prints after the closing @@.
    private static func hunkContextSuffix(_ header: String) -> String {
        // @@ -14,9 +14,22 @@ createSession(user)
        let parts = header.components(separatedBy: "@@")
        guard parts.count >= 3 else { return "" }
        return parts[2...].joined(separator: "@@").trimmingCharacters(in: .whitespaces)
    }

    private static func parseHunkHeader(_ header: String) -> (Int, Int) {
        // @@ -14,9 +14,22 @@ optional context
        var old = 1
        var new = 1
        let parts = header.split(separator: " ")
        for part in parts {
            if part.hasPrefix("-") {
                let nums = part.dropFirst().split(separator: ",")
                if let f = nums.first, let v = Int(f) { old = v }
            } else if part.hasPrefix("+") {
                let nums = part.dropFirst().split(separator: ",")
                if let f = nums.first, let v = Int(f) { new = v }
            }
        }
        return (old, new)
    }
}
