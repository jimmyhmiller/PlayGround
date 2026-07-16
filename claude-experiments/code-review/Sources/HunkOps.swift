import Foundation

// Splitting and partial (line-level) patch construction — the equivalents of
// `git add -p`'s "s" (split) and "e" (edit) commands.
extension Hunk {

    /// Indices of the add/del lines; context is never selectable.
    var changeIndices: [Int] {
        lines.indices.filter { lines[$0].kind != .ctx }
    }

    var isSplittable: Bool { splitRanges().count > 1 }

    /// Line ranges of the sub-hunks this hunk would split into.
    ///
    /// Mirrors git's `split_hunk`: a split happens at a context line that
    /// follows one change group and precedes another, and the context between
    /// two groups belongs to *both* sides — it is trailing context for the
    /// hunk before it and leading context for the hunk after it.
    func splitRanges() -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var start = 0
        var i = 0
        var nextStart: Int?
        var seenChange = false

        while i < lines.count {
            if lines[i].kind == .ctx {
                if seenChange && nextStart == nil { nextStart = i }
                i += 1
                continue
            }
            // A change line. If trailing context already began, the previous
            // sub-hunk ends here and the next one starts back at that context.
            if let ns = nextStart {
                ranges.append(start..<i)
                start = ns
                nextStart = nil
                seenChange = false
                i = ns
                continue
            }
            seenChange = true
            i += 1
        }
        ranges.append(start..<lines.count)
        return ranges.filter { !$0.isEmpty }
    }

    /// Line numbers on each side at a given line index.
    func starts(at index: Int) -> (old: Int, new: Int) {
        var o = oldStart
        var n = newStart
        for i in 0..<min(index, lines.count) {
            switch lines[i].kind {
            case .ctx: o += 1; n += 1
            case .del: o += 1
            case .add: n += 1
            }
        }
        return (o, n)
    }

    /// Splits into independently stageable sub-hunks. Returns `[self]` when
    /// there is nothing to split.
    func split() -> [Hunk] {
        let ranges = splitRanges()
        guard ranges.count > 1 else { return [self] }
        return ranges.enumerated().map { k, range in
            let subLines = Array(lines[range])
            let (o, n) = starts(at: range.lowerBound)
            let selected = Set(subLines.indices.filter { subLines[$0].kind != .ctx })
            let built = Self.buildPatch(
                fileHeader: fileHeader, lines: subLines, selected: selected,
                oldStart: o, newStart: n, contextSuffix: contextSuffix, reverse: false
            )
            return Hunk(
                id: "\(id)/s\(k)",
                header: built?.header ?? header,
                lines: subLines,
                fileHeader: fileHeader,
                rawPatch: built?.patch ?? rawPatch,
                staged: staged,
                oldStart: o,
                newStart: n,
                contextSuffix: contextSuffix
            )
        }
    }

    /// A patch containing only the selected add/del lines.
    ///
    /// Forward (staging): unselected adds are dropped (they aren't in the
    /// index yet) and unselected dels become context (the line stays).
    /// Reverse (unstaging): the roles swap — unselected adds become context
    /// because they *are* in the index, and unselected dels are dropped.
    func partialPatch(selecting selected: Set<Int>, reverse: Bool) -> String? {
        let changes = selected.filter { lines.indices.contains($0) && lines[$0].kind != .ctx }
        guard !changes.isEmpty else { return nil }
        return Self.buildPatch(
            fileHeader: fileHeader, lines: lines, selected: Set(changes),
            oldStart: oldStart, newStart: newStart, contextSuffix: contextSuffix,
            reverse: reverse
        )?.patch
    }

    fileprivate static func buildPatch(
        fileHeader: String,
        lines: [DiffLine],
        selected: Set<Int>,
        oldStart: Int,
        newStart: Int,
        contextSuffix: String,
        reverse: Bool
    ) -> (header: String, patch: String)? {
        var body: [String] = []
        var oldCount = 0
        var newCount = 0
        var sawChange = false

        func emit(_ prefix: String, _ line: DiffLine) {
            body.append(prefix + line.text)
            if line.noNewline { body.append("\\ No newline at end of file") }
        }

        for (idx, line) in lines.enumerated() {
            let picked = selected.contains(idx)
            switch line.kind {
            case .ctx:
                emit(" ", line); oldCount += 1; newCount += 1
            case .del:
                if picked {
                    emit("-", line); oldCount += 1; sawChange = true
                } else if reverse {
                    continue // not in the index; must not be restored
                } else {
                    emit(" ", line); oldCount += 1; newCount += 1
                }
            case .add:
                if picked {
                    emit("+", line); newCount += 1; sawChange = true
                } else if reverse {
                    emit(" ", line); oldCount += 1; newCount += 1
                } else {
                    continue // not in the index yet; leave it out
                }
            }
        }
        guard sawChange else { return nil }

        // A zero-length side anchors at the line before the change, which is
        // what git emits for pure insertions/deletions.
        let oldAnchor = oldCount == 0 ? max(0, oldStart - 1) : oldStart
        let newAnchor = newCount == 0 ? max(0, newStart - 1) : newStart
        let suffix = contextSuffix.isEmpty ? "" : " " + contextSuffix
        let header = "@@ -\(oldAnchor),\(oldCount) +\(newAnchor),\(newCount) @@\(suffix)"
        let patch = fileHeader + header + "\n" + body.joined(separator: "\n") + "\n"
        return (header, patch)
    }
}
