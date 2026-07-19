import XCTest
@testable import AgentReview

/// These tests run against real git in a throwaway repo. The patches this app
/// builds are applied to a real index, because "the patch looked right" is not
/// evidence — `git apply` accepting it and producing the expected index is.
final class HunkPatchTests: XCTestCase {

    // MARK: - Test repo helpers

    private var repo: URL!

    override func setUpWithError() throws {
        repo = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("agentreview-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: repo, withIntermediateDirectories: true)
        try run("git", "init", "-q")
        try run("git", "config", "user.email", "test@example.com")
        try run("git", "config", "user.name", "Test")
        try run("git", "config", "commit.gpgsign", "false")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: repo)
    }

    @discardableResult
    private func run(_ args: String...) throws -> String {
        try runArgs(args, stdin: nil)
    }

    @discardableResult
    private func runArgs(_ args: [String], stdin: String?) throws -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = args
        p.currentDirectoryURL = repo
        var env = ProcessInfo.processInfo.environment
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["HOME"] = repo.path
        p.environment = env
        let out = Pipe(), err = Pipe()
        p.standardOutput = out
        p.standardError = err
        if let stdin {
            let inPipe = Pipe()
            p.standardInput = inPipe
            try p.run()
            inPipe.fileHandleForWriting.write(Data(stdin.utf8))
            inPipe.fileHandleForWriting.closeFile()
        } else {
            p.standardInput = FileHandle.nullDevice
            try p.run()
        }
        let o = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let e = String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        p.waitUntilExit()
        if p.terminationStatus != 0 {
            throw NSError(domain: "git", code: Int(p.terminationStatus),
                          userInfo: [NSLocalizedDescriptionKey: "\(args.joined(separator: " ")) failed: \(e)\(o)"])
        }
        return o
    }

    private func write(_ name: String, _ content: String) throws {
        try content.write(to: repo.appendingPathComponent(name), atomically: true, encoding: .utf8)
    }

    private func commitAll() throws {
        try run("git", "add", "-A")
        try run("git", "commit", "-qm", "base")
    }

    /// The single hunk of the current unstaged diff.
    private func unstagedHunk(file: String = "f.txt") throws -> Hunk {
        let diff = try run("git", "diff", "--no-ext-diff", "--", file)
        let files = DiffParser.parse(diff, staged: false)
        let hunks = files.flatMap(\.hunks)
        guard let first = hunks.first else {
            throw NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "no hunk in diff:\n\(diff)"])
        }
        return first
    }

    private func applyCached(_ patch: String, reverse: Bool = false) throws {
        var args = ["git", "apply", "--cached", "--whitespace=nowarn"]
        if reverse { args.append("-R") }
        args.append("-")
        try runArgs(args, stdin: patch)
    }

    private func stagedContent(_ file: String = "f.txt") throws -> String {
        try run("git", "show", ":\(file)")
    }

    // MARK: - Splitting

    /// Two change groups separated by context must split into two hunks, and
    /// staging only the first must leave the second unstaged.
    func testSplitSeparatesChangeGroupsAndStagesIndependently() throws {
        let base = (1...20).map { "line \($0)" }.joined(separator: "\n") + "\n"
        try write("f.txt", base)
        try commitAll()

        var lines = base.components(separatedBy: "\n")
        lines[2] = "line 3 CHANGED"    // group 1
        lines[15] = "line 16 CHANGED"  // group 2, far away
        try write("f.txt", lines.joined(separator: "\n"))

        let hunks = try DiffParser.parse(run("git", "diff", "--no-ext-diff")).flatMap(\.hunks)
        // Far-apart edits already produce two hunks; force one hunk with big context.
        let wide = try run("git", "diff", "--no-ext-diff", "-U10")
        let oneHunk = DiffParser.parse(wide).flatMap(\.hunks)
        XCTAssertEqual(oneHunk.count, 1, "expected a single hunk with -U10; got \(oneHunk.count)")
        _ = hunks

        let hunk = oneHunk[0]
        XCTAssertTrue(hunk.isSplittable, "a hunk with two separated change groups must be splittable")

        let subs = hunk.split()
        XCTAssertEqual(subs.count, 2, "expected exactly two sub-hunks")

        // Each sub-hunk must apply on its own.
        try applyCached(subs[0].rawPatch)
        let staged = try stagedContent()
        XCTAssertTrue(staged.contains("line 3 CHANGED"), "first sub-hunk should be staged")
        XCTAssertFalse(staged.contains("line 16 CHANGED"), "second sub-hunk must NOT be staged")

        // And the second still applies afterwards.
        let remaining = try DiffParser.parse(run("git", "diff", "--no-ext-diff", "-U10")).flatMap(\.hunks)
        XCTAssertEqual(remaining.count, 1)
        try applyCached(remaining[0].rawPatch)
        XCTAssertTrue(try stagedContent().contains("line 16 CHANGED"))
    }

    func testSingleChangeGroupIsNotSplittable() throws {
        let base = (1...10).map { "line \($0)" }.joined(separator: "\n") + "\n"
        try write("f.txt", base)
        try commitAll()
        var lines = base.components(separatedBy: "\n")
        lines[4] = "only change"
        try write("f.txt", lines.joined(separator: "\n"))

        let hunk = try unstagedHunk()
        XCTAssertFalse(hunk.isSplittable)
        XCTAssertEqual(hunk.split().count, 1)
    }

    /// Every sub-hunk of a split must be applicable by git, in any order.
    func testAllSplitSubHunksApplyIndividually() throws {
        let base = (1...40).map { "line \($0)" }.joined(separator: "\n") + "\n"
        try write("f.txt", base)
        try commitAll()
        var lines = base.components(separatedBy: "\n")
        lines[3] = "A changed"
        lines[13] = "B changed"
        lines[23] = "C changed"
        try write("f.txt", lines.joined(separator: "\n"))

        let hunk = try DiffParser.parse(run("git", "diff", "--no-ext-diff", "-U8")).flatMap(\.hunks).first!
        let subs = hunk.split()
        XCTAssertEqual(subs.count, 3, "three separated edits should split three ways")

        // Apply the middle one only — the trickiest case for line numbers.
        try applyCached(subs[1].rawPatch)
        let staged = try stagedContent()
        XCTAssertFalse(staged.contains("A changed"))
        XCTAssertTrue(staged.contains("B changed"))
        XCTAssertFalse(staged.contains("C changed"))
    }

    // MARK: - Partial (line-level) staging

    /// Selecting one added line out of several must stage exactly that line.
    func testPartialPatchStagesOnlySelectedAddedLine() throws {
        try write("f.txt", "a\nb\nc\n")
        try commitAll()
        try write("f.txt", "a\nNEW1\nNEW2\nb\nc\n")

        let hunk = try unstagedHunk()
        let addIdx = hunk.lines.indices.filter { hunk.lines[$0].kind == .add }
        XCTAssertEqual(addIdx.count, 2)

        // Stage only NEW1.
        let first = hunk.lines.firstIndex { $0.kind == .add && $0.text == "NEW1" }!
        let patch = try XCTUnwrap(hunk.partialPatch(selecting: [first], reverse: false))
        try applyCached(patch)

        XCTAssertEqual(try stagedContent(), "a\nNEW1\nb\nc\n",
                       "only the selected added line should be in the index")
    }

    /// Unselected deletions must survive as context, not vanish.
    func testPartialPatchStagesOnlySelectedDeletion() throws {
        try write("f.txt", "keep1\ndrop1\ndrop2\nkeep2\n")
        try commitAll()
        try write("f.txt", "keep1\nkeep2\n")

        let hunk = try unstagedHunk()
        let dropOne = hunk.lines.firstIndex { $0.kind == .del && $0.text == "drop1" }!
        let patch = try XCTUnwrap(hunk.partialPatch(selecting: [dropOne], reverse: false))
        try applyCached(patch)

        XCTAssertEqual(try stagedContent(), "keep1\ndrop2\nkeep2\n",
                       "unselected deletion must remain in the index as context")
    }

    /// Interleaved del/add groups: selecting one deletion and one addition.
    ///
    /// The expected index is "a, old2, new1, z" — NOT "a, new1, old2, z". A
    /// unified diff carries no pairing between `-old1` and `+new1`, so the
    /// unselected deletion becomes context and stays where the diff put it,
    /// ahead of the additions. Verified against real `git add -p` + `e` with
    /// this exact edit, which produces the same index byte for byte.
    func testPartialPatchMixedAddAndDeleteMatchesGitSemantics() throws {
        try write("f.txt", "a\nold1\nold2\nz\n")
        try commitAll()
        try write("f.txt", "a\nnew1\nnew2\nz\n")

        let hunk = try unstagedHunk()
        let delOld1 = hunk.lines.firstIndex { $0.kind == .del && $0.text == "old1" }!
        let addNew1 = hunk.lines.firstIndex { $0.kind == .add && $0.text == "new1" }!
        let patch = try XCTUnwrap(hunk.partialPatch(selecting: [delOld1, addNew1], reverse: false))
        try applyCached(patch)

        XCTAssertEqual(try stagedContent(), "a\nold2\nnew1\nz\n",
                       "old1 removed and new1 added; unselected old2 survives as context")
    }

    func testPartialPatchWithNoChangesSelectedReturnsNil() throws {
        try write("f.txt", "a\nb\n")
        try commitAll()
        try write("f.txt", "a\nCHANGED\n")
        let hunk = try unstagedHunk()
        let ctxIdx = hunk.lines.firstIndex { $0.kind == .ctx }
        XCTAssertNil(hunk.partialPatch(selecting: [], reverse: false))
        if let ctxIdx {
            XCTAssertNil(hunk.partialPatch(selecting: [ctxIdx], reverse: false),
                         "selecting only context is not a change")
        }
    }

    /// Reverse (unstage) selective patch: unselected adds become context.
    func testPartialUnstageOfSelectedLine() throws {
        try write("f.txt", "a\nz\n")
        try commitAll()
        try write("f.txt", "a\nS1\nS2\nz\n")
        try run("git", "add", "f.txt")   // both lines staged

        let cached = try run("git", "diff", "--cached", "--no-ext-diff")
        let hunk = DiffParser.parse(cached, staged: true).flatMap(\.hunks).first!
        let s1 = hunk.lines.firstIndex { $0.kind == .add && $0.text == "S1" }!

        // Unstage only S1; S2 must stay staged.
        let patch = try XCTUnwrap(hunk.partialPatch(selecting: [s1], reverse: true))
        try applyCached(patch, reverse: true)

        XCTAssertEqual(try stagedContent(), "a\nS2\nz\n",
                       "only S1 should have been unstaged")
    }

    // MARK: - No-newline handling

    /// A file with no trailing newline must not silently gain one.
    func testNoNewlineMarkerSurvivesRebuiltPatch() throws {
        try write("f.txt", "alpha\nbeta")      // no trailing newline
        try commitAll()
        try write("f.txt", "alpha\nbetaX")     // still no trailing newline

        let hunk = try unstagedHunk()
        XCTAssertTrue(hunk.lines.contains { $0.noNewline },
                      "parser must record the no-newline marker")

        let changes = Set(hunk.changeIndices)
        let patch = try XCTUnwrap(hunk.partialPatch(selecting: changes, reverse: false))
        XCTAssertTrue(patch.contains("\\ No newline at end of file"),
                      "rebuilt patch must carry the marker")
        try applyCached(patch)

        XCTAssertEqual(try stagedContent(), "alpha\nbetaX",
                       "staged file must still lack a trailing newline")
    }

    // MARK: - Header math

    /// The rebuilt full-selection patch must be equivalent to git's own.
    func testFullSelectionPatchMatchesGitApplyResult() throws {
        let base = (1...12).map { "l\($0)" }.joined(separator: "\n") + "\n"
        try write("f.txt", base)
        try commitAll()
        var lines = base.components(separatedBy: "\n")
        lines[5] = "l6 modified"
        lines.insert("inserted", at: 8)
        try write("f.txt", lines.joined(separator: "\n"))

        let hunk = try DiffParser.parse(run("git", "diff", "--no-ext-diff", "-U10")).flatMap(\.hunks).first!
        let rebuilt = try XCTUnwrap(hunk.partialPatch(selecting: Set(hunk.changeIndices), reverse: false))
        try applyCached(rebuilt)

        let expected = lines.joined(separator: "\n")
        XCTAssertEqual(try stagedContent(), expected,
                       "selecting every change must equal staging the whole hunk")
    }
}
