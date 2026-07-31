import Foundation

struct StatusEntry {
    let path: String
    let index: Character    // staged status
    let worktree: Character // unstaged status
}

struct WorkingTreeState {
    var unstagedDiffs: [FileDiff] = []
    var stagedDiffs: [FileDiff] = []
    var branchDiffs: [FileDiff] = []
    var untracked: [String] = []
    var entries: [StatusEntry] = []
}

struct WorktreeEntry {
    let path: URL
    let branch: String?   // nil when detached
    let prunable: Bool    // git can no longer reach the checkout
}

enum GitClient {
    // MARK: - Project discovery

    /// Finds every repository under `root` and folds each one's linked
    /// worktrees in underneath it.
    ///
    /// Two things the filesystem walk alone cannot do, and that git is asked
    /// about instead:
    ///  - worktrees are routinely kept *inside* the repo (`.claude/worktrees/…`),
    ///    below a hidden directory and past the point where the walk stops
    ///    descending, so they are invisible to a scan;
    ///  - worktrees kept *beside* the repo (`next.js-wt-foo`) look exactly like
    ///    independent clones to a scan, and would otherwise each show up as
    ///    another copy of the same project.
    static func discoverProjects(root: URL) async -> [Project] {
        let candidates = await scanForRepos(root: root)

        // `git worktree list` always reports the main worktree first, which is
        // what tells a linked worktree apart from a clone of its own.
        var lists: [String: [WorktreeEntry]] = [:]
        await withTaskGroup(of: (String, [WorktreeEntry]).self) { group in
            for c in candidates {
                group.addTask { (c.id, await worktrees(repo: c.path)) }
            }
            for await (id, list) in group { lists[id] = list }
        }

        // Group the candidates by the repo they belong to. A repo whose main
        // worktree sits outside `root` still earns an entry, so its worktrees
        // under `root` are not silently dropped.
        var groups: [String: (main: URL, list: [WorktreeEntry])] = [:]
        var order: [String] = []
        for c in candidates {
            let list = lists[c.id] ?? []
            let main = list.first?.path ?? c.path
            let key = canonical(main)
            if groups[key] == nil {
                groups[key] = (main, list)
                order.append(key)
            }
        }

        var projects = order.compactMap { key -> Project? in
            guard let group = groups[key] else { return nil }
            let links = group.list
                .dropFirst()
                .filter { !$0.prunable && isLiveWorktree($0.path) }
                .map { Project(name: $0.path.lastPathComponent, path: $0.path,
                               modified: gitModified($0.path), branch: $0.branch) }
                .sorted { $0.modified > $1.modified }
            return Project(name: group.main.lastPathComponent, path: group.main,
                           modified: gitModified(group.main),
                           branch: group.list.first?.branch, worktrees: links)
        }
        projects.sort { $0.modified > $1.modified }
        return disambiguated(projects)
    }

    /// `git worktree list --porcelain`, main worktree first.
    static func worktrees(repo: URL) async -> [WorktreeEntry] {
        let r = await Shell.run(readOnly(["worktree", "list", "--porcelain"]), cwd: repo)
        guard r.ok else { return [] }

        var entries: [WorktreeEntry] = []
        var path: URL?
        var branch: String?
        var prunable = false

        func flush() {
            if let path { entries.append(WorktreeEntry(path: path, branch: branch, prunable: prunable)) }
            path = nil
            branch = nil
            prunable = false
        }

        for line in r.stdout.components(separatedBy: "\n") {
            if let value = line.dropPrefix("worktree ") {
                flush()
                path = URL(fileURLWithPath: value)
            } else if let value = line.dropPrefix("branch ") {
                branch = value.dropPrefix("refs/heads/") ?? value
            } else if line.hasPrefix("prunable") {
                prunable = true
            }
        }
        flush()
        return entries
    }

    // MARK: - Discovery helpers

    private struct RepoCandidate {
        let path: URL
        var id: String { path.path }
    }

    /// Filesystem walk for repositories, up to two directory levels deep. Stops
    /// descending once a repo is found — anything nested inside comes from
    /// `git worktree list` instead, which is both cheaper and complete.
    private static func scanForRepos(root: URL) async -> [RepoCandidate] {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let fm = FileManager.default
                var found: [RepoCandidate] = []
                let skip: Set<String> = ["node_modules", "target", "dist", "build", ".build", "vendor"]

                func scan(_ dir: URL, depth: Int) {
                    // A worktree's ".git" is a file, not a directory; both count.
                    if fm.fileExists(atPath: dir.appendingPathComponent(".git").path) {
                        found.append(RepoCandidate(path: dir))
                        return
                    }
                    guard depth > 0 else { return }
                    let children = (try? fm.contentsOfDirectory(
                        at: dir,
                        includingPropertiesForKeys: [.isDirectoryKey],
                        options: [.skipsHiddenFiles]
                    )) ?? []
                    for child in children {
                        guard (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
                        guard !skip.contains(child.lastPathComponent) else { continue }
                        scan(child, depth: depth - 1)
                    }
                }

                let children = (try? fm.contentsOfDirectory(
                    at: root,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles]
                )) ?? []
                for child in children {
                    guard (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
                    scan(child, depth: 2)
                }
                cont.resume(returning: found)
            }
        }
    }

    /// git prints fully resolved paths (`/private/tmp`, not `/tmp`), so both
    /// sides get resolved before being compared.
    private static func canonical(_ url: URL) -> String {
        url.resolvingSymlinksInPath().standardizedFileURL.path
    }

    private static func gitModified(_ dir: URL) -> Date {
        let attrs = try? FileManager.default.attributesOfItem(
            atPath: dir.appendingPathComponent(".git").path
        )
        return (attrs?[.modificationDate] as? Date) ?? .distantPast
    }

    /// Throwaway worktrees under a temp directory are build/benchmark scratch,
    /// not something anyone reviews, so they stay out of the sidebar.
    private static func isLiveWorktree(_ path: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: path.path) else { return false }
        let resolved = canonical(path)
        let tempRoots = ["/tmp/", "/private/tmp/", "/var/folders/", "/private/var/folders/",
                         canonical(URL(fileURLWithPath: NSTemporaryDirectory())) + "/"]
        return !tempRoots.contains { resolved.hasPrefix($0) }
    }

    /// Distinct clones can share a directory name — `personal/next.js` and
    /// `vercel/next.js` both read as "next.js". Qualify the colliding ones with
    /// as many parent directories as it takes to tell them apart, so the
    /// sidebar never shows the same label twice.
    private static func disambiguated(_ projects: [Project]) -> [Project] {
        var result = projects
        for depth in 1..<6 {
            var counts: [String: Int] = [:]
            for p in result { counts[p.name, default: 0] += 1 }
            let collisions = Set(counts.filter { $0.value > 1 }.keys)
            if collisions.isEmpty { break }

            var changed = false
            for i in result.indices where collisions.contains(result[i].name) {
                let components = result[i].path.pathComponents.filter { $0 != "/" }
                guard components.count > depth else { continue }
                result[i].name = components.suffix(depth + 1).joined(separator: "/")
                changed = true
            }
            if !changed { break }
        }
        return result
    }

    /// Read-only git invocation. `--no-optional-locks` keeps this app's
    /// background polling from ever taking .git/index.lock, which would
    /// collide with the user's own commits in the same repo.
    private static func readOnly(_ args: [String]) -> [String] {
        ["git", "--no-optional-locks"] + args
    }

    /// "owner/repo" if origin points at GitHub.
    static func remoteSlug(repo: URL) async -> String? {
        let r = await Shell.run(readOnly(["remote", "get-url", "origin"]), cwd: repo)
        guard r.ok else { return nil }
        let url = r.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let range = url.range(of: "github.com") else { return nil }
        var rest = String(url[range.upperBound...])
        if rest.hasPrefix(":") || rest.hasPrefix("/") { rest = String(rest.dropFirst()) }
        if rest.hasSuffix(".git") { rest = String(rest.dropLast(4)) }
        let parts = rest.split(separator: "/")
        guard parts.count >= 2 else { return nil }
        return "\(parts[0])/\(parts[1])"
    }

    // MARK: - Working tree

    static func status(repo: URL) async -> [StatusEntry] {
        let r = await Shell.run(readOnly(["status", "--porcelain"]), cwd: repo)
        guard r.ok else { return [] }
        var entries: [StatusEntry] = []
        for line in r.stdout.components(separatedBy: "\n") where line.count > 3 {
            let x = line[line.startIndex]
            let y = line[line.index(after: line.startIndex)]
            var path = String(line.dropFirst(3))
            if let arrow = path.range(of: " -> ") {
                path = String(path[arrow.upperBound...])
            }
            if path.hasPrefix("\"") && path.hasSuffix("\"") && path.count >= 2 {
                path = String(path.dropFirst().dropLast())
            }
            entries.append(StatusEntry(path: path, index: x, worktree: y))
        }
        return entries
    }

    static func workingTree(repo: URL) async -> WorkingTreeState {
        async let statusEntries = status(repo: repo)
        async let unstagedRun = Shell.run(readOnly(["diff", "--no-ext-diff"]), cwd: repo)
        async let stagedRun = Shell.run(readOnly(["diff", "--cached", "--no-ext-diff"]), cwd: repo)

        var state = WorkingTreeState()
        state.entries = await statusEntries
        state.unstagedDiffs = DiffParser.parse(await unstagedRun.stdout, staged: false)
        state.stagedDiffs = DiffParser.parse(await stagedRun.stdout, staged: true)
        state.untracked = state.entries.filter { $0.index == "?" && $0.worktree == "?" }.map(\.path)

        // Also get the full branch diff vs the base branch so the review pane shows
        // everything the branch introduced, not just pending changes.
        if let baseRef = await baseBranchRef(repo: repo) {
            if let mb = await mergeBase(repo: repo, base: baseRef) {
                let branchRun = await Shell.run(readOnly(["diff", "--no-ext-diff", mb, "HEAD"]), cwd: repo)
                state.branchDiffs = DiffParser.parse(branchRun.stdout, staged: false)
            }
        }

        return state
    }

    /// Finds the base branch ref. Tries the common names first, then falls back to
    /// `origin/HEAD` (the remote's default branch) for repos that use `canary`,
    /// `develop`, or another non-main convention.
    private static func baseBranchRef(repo: URL) async -> String? {
        for candidate in ["origin/main", "origin/master", "main", "master"] {
            let r = await Shell.run(readOnly(["rev-parse", "--verify", candidate]), cwd: repo)
            if r.ok { return candidate }
        }
        // origin/HEAD resolves to whatever the remote considers its default —
        // covers canary, develop, trunk, etc.
        let r = await Shell.run(readOnly(["rev-parse", "--verify", "origin/HEAD"]), cwd: repo)
        if r.ok { return "origin/HEAD" }
        return nil
    }

    /// Returns the merge-base commit between `base` and HEAD.
    private static func mergeBase(repo: URL, base: String) async -> String? {
        let r = await Shell.run(readOnly(["merge-base", base, "HEAD"]), cwd: repo)
        guard r.ok else { return nil }
        let mb = r.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return mb.isEmpty ? nil : mb
    }

    // MARK: - Branch comparison

    /// Figures out whether `main` or `master` is the base branch of the
    /// current branch, then counts commits and files changed vs `merge-base`.
    static func branchComparison(repo: URL) async -> BranchComparison? {
        guard await currentBranch(repo: repo) != nil else { return nil }
        guard let base = await baseBranchRef(repo: repo) else { return nil }
        guard let mb = await mergeBase(repo: repo, base: base) else { return nil }

        let display: String
        if base == "origin/HEAD" {
            // Resolve origin/HEAD to the actual branch name like "canary".
            let sym = await Shell.run(readOnly(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"]), cwd: repo)
            var name = sym.ok ? sym.stdout.trimmingCharacters(in: .whitespacesAndNewlines) : "HEAD"
            if name.hasPrefix("origin/") { name = String(name.dropFirst(7)) }
            display = name
        } else {
            display = base.hasPrefix("origin/") ? String(base.dropFirst(7)) : base
        }

        async let aheadRun = Shell.run(readOnly(["rev-list", "--count", "\(base)..HEAD"]), cwd: repo)
        async let behindRun = Shell.run(readOnly(["rev-list", "--count", "HEAD..\(base)"]), cwd: repo)
        async let filesRun = Shell.run(readOnly(["diff", "--name-only", mb, "HEAD"]), cwd: repo)

        let ahead = Int((await aheadRun).stdout.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
        let behind = Int((await behindRun).stdout.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
        let files = (await filesRun).ok
            ? (await filesRun).stdout.components(separatedBy: "\n").filter { !$0.isEmpty }.count
            : 0

        return BranchComparison(baseBranch: display, ahead: ahead, behind: behind, filesChanged: files)
    }

    /// Returns the current branch name, or nil when detached.
    private static func currentBranch(repo: URL) async -> String? {
        let r = await Shell.run(readOnly(["rev-parse", "--abbrev-ref", "HEAD"]), cwd: repo)
        guard r.ok else { return nil }
        let name = r.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return name == "HEAD" ? nil : name
    }

    // MARK: - Staging

    static func stageFile(repo: URL, path: String) async -> ShellResult {
        await Shell.run(["git", "add", "--", path], cwd: repo)
    }

    static func unstageFile(repo: URL, path: String) async -> ShellResult {
        await Shell.run(["git", "reset", "-q", "HEAD", "--", path], cwd: repo)
    }

    static func stageHunk(repo: URL, patch: String) async -> ShellResult {
        await Shell.run(["git", "apply", "--cached", "--whitespace=nowarn", "-"], cwd: repo, stdin: patch)
    }

    static func unstageHunk(repo: URL, patch: String) async -> ShellResult {
        await Shell.run(["git", "apply", "--cached", "-R", "--whitespace=nowarn", "-"], cwd: repo, stdin: patch)
    }
}

private extension String {
    /// The remainder after `prefix`, or nil when the string does not start with it.
    func dropPrefix(_ prefix: String) -> String? {
        hasPrefix(prefix) ? String(dropFirst(prefix.count)) : nil
    }
}
