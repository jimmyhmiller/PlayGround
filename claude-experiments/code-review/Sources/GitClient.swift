import Foundation

struct StatusEntry {
    let path: String
    let index: Character    // staged status
    let worktree: Character // unstaged status
}

struct WorkingTreeState {
    var unstagedDiffs: [FileDiff] = []
    var stagedDiffs: [FileDiff] = []
    var untracked: [String] = []
    var entries: [StatusEntry] = []
}

enum GitClient {
    // MARK: - Project discovery

    /// Finds git repositories under `root`, up to two directory levels deep.
    /// Does not descend into a repo once found.
    static func discoverProjects(root: URL) async -> [Project] {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let fm = FileManager.default
                var found: [Project] = []
                let skip: Set<String> = ["node_modules", "target", "dist", "build", ".build", "vendor"]

                func scan(_ dir: URL, depth: Int) {
                    let gitDir = dir.appendingPathComponent(".git")
                    if fm.fileExists(atPath: gitDir.path) {
                        let attrs = try? fm.attributesOfItem(atPath: gitDir.path)
                        let modified = (attrs?[.modificationDate] as? Date) ?? .distantPast
                        found.append(Project(name: dir.lastPathComponent, path: dir, modified: modified))
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
                found.sort { $0.modified > $1.modified }
                cont.resume(returning: found)
            }
        }
    }

    /// "owner/repo" if origin points at GitHub.
    static func remoteSlug(repo: URL) async -> String? {
        let r = await Shell.run(["git", "remote", "get-url", "origin"], cwd: repo)
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
        let r = await Shell.run(["git", "status", "--porcelain"], cwd: repo)
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
        async let unstagedRun = Shell.run(["git", "diff", "--no-ext-diff"], cwd: repo)
        async let stagedRun = Shell.run(["git", "diff", "--cached", "--no-ext-diff"], cwd: repo)

        var state = WorkingTreeState()
        state.entries = await statusEntries
        state.unstagedDiffs = DiffParser.parse(await unstagedRun.stdout, staged: false)
        state.stagedDiffs = DiffParser.parse(await stagedRun.stdout, staged: true)
        state.untracked = state.entries.filter { $0.index == "?" && $0.worktree == "?" }.map(\.path)
        return state
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
