import SwiftUI
import AppKit

// MARK: - Diff row models (what the main pane renders)

struct ThreadModel: Identifiable {
    let id: String
    let path: String
    let line: Int?
    let comments: [ReviewComment]
    let showComposer: Bool
}

enum DiffRowModel: Identifiable {
    case hunk(Hunk)
    case uline(String, DiffLine)
    case pair(String, DiffLine?, DiffLine?)
    case thread(ThreadModel)
    case note(String, String)

    var id: String {
        switch self {
        case .hunk(let h): return "h:" + h.id
        case .uline(let id, _): return id
        case .pair(let id, _, _): return id
        case .thread(let t): return t.id
        case .note(let id, _): return id
        }
    }
}

// MARK: - Store

@MainActor
final class AppStore: ObservableObject {
    static let codeRoot = FileManager.default
        .homeDirectoryForCurrentUser
        .appendingPathComponent("Documents/Code")

    // Sidebar
    @Published var projects: [Project] = []
    @Published var expanded: Set<String> = []
    @Published var prsByProject: [String: [PullRequest]] = [:]
    @Published var prErrors: [String: String] = [:]
    @Published var loadingPRs: Set<String> = []
    @Published var wtCounts: [String: Int] = [:]
    @Published var projectFilter: String = ""
    @Published var needsAuth: Set<String> = []

    // Selection & screens
    @Published var selection: Selection = .none
    @Published var screen: Screen = .empty
    @Published var mode: DiffMode = .unified
    @Published var verdict: Verdict = .comment

    // Current review content
    @Published var displayFiles: [DisplayFile] = []
    @Published var changeEntries: [FileEntry] = []
    @Published var stagedEntries: [FileEntry] = []
    @Published var selectedPath: String?
    @Published var isLoading = false
    @Published var loadError: String?

    // Comments & composer
    @Published var comments: [ReviewComment] = []
    @Published var savedReplies: [String] = []
    @Published var composer: ComposerState?
    @Published var editingID: String?
    @Published var editText: String = ""

    // Misc UI
    @Published var toast: String?
    @Published var summarySnapshot: String = ""
    @Published var login: String = ""

    private var remoteComments: [RemoteComment] = []
    private var persisted: PersistedState
    private var cache: AppCache
    private var slugs: [String: String] = [:]
    private var loadGen = 0
    private var toastGen = 0
    private var refreshedThisSession: Set<String> = []

    private(set) var currentProject: Project?
    private(set) var currentPR: PullRequest?

    init() {
        persisted = Persistence.load()
        cache = CacheStore.load()
        savedReplies = persisted.savedReplies
        // Render instantly from the last known state; refresh in the background.
        slugs = cache.slugs
        prsByProject = cache.prs
        wtCounts = cache.wtCounts
        needsAuth = cache.needsAuth
        expanded = cache.expanded
        Task { await bootstrap() }
    }

    // MARK: - Bootstrap & refresh

    func bootstrap() async {
        projects = await GitClient.discoverProjects(root: Self.codeRoot)
        if expanded.isEmpty, let first = projects.first {
            expanded.insert(first.id)
        }
        if let l = await GitHubClient.currentLogin() { login = l }
        // Refresh expanded projects plus the most recently active repos.
        for project in projects where expanded.contains(project.id) {
            ensureProjectLoaded(project)
        }
        for project in projects.prefix(6) {
            ensureProjectLoaded(project)
        }
    }

    func refresh() {
        refreshedThisSession.removeAll()
        Task { projects = await GitClient.discoverProjects(root: Self.codeRoot) }
        for id in Array(expanded) {
            if let p = projects.first(where: { $0.id == id }) {
                ensureProjectLoaded(p, force: true)
            }
        }
        switch selection {
        case .none:
            break
        case .pr(let projectID, let number):
            if let p = projects.first(where: { $0.id == projectID }),
               let pr = prsByProject[projectID]?.first(where: { $0.number == number }) {
                select(pr: pr, in: p)
            }
        case .workingTree(let projectID):
            if let p = projects.first(where: { $0.id == projectID }) {
                selectWorkingTree(p)
            }
        }
    }

    func ensureProjectLoaded(_ project: Project, force: Bool = false) {
        if !force {
            if refreshedThisSession.contains(project.id) || loadingPRs.contains(project.id) { return }
        }
        refreshedThisSession.insert(project.id)
        // Only show a spinner when we have nothing cached to display.
        if prsByProject[project.id] == nil {
            loadingPRs.insert(project.id)
        }
        Task {
            async let slugTask = GitClient.remoteSlug(repo: project.path)
            async let statusTask = GitClient.status(repo: project.path)
            let slug = await slugTask
            let statusEntries = await statusTask

            var prs: [PullRequest]?
            var error: String?
            var denied = false
            if slug != nil {
                switch await GitHubClient.prList(repo: project.path) {
                case .success(let list):
                    prs = list
                case .failure(let e):
                    if GitHubClient.isAccessError(e.message) {
                        denied = true
                    } else {
                        error = e.message.isEmpty ? "could not load pull requests" : e.message
                    }
                }
            }
            if let slug {
                slugs[project.id] = slug
                cache.slugs[project.id] = slug
            }
            wtCounts[project.id] = statusEntries.count
            cache.wtCounts[project.id] = statusEntries.count

            if denied {
                needsAuth.insert(project.id)
                cache.needsAuth.insert(project.id)
                if prsByProject[project.id] == nil { prsByProject[project.id] = [] }
            } else if let prs {
                needsAuth.remove(project.id)
                cache.needsAuth.remove(project.id)
                prsByProject[project.id] = prs
                cache.prs[project.id] = prs
            } else if slug == nil, prsByProject[project.id] == nil {
                prsByProject[project.id] = []
            }
            prErrors[project.id] = error
            loadingPRs.remove(project.id)
            CacheStore.save(cache)
        }
    }

    func toggleProject(_ project: Project) {
        if expanded.contains(project.id) {
            expanded.remove(project.id)
        } else {
            expanded.insert(project.id)
            ensureProjectLoaded(project)
        }
        cache.expanded = expanded
        CacheStore.save(cache)
    }

    // MARK: - GitHub auth

    /// Hands the interactive `gh auth login` flow to Terminal (it needs a TTY),
    /// so the user can add the account this machine is missing.
    func startGitHubAuth() {
        Task {
            let script = """
            tell application "Terminal"
                activate
                do script "gh auth login"
            end tell
            """
            let r = await Shell.run(["osascript", "-e", script])
            if r.ok {
                showToast("Finish signing in in Terminal, then press ⌘R to refresh")
            } else {
                let pb = NSPasteboard.general
                pb.clearContents()
                pb.setString("gh auth login", forType: .string)
                showToast("Couldn't open Terminal — `gh auth login` copied to clipboard")
            }
        }
    }

    func slug(for projectID: String) -> String? { slugs[projectID] }

    // MARK: - Review keys & local state

    var reviewKey: String? {
        switch selection {
        case .none:
            return nil
        case .workingTree(let projectID):
            return "wt:" + projectID
        case .pr(let projectID, let number):
            return (slugs[projectID] ?? projectID) + "#\(number)"
        }
    }

    private func localState() -> ReviewLocalState {
        guard let key = reviewKey else { return ReviewLocalState() }
        return persisted.reviews[key] ?? ReviewLocalState()
    }

    private func mutateLocal(_ mutate: (inout ReviewLocalState) -> Void) {
        guard let key = reviewKey else { return }
        var state = persisted.reviews[key] ?? ReviewLocalState()
        mutate(&state)
        persisted.reviews[key] = state
        Persistence.save(persisted)
    }

    var viewedSet: Set<String> { localState().viewed }

    var isWorkingTree: Bool {
        if case .workingTree = selection { return true }
        return false
    }

    var isPR: Bool {
        if case .pr = selection { return true }
        return false
    }

    // MARK: - Selecting things

    private func prKey(_ project: Project, _ number: Int) -> String {
        (slugs[project.id] ?? project.id) + "#\(number)"
    }

    func select(pr: PullRequest, in project: Project) {
        loadGen += 1
        let gen = loadGen
        selection = .pr(projectID: project.id, number: pr.number)
        currentProject = project
        currentPR = pr
        screen = .main
        loadError = nil
        comments = []
        remoteComments = []
        composer = nil
        editingID = nil
        verdict = Verdict(rawValue: localState().verdict) ?? .comment

        let key = prKey(project, pr.number)
        var showedCache = false
        if let cachedDiff = cache.prDiffs[key] {
            applyPRDiff(cachedDiff, comments: cache.prComments[key] ?? [], resetSelection: true)
            showedCache = true
            isLoading = false
        } else {
            isLoading = true
            displayFiles = []
            changeEntries = []
            stagedEntries = []
            selectedPath = nil
        }

        Task {
            async let diffTask = GitHubClient.prDiff(repo: project.path, number: pr.number)
            async let commentsTask = GitHubClient.prComments(repo: project.path, number: pr.number)
            let diffResult = await diffTask
            let fetched = await commentsTask
            guard gen == loadGen else { return }

            switch diffResult {
            case .failure(let e):
                if showedCache {
                    showToast("Refresh failed — showing cached data")
                } else {
                    loadError = e.message.isEmpty ? "Could not load the PR diff." : e.message
                    remoteComments = fetched
                    rebuildComments()
                    rebuildEntries()
                }
            case .success(let text):
                let changed = cache.prDiffs[key] != text || (cache.prComments[key] ?? []) != fetched
                cache.prDiffs[key] = text
                cache.prComments[key] = fetched
                cache.prTouched[key] = Date()
                CacheStore.save(cache)
                if changed || !showedCache {
                    applyPRDiff(text, comments: fetched, resetSelection: !showedCache)
                }
            }
            isLoading = false
        }
    }

    private func applyPRDiff(_ text: String, comments fetched: [RemoteComment], resetSelection: Bool) {
        let files = DiffParser.parse(text).map { fd in
            DisplayFile(path: fd.path, status: fd.status, additions: fd.additions,
                        deletions: fd.deletions, hunks: fd.hunks,
                        isBinary: fd.isBinary, untracked: false)
        }
        displayFiles = files
        if resetSelection || !(selectedPath.map { p in files.contains { $0.path == p } } ?? false) {
            selectedPath = files.first?.path
        }
        remoteComments = fetched
        loadError = nil
        rebuildComments()
        rebuildEntries()
    }

    func selectWorkingTree(_ project: Project) {
        loadGen += 1
        let gen = loadGen
        let keepPath = (currentProject?.id == project.id && isWorkingTree) ? selectedPath : nil
        selection = .workingTree(projectID: project.id)
        currentProject = project
        currentPR = nil
        screen = .main
        isLoading = true
        loadError = nil
        comments = []
        remoteComments = []
        composer = nil
        editingID = nil
        verdict = Verdict(rawValue: localState().verdict) ?? .comment

        Task {
            let state = await GitClient.workingTree(repo: project.path)
            guard gen == loadGen else { return }
            applyWorkingTree(state, project: project, keepPath: keepPath)
            rebuildComments()
            rebuildEntries()
            isLoading = false
        }
    }

    private func applyWorkingTree(_ state: WorkingTreeState, project: Project, keepPath: String?) {
        var byPath: [String: DisplayFile] = [:]
        var order: [String] = []

        func merge(path: String, status: String, hunks: [Hunk], add: Int, del: Int, binary: Bool, untracked: Bool) {
            if var existing = byPath[path] {
                existing = DisplayFile(
                    path: path,
                    status: existing.status == "M" ? status : existing.status,
                    additions: existing.additions + add,
                    deletions: existing.deletions + del,
                    hunks: existing.hunks + hunks,
                    isBinary: existing.isBinary || binary,
                    untracked: existing.untracked || untracked
                )
                byPath[path] = existing
            } else {
                order.append(path)
                byPath[path] = DisplayFile(path: path, status: status, additions: add, deletions: del,
                                           hunks: hunks, isBinary: binary, untracked: untracked)
            }
        }

        for fd in state.stagedDiffs {
            merge(path: fd.path, status: fd.status, hunks: fd.hunks,
                  add: fd.additions, del: fd.deletions, binary: fd.isBinary, untracked: false)
        }
        for fd in state.unstagedDiffs {
            merge(path: fd.path, status: fd.status, hunks: fd.hunks,
                  add: fd.additions, del: fd.deletions, binary: fd.isBinary, untracked: false)
        }
        for path in state.untracked {
            let df = DiffParser.untrackedFile(repo: project.path, path: path)
            merge(path: path, status: "A", hunks: df.hunks,
                  add: df.additions, del: df.deletions, binary: df.isBinary, untracked: true)
        }

        displayFiles = order.compactMap { byPath[$0] }
        wtCounts[project.id] = state.entries.count

        var unstaged: [FileEntry] = []
        var staged: [FileEntry] = []
        let unstagedByPath = Dictionary(uniqueKeysWithValues: state.unstagedDiffs.map { ($0.path, $0) })
        let stagedByPath = Dictionary(uniqueKeysWithValues: state.stagedDiffs.map { ($0.path, $0) })

        for entry in state.entries {
            let path = entry.path
            if entry.index == "?" {
                unstaged.append(FileEntry(path: path, status: "A", add: byPath[path]?.additions ?? 0,
                                          del: 0, checked: false, inStagedSection: false))
                continue
            }
            if entry.worktree != " " {
                let fd = unstagedByPath[path]
                unstaged.append(FileEntry(path: path, status: String(entry.worktree), add: fd?.additions ?? 0,
                                          del: fd?.deletions ?? 0, checked: false, inStagedSection: false))
            }
            if entry.index != " " {
                let fd = stagedByPath[path]
                staged.append(FileEntry(path: path, status: String(entry.index), add: fd?.additions ?? 0,
                                        del: fd?.deletions ?? 0, checked: true, inStagedSection: true))
            }
        }
        changeEntries = unstaged
        stagedEntries = staged

        if let keepPath, byPath[keepPath] != nil {
            selectedPath = keepPath
        } else {
            selectedPath = displayFiles.first?.path
        }
    }

    private func reloadWorkingTree() {
        guard let project = currentProject, isWorkingTree else { return }
        let keep = selectedPath
        Task {
            let state = await GitClient.workingTree(repo: project.path)
            applyWorkingTree(state, project: project, keepPath: keep)
            rebuildEntries()
        }
    }

    // MARK: - File list actions

    func selectFile(_ path: String) {
        selectedPath = path
        composer = nil
        editingID = nil
    }

    func toggleEntry(_ entry: FileEntry) {
        guard let project = currentProject else { return }
        if isWorkingTree {
            Task {
                let result = entry.inStagedSection
                    ? await GitClient.unstageFile(repo: project.path, path: entry.path)
                    : await GitClient.stageFile(repo: project.path, path: entry.path)
                if !result.ok {
                    showToast("git: " + result.stderr.trimmingCharacters(in: .whitespacesAndNewlines))
                }
                reloadWorkingTree()
            }
        } else {
            mutateLocal { state in
                if state.viewed.contains(entry.path) {
                    state.viewed.remove(entry.path)
                } else {
                    state.viewed.insert(entry.path)
                }
            }
            rebuildEntries()
        }
    }

    func toggleViewed(_ path: String) {
        mutateLocal { state in
            if state.viewed.contains(path) {
                state.viewed.remove(path)
            } else {
                state.viewed.insert(path)
            }
        }
        rebuildEntries()
    }

    func toggleHunk(_ hunk: Hunk) {
        guard let project = currentProject, isWorkingTree else { return }
        guard !hunk.rawPatch.isEmpty else {
            showToast("Untracked file — stage the whole file from the list on the left.")
            return
        }
        Task {
            let result = hunk.staged
                ? await GitClient.unstageHunk(repo: project.path, patch: hunk.rawPatch)
                : await GitClient.stageHunk(repo: project.path, patch: hunk.rawPatch)
            if !result.ok {
                showToast("git apply: " + result.stderr.trimmingCharacters(in: .whitespacesAndNewlines))
            }
            reloadWorkingTree()
        }
    }

    private func rebuildEntries() {
        var counts: [String: Int] = [:]
        for c in comments {
            if let p = c.path { counts[p, default: 0] += 1 }
        }
        if isPR {
            let viewed = viewedSet
            changeEntries = displayFiles.map { f in
                FileEntry(path: f.path, status: f.status, add: f.additions, del: f.deletions,
                          checked: viewed.contains(f.path), inStagedSection: false,
                          commentCount: counts[f.path] ?? 0)
            }
            stagedEntries = []
        } else {
            changeEntries = changeEntries.map { e in
                var e2 = e
                e2.commentCount = counts[e.path] ?? 0
                return e2
            }
            stagedEntries = stagedEntries.map { e in
                var e2 = e
                e2.commentCount = 0
                return e2
            }
        }
    }

    // MARK: - Comments

    func rebuildComments() {
        let state = localState()
        var out: [ReviewComment] = []
        for rc in remoteComments {
            let rid = "r\(rc.id)"
            if state.deletedRemote.contains(rid) { continue }
            let raw = state.reworded[rid] ?? rc.body
            let (cat, clean) = CategoryParser.split(raw)
            let mine = !login.isEmpty && rc.author == login
            out.append(ReviewComment(
                id: rid, author: rc.author, isAI: rc.isBot, isMine: mine,
                category: cat, body: clean, path: rc.path, line: rc.line, side: rc.side
            ))
        }
        for lc in state.localComments {
            let (cat, clean) = CategoryParser.split(lc.body)
            out.append(ReviewComment(
                id: lc.id, author: "You", isAI: false, isMine: true,
                category: cat, body: clean, path: lc.path, line: lc.line, side: "RIGHT"
            ))
        }
        comments = out
    }

    func rawBody(of comment: ReviewComment) -> String {
        let state = localState()
        if comment.id.hasPrefix("r") {
            if let reworded = state.reworded[comment.id] { return reworded }
            if let intID = Int(comment.id.dropFirst()),
               let rc = remoteComments.first(where: { $0.id == intID }) {
                return rc.body
            }
        }
        if let lc = state.localComments.first(where: { $0.id == comment.id }) {
            return lc.body
        }
        return comment.body
    }

    func beginEdit(_ comment: ReviewComment) {
        editingID = comment.id
        editText = rawBody(of: comment)
    }

    func cancelEdit() {
        editingID = nil
        editText = ""
    }

    func saveEdit() {
        guard let id = editingID else { return }
        let text = editText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { cancelEdit(); return }
        mutateLocal { state in
            if id.hasPrefix("r") {
                state.reworded[id] = text
            } else if let idx = state.localComments.firstIndex(where: { $0.id == id }) {
                state.localComments[idx].body = text
            }
        }
        cancelEdit()
        rebuildComments()
        rebuildEntries()
    }

    func deleteComment(_ comment: ReviewComment) {
        mutateLocal { state in
            if comment.id.hasPrefix("r") {
                state.deletedRemote.insert(comment.id)
            } else {
                state.localComments.removeAll { $0.id == comment.id }
            }
        }
        rebuildComments()
        rebuildEntries()
    }

    func openComposer(path: String, line: Int?) {
        if let existing = composer, existing.path == path, existing.line == line,
           existing.text.isEmpty {
            composer = nil
            return
        }
        composer = ComposerState(path: path, line: line)
    }

    func closeComposer() { composer = nil }

    func addComment() {
        guard let c = composer else { return }
        let text = c.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let lc = LocalComment(id: "l" + UUID().uuidString, path: c.path, line: c.line,
                              body: text, createdAt: Date())
        mutateLocal { $0.localComments.append(lc) }
        composer = nil
        rebuildComments()
        rebuildEntries()
    }

    // MARK: - Saved replies

    func insertReply(_ reply: String) {
        guard var c = composer else { return }
        let trimmed = c.text.trimmingCharacters(in: .whitespacesAndNewlines)
        c.text = trimmed.isEmpty ? reply : trimmed + "\n" + reply
        composer = c
    }

    func removeReply(_ reply: String) {
        savedReplies.removeAll { $0 == reply }
        persisted.savedReplies = savedReplies
        Persistence.save(persisted)
    }

    func saveCurrentReply() {
        guard let c = composer else { return }
        let text = c.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !savedReplies.contains(text) else { return }
        savedReplies.append(text)
        persisted.savedReplies = savedReplies
        Persistence.save(persisted)
        showToast("Saved reply added")
    }

    // MARK: - Verdict, summary & submission

    func setVerdict(_ v: Verdict) {
        verdict = v
        mutateLocal { $0.verdict = v.rawValue }
    }

    var reviewTitle: String {
        guard let project = currentProject else { return "" }
        if let pr = currentPR {
            return "\(project.name) #\(pr.number) (\(pr.title))"
        }
        return "\(project.name) (working tree)"
    }

    func buildSummary() -> String {
        var out = "# Code review — \(reviewTitle)\n"
        out += "Verdict: \(verdict.rawValue)\n"
        let active = comments
        var byPath: [String: [ReviewComment]] = [:]
        var pathOrder: [String] = []
        for c in active {
            let p = c.path ?? "(general)"
            if byPath[p] == nil { pathOrder.append(p) }
            byPath[p, default: []].append(c)
        }
        // Keep the file-list order where possible.
        let ordered = displayFiles.map(\.path).filter { byPath[$0] != nil }
            + pathOrder.filter { p in !displayFiles.contains(where: { $0.path == p }) }
        if active.isEmpty {
            out += "\n(no comments)\n"
        }
        for path in ordered {
            guard let cs = byPath[path] else { continue }
            out += "\n## \(path)\n"
            for c in cs.sorted(by: { ($0.line ?? 0) < ($1.line ?? 0) }) {
                var lineStr = ""
                if let l = c.line { lineStr = "L\(l): " }
                let cat = c.category.map { "[\($0)] " } ?? ""
                out += "- \(cat)\(lineStr)\(c.body.replacingOccurrences(of: "\n", with: "\n  "))\n"
            }
        }
        let fileCount = displayFiles.count
        out += "\n\(active.count) comment\(active.count == 1 ? "" : "s") · \(fileCount) file\(fileCount == 1 ? "" : "s") changed\n"
        return out
    }

    func copySummary() {
        guard selection != .none else { return }
        let text = buildSummary()
        summarySnapshot = text
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        screen = .submitted
    }

    func backToReview() {
        screen = selection == .none ? .empty : .main
    }

    func submitToGitHub() {
        guard let project = currentProject, let pr = currentPR else { return }
        let body = summarySnapshot.isEmpty ? buildSummary() : summarySnapshot
        showToast("Submitting review to GitHub…")
        Task {
            let result = await GitHubClient.submitReview(
                repo: project.path, number: pr.number, verdict: verdict, body: body
            )
            switch result {
            case .success(let msg): showToast(msg)
            case .failure(let e): showToast("gh: " + (e.message.isEmpty ? "review submission failed" : e.message))
            }
        }
    }

    func openCurrentPROnGitHub() {
        guard let project = currentProject, let pr = currentPR,
              let slug = slugs[project.id],
              let url = GitHubClient.prURL(slug: slug, number: pr.number) else { return }
        NSWorkspace.shared.open(url)
    }

    // MARK: - Toast

    func showToast(_ message: String) {
        toastGen += 1
        let gen = toastGen
        toast = message
        Task {
            try? await Task.sleep(nanoseconds: 3_500_000_000)
            if gen == toastGen { toast = nil }
        }
    }

    // MARK: - Diff rows

    var selectedFile: DisplayFile? {
        guard let path = selectedPath else { return displayFiles.first }
        return displayFiles.first(where: { $0.path == path }) ?? displayFiles.first
    }

    func rows(for file: DisplayFile) -> [DiffRowModel] {
        var rows: [DiffRowModel] = []
        let fileComments = comments.filter { $0.path == file.path }
        var byLine: [Int: [ReviewComment]] = [:]
        var unanchored: [ReviewComment] = []
        for c in fileComments {
            if let l = c.line { byLine[l, default: []].append(c) } else { unanchored.append(c) }
        }
        var placed = Set<Int>()
        let composerActive = composer?.path == file.path
        let composerLine = composer?.line ?? nil
        var composerPlaced = false

        func maybeThread(after lineNo: Int?) {
            guard let n = lineNo else { return }
            let cs = placed.contains(n) ? [] : (byLine[n] ?? [])
            let wantComposer = composerActive && composerLine == n && !composerPlaced
            guard !cs.isEmpty || wantComposer else { return }
            placed.insert(n)
            if wantComposer { composerPlaced = true }
            rows.append(.thread(ThreadModel(
                id: "t:\(file.path):\(n)", path: file.path, line: n,
                comments: cs, showComposer: wantComposer
            )))
        }

        if file.isBinary && file.hunks.isEmpty {
            rows.append(.note("bin:\(file.path)", "Binary file — no text diff to show."))
        } else if file.hunks.isEmpty {
            rows.append(.note("empty:\(file.path)", "No changes to show for this file."))
        }

        for hunk in file.hunks {
            rows.append(.hunk(hunk))
            if mode == .unified {
                for (idx, line) in hunk.lines.enumerated() {
                    rows.append(.uline("\(hunk.id):\(idx)", line))
                    maybeThread(after: line.newNo ?? line.oldNo)
                }
            } else {
                for (idx, pair) in splitPairs(hunk).enumerated() {
                    rows.append(.pair("\(hunk.id):p\(idx)", pair.0, pair.1))
                    maybeThread(after: pair.1?.newNo ?? pair.0?.oldNo)
                }
            }
        }

        // Threads whose lines never appeared in the diff, then file-level comments.
        for (n, cs) in byLine.filter({ !placed.contains($0.key) }).sorted(by: { $0.key < $1.key }) {
            let wantComposer = composerActive && composerLine == n && !composerPlaced
            if wantComposer { composerPlaced = true }
            rows.append(.thread(ThreadModel(
                id: "t:\(file.path):\(n)", path: file.path, line: n,
                comments: cs, showComposer: wantComposer
            )))
        }
        let wantFileComposer = composerActive && composerLine == nil && !composerPlaced
        if !unanchored.isEmpty || wantFileComposer {
            rows.append(.thread(ThreadModel(
                id: "t:\(file.path):file", path: file.path, line: nil,
                comments: unanchored, showComposer: wantFileComposer
            )))
        }
        return rows
    }

    private func splitPairs(_ hunk: Hunk) -> [(DiffLine?, DiffLine?)] {
        var out: [(DiffLine?, DiffLine?)] = []
        var i = 0
        let lines = hunk.lines
        while i < lines.count {
            if lines[i].kind == .ctx {
                out.append((lines[i], lines[i]))
                i += 1
            } else {
                var dels: [DiffLine] = []
                var adds: [DiffLine] = []
                while i < lines.count, lines[i].kind == .del { dels.append(lines[i]); i += 1 }
                while i < lines.count, lines[i].kind == .add { adds.append(lines[i]); i += 1 }
                for j in 0..<max(dels.count, adds.count) {
                    out.append((j < dels.count ? dels[j] : nil, j < adds.count ? adds[j] : nil))
                }
            }
        }
        return out
    }
}
