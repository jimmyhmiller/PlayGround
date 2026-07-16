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
    @Published var authIssues: [String: AccessIssue] = [:]
    @Published var hiddenProjects: Set<String> = []
    @Published var projectOrder: [String] = []
    @Published var showHidden = false

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
        hiddenProjects = persisted.hiddenProjects ?? []
        projectOrder = persisted.projectOrder ?? []
        // Render instantly from the last known state; refresh in the background.
        slugs = cache.slugs
        prsByProject = cache.prs
        wtCounts = cache.wtCounts
        needsAuth = cache.needsAuth
        authIssues = cache.authIssues ?? [:]
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
        // Load tree + PR status for every project (throttled) so the sidebar
        // colors are meaningful at a glance; cached results show instantly.
        await loadAllProjects()
    }

    func refresh() {
        refreshedThisSession.removeAll()
        Task {
            projects = await GitClient.discoverProjects(root: Self.codeRoot)
            await loadAllProjects(force: true)
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
        Task { await loadProject(project) }
    }

    /// How many projects get an eager GitHub PR fetch on launch/refresh. The
    /// tree scan finds every clone under ~/Documents/Code (third-party ones
    /// included), and hitting GitHub for all of them would be slow and
    /// pointless — the rest load when expanded.
    private static let prPriorityLimit = 30

    /// Refreshes every project's local git status (cheap), then fetches PRs for
    /// the projects nearest the top of the user's order. Hidden projects are
    /// skipped entirely.
    private func loadAllProjects(force: Bool = false) async {
        let targets = orderedProjects.filter {
            (force || !refreshedThisSession.contains($0.id)) && !hiddenProjects.contains($0.id)
        }
        for t in targets { refreshedThisSession.insert(t.id) }

        // Phase 1 — local only, so dirty dots light up across the whole list fast.
        await runLimited(targets, limit: 12) { await self.loadLocal($0) }

        // Phase 2 — GitHub, for the projects the user actually keeps on top.
        let priority = targets.filter { slugs[$0.id] != nil }.prefix(Self.prPriorityLimit)
        await runLimited(Array(priority), limit: 6) { await self.loadPRs($0) }

        for p in targets where !priority.contains(where: { $0.id == p.id }) {
            loadingPRs.remove(p.id)
        }
        CacheStore.save(cache)
    }

    private func runLimited(_ items: [Project], limit: Int, work: @escaping (Project) async -> Void) async {
        await withTaskGroup(of: Void.self) { group in
            var idx = 0
            while idx < items.count, idx < limit {
                let p = items[idx]
                group.addTask { await work(p) }
                idx += 1
            }
            for await _ in group {
                if idx < items.count {
                    let p = items[idx]
                    group.addTask { await work(p) }
                    idx += 1
                }
            }
        }
    }

    /// Local git facts: remote slug + working-tree dirtiness. No network.
    private func loadLocal(_ project: Project) async {
        async let slugTask = GitClient.remoteSlug(repo: project.path)
        async let statusTask = GitClient.status(repo: project.path)
        let slug = await slugTask
        let statusEntries = await statusTask

        if let slug {
            slugs[project.id] = slug
            cache.slugs[project.id] = slug
        }
        wtCounts[project.id] = statusEntries.count
        cache.wtCounts[project.id] = statusEntries.count
        if slug == nil, prsByProject[project.id] == nil {
            prsByProject[project.id] = []
            loadingPRs.remove(project.id)
        }
    }

    /// Open pull requests from GitHub for one project.
    private func loadPRs(_ project: Project) async {
        guard slugs[project.id] != nil else {
            loadingPRs.remove(project.id)
            return
        }
        var error: String?
        switch await GitHubClient.prList(repo: project.path) {
        case .success(let list):
            needsAuth.remove(project.id)
            cache.needsAuth.remove(project.id)
            authIssues[project.id] = nil
            cache.authIssues?[project.id] = nil
            prsByProject[project.id] = list
            cache.prs[project.id] = list
        case .failure(let e):
            if let issue = GitHubClient.accessIssue(from: e.message) {
                needsAuth.insert(project.id)
                cache.needsAuth.insert(project.id)
                authIssues[project.id] = issue
                cache.authIssues = (cache.authIssues ?? [:]).merging([project.id: issue]) { _, new in new }
                if prsByProject[project.id] == nil { prsByProject[project.id] = [] }
            } else {
                error = e.message.isEmpty ? "could not load pull requests" : e.message
            }
        }
        prErrors[project.id] = error
        loadingPRs.remove(project.id)
        scheduleCacheSave()
    }

    private func loadProject(_ project: Project) async {
        await loadLocal(project)
        await loadPRs(project)
    }

    private var cacheSavePending = false
    private func scheduleCacheSave() {
        guard !cacheSavePending else { return }
        cacheSavePending = true
        Task {
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            cacheSavePending = false
            CacheStore.save(cache)
        }
    }

    // MARK: - Project ordering & hiding

    /// Projects in the user's chosen order; anything unordered falls back to
    /// most-recently-active.
    var orderedProjects: [Project] {
        let orderIndex = Dictionary(uniqueKeysWithValues: projectOrder.enumerated().map { ($1, $0) })
        return projects.sorted { a, b in
            switch (orderIndex[a.id], orderIndex[b.id]) {
            case let (x?, y?): return x < y
            case (.some, .none): return true
            case (.none, .some): return false
            case (.none, .none): return a.modified > b.modified
            }
        }
    }

    func moveProject(_ id: String, before targetID: String) {
        var order = orderedProjects.map(\.id)
        guard let from = order.firstIndex(of: id),
              let to = order.firstIndex(of: targetID),
              from != to else { return }
        order.move(fromOffsets: IndexSet(integer: from), toOffset: to > from ? to + 1 : to)
        projectOrder = order
        persistProjectPrefs()
    }

    func moveProjectToTop(_ id: String) {
        var order = orderedProjects.map(\.id)
        order.removeAll { $0 == id }
        order.insert(id, at: 0)
        projectOrder = order
        persistProjectPrefs()
    }

    func moveProjectToBottom(_ id: String) {
        var order = orderedProjects.map(\.id)
        order.removeAll { $0 == id }
        order.append(id)
        projectOrder = order
        persistProjectPrefs()
    }

    func hideProject(_ id: String) {
        hiddenProjects.insert(id)
        expanded.remove(id)
        cache.expanded = expanded
        persistProjectPrefs()
        scheduleCacheSave()
    }

    func unhideProject(_ id: String) {
        hiddenProjects.remove(id)
        persistProjectPrefs()
        if let p = projects.first(where: { $0.id == id }) {
            ensureProjectLoaded(p)
        }
    }

    private func persistProjectPrefs() {
        persisted.hiddenProjects = hiddenProjects
        persisted.projectOrder = projectOrder
        Persistence.save(persisted)
    }

    // MARK: - At-a-glance project status

    /// Sidebar square color summarizing GitHub state, plus a dirty-tree flag.
    func glance(for project: Project) -> (color: Color, dirty: Bool) {
        let dirty = (wtCounts[project.id] ?? 0) > 0
        guard let prs = prsByProject[project.id] else {
            return (Color(hex: 0x3a3a40), dirty) // not loaded yet
        }
        let open = prs.filter { $0.state == "OPEN" }
        if open.isEmpty {
            return (Color(hex: 0x5a5a60), dirty)
        }
        if open.contains(where: { $0.checks.failed > 0 || $0.reviewDecision == "CHANGES_REQUESTED" }) {
            return (Th.red, dirty)
        }
        let allSettled = open.allSatisfy { $0.checks.pending == 0 }
        let allApproved = open.allSatisfy { $0.reviewDecision == "APPROVED" }
        if allSettled && allApproved {
            return (Th.green, dirty)
        }
        return (Th.yellow, dirty)
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

    /// Resolves an access problem the way GitHub actually wants it resolved:
    /// SSO grants happen in the browser, missing logins go through `gh auth login`.
    func resolveAccess(for projectID: String) {
        switch authIssues[projectID] {
        case .sso(let urlString, let org):
            if let urlString, let url = URL(string: urlString) {
                NSWorkspace.shared.open(url)
                showToast("Authorize \(org ?? "the org") in your browser, then press ⌘R")
            } else {
                // gh didn't hand back a link; re-running the request prints one.
                startGitHubAuth()
            }
        case .denied, .none:
            startGitHubAuth()
        }
    }

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
