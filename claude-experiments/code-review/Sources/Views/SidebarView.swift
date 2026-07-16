import SwiftUI
import UniformTypeIdentifiers

struct SidebarView: View {
    @EnvironmentObject var store: AppStore
    @State private var draggingID: String?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Text("PROJECTS")
                    .font(.system(size: 11, weight: .semibold))
                    .kerning(0.5)
                    .foregroundColor(Th.dimmer)
                Spacer()
                if !store.hiddenProjects.isEmpty {
                    Button(action: { store.showHidden.toggle() }) {
                        HStack(spacing: 3) {
                            Image(systemName: store.showHidden ? "eye" : "eye.slash")
                                .font(.system(size: 10))
                            Text("\(store.hiddenProjects.count)")
                                .font(.system(size: 10))
                        }
                        .foregroundColor(store.showHidden ? Th.text3 : Th.dimmer)
                    }
                    .buttonStyle(.plain)
                    .help(store.showHidden ? "Hide the hidden projects section" : "Show hidden projects")
                }
                Button(action: { store.refresh() }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(Th.dimmer)
                }
                .buttonStyle(.plain)
                .help("Refresh (⌘R)")
            }
            .padding(.horizontal, 18)
            .padding(.top, 12)
            .padding(.bottom, 4)

            TextField("Filter projects…", text: $store.projectFilter)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundColor(Th.text)
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(RoundedRectangle(cornerRadius: 7).fill(Color.white.opacity(0.06)))
                .padding(.horizontal, 12)
                .padding(.bottom, 6)

            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(visibleProjects) { project in
                        ProjectRow(project: project)
                            .onDrag {
                                draggingID = project.id
                                return NSItemProvider(object: project.id as NSString)
                            }
                            .onDrop(of: [UTType.text], delegate: ProjectDropDelegate(
                                target: project, draggingID: $draggingID, store: store
                            ))
                            .contextMenu {
                                Button("Move to Top") { store.moveProjectToTop(project.id) }
                                Button("Move to Bottom") { store.moveProjectToBottom(project.id) }
                                Divider()
                                Button("Hide Project") { store.hideProject(project.id) }
                            }
                        if store.expanded.contains(project.id) {
                            ProjectDetail(project: project)
                        }
                    }

                    if store.showHidden {
                        let hidden = hiddenProjects
                        if !hidden.isEmpty {
                            Text("HIDDEN")
                                .font(.system(size: 11, weight: .semibold))
                                .kerning(0.5)
                                .foregroundColor(Th.faint)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 18)
                                .padding(.top, 14)
                                .padding(.bottom, 4)
                            ForEach(hidden) { project in
                                HiddenProjectRow(project: project)
                            }
                        }
                    }
                }
                .padding(.bottom, 20)
                .animation(.default, value: store.projectOrder)
            }

            AccountFooter()
        }
        .frame(width: 236)
        .background(Th.sidebar)
        .overlay(alignment: .trailing) {
            Rectangle().fill(Th.border).frame(width: 1)
        }
    }

    private func matchesFilter(_ project: Project) -> Bool {
        let filter = store.projectFilter.trimmingCharacters(in: .whitespaces).lowercased()
        return filter.isEmpty || project.name.lowercased().contains(filter)
    }

    private var visibleProjects: [Project] {
        store.orderedProjects.filter { !store.hiddenProjects.contains($0.id) && matchesFilter($0) }
    }

    private var hiddenProjects: [Project] {
        store.orderedProjects.filter { store.hiddenProjects.contains($0.id) && matchesFilter($0) }
    }
}

private struct ProjectDropDelegate: DropDelegate {
    let target: Project
    @Binding var draggingID: String?
    let store: AppStore

    func dropEntered(info: DropInfo) {
        guard let dragging = draggingID, dragging != target.id else { return }
        let targetID = target.id
        Task { @MainActor in
            store.moveProject(dragging, before: targetID)
        }
    }

    func dropUpdated(info: DropInfo) -> DropProposal? {
        DropProposal(operation: .move)
    }

    func performDrop(info: DropInfo) -> Bool {
        draggingID = nil
        return true
    }
}

private struct ProjectRow: View {
    @EnvironmentObject var store: AppStore
    let project: Project

    var body: some View {
        let isExpanded = store.expanded.contains(project.id)
        let openCount = store.prsByProject[project.id]?.filter { $0.state == "OPEN" }.count
        let glance = store.glance(for: project)

        Button(action: { store.toggleProject(project) }) {
            HStack(spacing: 8) {
                ZStack(alignment: .topTrailing) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(glance.color)
                        .frame(width: 8, height: 8)
                    if glance.dirty {
                        Circle()
                            .fill(Th.orange)
                            .frame(width: 5, height: 5)
                            .overlay(Circle().strokeBorder(Th.sidebar, lineWidth: 1))
                            .offset(x: 3, y: -3)
                    }
                }
                .frame(width: 11, height: 11)
                .help(glanceHelp(glance))

                Text(project.name)
                    .font(.system(size: 13, weight: isExpanded ? .semibold : .medium))
                    .foregroundColor(isExpanded ? Th.text : Color(hex: 0x9a9aa0))
                    .lineLimit(1)
                Spacer()
                if store.loadingPRs.contains(project.id) {
                    ProgressView().controlSize(.mini)
                } else if let count = openCount, count > 0 {
                    Text("\(count)")
                        .font(.system(size: 11))
                        .foregroundColor(Th.dim)
                }
                Image(systemName: "chevron.right")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundColor(Th.faint)
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 5)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(isExpanded ? 1 : 0.8)
    }

    private func glanceHelp(_ glance: (color: Color, dirty: Bool)) -> String {
        var parts: [String] = []
        let prs = (store.prsByProject[project.id] ?? []).filter { $0.state == "OPEN" }
        if prs.isEmpty {
            parts.append("no open PRs")
        } else if prs.contains(where: { $0.checks.failed > 0 || $0.reviewDecision == "CHANGES_REQUESTED" }) {
            parts.append("PRs failing or changes requested")
        } else {
            parts.append("\(prs.count) open PR\(prs.count == 1 ? "" : "s")")
        }
        if glance.dirty { parts.append("uncommitted changes") }
        return parts.joined(separator: " · ")
    }
}

private struct HiddenProjectRow: View {
    @EnvironmentObject var store: AppStore
    let project: Project

    var body: some View {
        HStack(spacing: 8) {
            RoundedRectangle(cornerRadius: 2)
                .fill(Color(hex: 0x3a3a40))
                .frame(width: 8, height: 8)
            Text(project.name)
                .font(.system(size: 13))
                .foregroundColor(Th.faint)
                .lineLimit(1)
            Spacer()
            Button(action: { store.unhideProject(project.id) }) {
                Image(systemName: "eye")
                    .font(.system(size: 10))
                    .foregroundColor(Th.dim)
            }
            .buttonStyle(.plain)
            .help("Unhide project")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 5)
        .contentShape(Rectangle())
        .contextMenu {
            Button("Unhide Project") { store.unhideProject(project.id) }
        }
    }
}

private struct ProjectDetail: View {
    @EnvironmentObject var store: AppStore
    let project: Project

    var body: some View {
        VStack(spacing: 2) {
            WorkingTreeRow(project: project)

            if store.needsAuth.contains(project.id) {
                let issue = store.authIssues[project.id] ?? .denied
                Button(action: { store.resolveAccess(for: project.id) }) {
                    HStack(spacing: 6) {
                        Image(systemName: "lock.fill")
                            .font(.system(size: 9))
                            .foregroundColor(Th.faint)
                        Text("\(issue.shortLabel) · \(issue.actionLabel)")
                            .font(.system(size: 11))
                            .foregroundColor(Th.dim)
                            .lineLimit(1)
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(issue.help)
            } else {
                if let error = store.prErrors[project.id], visiblePRs.isEmpty {
                    Text(error)
                        .font(.system(size: 10.5))
                        .foregroundColor(Th.redSoft)
                        .lineLimit(3)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                }

                ForEach(visiblePRs) { pr in
                    PRRow(project: project, pr: pr)
                }

                if visiblePRs.isEmpty, store.prErrors[project.id] == nil,
                   store.prsByProject[project.id] != nil {
                    Text(store.slug(for: project.id) == nil ? "no GitHub remote" : "no open pull requests")
                        .font(.system(size: 11))
                        .foregroundColor(Th.faint)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                }
            }
        }
        .padding(.leading, 16)
        .padding(.trailing, 10)
        .padding(.bottom, 6)
    }

    private var visiblePRs: [PullRequest] {
        (store.prsByProject[project.id] ?? []).filter { $0.state == "OPEN" }
    }
}

private struct AccountFooter: View {
    @EnvironmentObject var store: AppStore

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: "person.crop.circle")
                .font(.system(size: 12))
                .foregroundColor(Th.dim)
            Text(store.login.isEmpty ? "not signed in" : store.login)
                .font(.system(size: 11.5))
                .foregroundColor(Th.text3)
                .lineLimit(1)
            Spacer()
            Button(action: { store.startGitHubAuth() }) {
                Image(systemName: "key.fill")
                    .font(.system(size: 10))
                    .foregroundColor(Th.dim)
                    .padding(4)
                    .background(RoundedRectangle(cornerRadius: 5).fill(Color.white.opacity(0.06)))
            }
            .buttonStyle(.plain)
            .help("Add or refresh GitHub authentication (opens Terminal)")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .overlay(alignment: .top) {
            Rectangle().fill(Th.border).frame(height: 1)
        }
    }
}

private struct WorkingTreeRow: View {
    @EnvironmentObject var store: AppStore
    let project: Project

    var body: some View {
        let selected = store.selection == .workingTree(projectID: project.id)
        let count = store.wtCounts[project.id]

        Button(action: { store.selectWorkingTree(project) }) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 7) {
                    Circle()
                        .fill((count ?? 0) > 0 ? Th.orange : Color(hex: 0x5a5a60))
                        .frame(width: 7, height: 7)
                    Text("Working tree")
                        .font(.system(size: 12.5, weight: selected ? .semibold : .medium))
                        .foregroundColor(selected ? Th.text : Th.text2)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
                HStack(spacing: 6) {
                    Image(systemName: "internaldrive")
                        .font(.system(size: 9))
                        .foregroundColor(Th.dim)
                        .frame(width: 15, height: 15)
                    Text(count.map { "\($0) changed file\($0 == 1 ? "" : "s")" } ?? "uncommitted changes")
                        .font(.system(size: 11))
                        .foregroundColor(Th.dim)
                        .lineLimit(1)
                }
                .padding(.leading, 14)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(selected ? Th.accent.opacity(0.18) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(selected ? Th.accent.opacity(0.32) : Color.clear, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private struct PRRow: View {
    @EnvironmentObject var store: AppStore
    let project: Project
    let pr: PullRequest

    var body: some View {
        let selected = store.selection == .pr(projectID: project.id, number: pr.number)

        Button(action: { store.select(pr: pr, in: project) }) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 7) {
                    Circle().fill(dotColor).frame(width: 7, height: 7)
                    Text(pr.title)
                        .font(.system(size: 12.5, weight: selected ? .semibold : .medium))
                        .foregroundColor(selected ? Th.text : Th.text2)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
                HStack(spacing: 6) {
                    Avatar(isAI: pr.authorIsBot, label: initials(pr.author), size: 15)
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundColor(selected ? Th.dim : Color(hex: 0x7a7a80))
                        .lineLimit(1)
                }
                .padding(.leading, 14)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(selected ? Th.accent.opacity(0.18) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(selected ? Th.accent.opacity(0.32) : Color.clear, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var dotColor: Color {
        switch pr.statusLabel {
        case "merged": return Th.purple
        case "closed": return Th.red
        case "approved": return Th.green
        case "changes requested": return Th.red
        case "draft": return Th.faint
        default: return Th.yellow
        }
    }

    private var subtitle: String {
        var parts: [String] = [pr.author]
        switch pr.statusLabel {
        case "open", "approved":
            parts.append(pr.branch)
        default:
            parts.append(pr.statusLabel)
        }
        let ago = TimeFmt.shortAgo(pr.updatedAt)
        if !ago.isEmpty { parts.append(ago) }
        return parts.joined(separator: " · ")
    }
}
