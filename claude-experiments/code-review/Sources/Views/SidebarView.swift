import SwiftUI

struct SidebarView: View {
    @EnvironmentObject var store: AppStore

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("PROJECTS")
                    .font(.system(size: 11, weight: .semibold))
                    .kerning(0.5)
                    .foregroundColor(Th.dimmer)
                Spacer()
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
                    ForEach(filteredProjects) { project in
                        ProjectRow(project: project)
                        if store.expanded.contains(project.id) {
                            ProjectDetail(project: project)
                        }
                    }
                }
                .padding(.bottom, 20)
            }

            AccountFooter()
        }
        .frame(width: 236)
        .background(Th.sidebar)
        .overlay(alignment: .trailing) {
            Rectangle().fill(Th.border).frame(width: 1)
        }
    }

    private var filteredProjects: [Project] {
        let filter = store.projectFilter.trimmingCharacters(in: .whitespaces).lowercased()
        guard !filter.isEmpty else { return store.projects }
        return store.projects.filter { $0.name.lowercased().contains(filter) }
    }
}

private struct ProjectRow: View {
    @EnvironmentObject var store: AppStore
    let project: Project

    var body: some View {
        let isExpanded = store.expanded.contains(project.id)
        let openCount = store.prsByProject[project.id]?.filter { $0.state == "OPEN" }.count

        Button(action: { store.toggleProject(project) }) {
            HStack(spacing: 8) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(isExpanded ? Th.accent : Color(hex: 0x5a5a60))
                    .frame(width: 8, height: 8)
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
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 5)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(isExpanded ? 1 : 0.75)
    }
}

private struct ProjectDetail: View {
    @EnvironmentObject var store: AppStore
    let project: Project

    var body: some View {
        VStack(spacing: 2) {
            WorkingTreeRow(project: project)

            if store.needsAuth.contains(project.id) {
                Button(action: { store.startGitHubAuth() }) {
                    HStack(spacing: 6) {
                        Image(systemName: "lock.fill")
                            .font(.system(size: 9))
                            .foregroundColor(Th.faint)
                        Text("private repo · Sign in…")
                            .font(.system(size: 11))
                            .foregroundColor(Th.dim)
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("This account can't see this repo's pull requests. Sign in with the right GitHub account.")
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
