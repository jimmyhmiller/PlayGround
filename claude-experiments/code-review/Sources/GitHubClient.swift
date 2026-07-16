import Foundation

enum GitHubClient {
    static func currentLogin() async -> String? {
        let r = await Shell.run(["gh", "api", "user", "-q", ".login"])
        guard r.ok else { return nil }
        let login = r.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return login.isEmpty ? nil : login
    }

    static func prList(repo: URL) async -> Result<[PullRequest], String> {
        let fields = "number,title,author,headRefName,updatedAt,state,isDraft,reviewDecision,additions,deletions,statusCheckRollup"
        let r = await Shell.run(
            ["gh", "pr", "list", "--state", "all", "--limit", "30", "--json", fields],
            cwd: repo
        )
        guard r.ok else {
            return .failure(r.stderr.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard let data = r.stdout.data(using: .utf8),
              let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] else {
            return .failure("could not parse gh pr list output")
        }
        let prs = arr.map { obj -> PullRequest in
            let author = (obj["author"] as? [String: Any])
            let login = (author?["login"] as? String) ?? "unknown"
            let isBot = (author?["is_bot"] as? Bool) ?? looksLikeBot(login)
            var checks = ChecksSummary()
            for check in (obj["statusCheckRollup"] as? [[String: Any]]) ?? [] {
                let conclusion = ((check["conclusion"] as? String) ?? (check["state"] as? String) ?? "").uppercased()
                switch conclusion {
                case "SUCCESS", "NEUTRAL", "SKIPPED":
                    checks.passed += 1
                case "FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE":
                    checks.failed += 1
                default:
                    checks.pending += 1
                }
            }
            return PullRequest(
                number: (obj["number"] as? Int) ?? 0,
                title: (obj["title"] as? String) ?? "",
                author: login,
                authorIsBot: isBot,
                branch: (obj["headRefName"] as? String) ?? "",
                updatedAt: (obj["updatedAt"] as? String).flatMap { TimeFmt.iso.date(from: $0) },
                state: (obj["state"] as? String) ?? "OPEN",
                isDraft: (obj["isDraft"] as? Bool) ?? false,
                reviewDecision: (obj["reviewDecision"] as? String) ?? "",
                additions: (obj["additions"] as? Int) ?? 0,
                deletions: (obj["deletions"] as? Int) ?? 0,
                checks: checks
            )
        }
        return .success(prs)
    }

    static func prDiff(repo: URL, number: Int) async -> Result<String, String> {
        let r = await Shell.run(["gh", "pr", "diff", String(number)], cwd: repo)
        guard r.ok else {
            return .failure(r.stderr.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return .success(r.stdout)
    }

    static func prComments(repo: URL, number: Int) async -> [RemoteComment] {
        let r = await Shell.run(
            ["gh", "api", "repos/{owner}/{repo}/pulls/\(number)/comments", "--paginate"],
            cwd: repo
        )
        guard r.ok, let data = r.stdout.data(using: .utf8) else { return [] }
        // --paginate can concatenate multiple JSON arrays; normalize by splitting "][".
        let normalized = r.stdout.replacingOccurrences(of: "][", with: ",")
        let parseData = normalized.data(using: .utf8) ?? data
        guard let arr = (try? JSONSerialization.jsonObject(with: parseData)) as? [[String: Any]] else { return [] }
        return arr.compactMap { obj -> RemoteComment? in
            guard let id = obj["id"] as? Int else { return nil }
            let user = obj["user"] as? [String: Any]
            let login = (user?["login"] as? String) ?? "unknown"
            let type = (user?["type"] as? String) ?? ""
            return RemoteComment(
                id: id,
                author: login,
                isBot: type == "Bot" || looksLikeBot(login),
                body: (obj["body"] as? String) ?? "",
                path: obj["path"] as? String,
                line: (obj["line"] as? Int) ?? (obj["original_line"] as? Int),
                side: (obj["side"] as? String) ?? "RIGHT",
                createdAt: (obj["created_at"] as? String) ?? ""
            )
        }
    }

    static func submitReview(repo: URL, number: Int, verdict: Verdict, body: String) async -> Result<String, String> {
        let r = await Shell.run(
            ["gh", "pr", "review", String(number), verdict.ghFlag, "--body-file", "-"],
            cwd: repo,
            stdin: body
        )
        if r.ok {
            return .success("Review submitted to GitHub")
        }
        return .failure(r.stderr.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    static func prURL(slug: String, number: Int) -> URL? {
        URL(string: "https://github.com/\(slug)/pull/\(number)")
    }
}
