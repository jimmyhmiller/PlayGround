import Foundation

struct ShellResult {
    let stdout: String
    let stderr: String
    let code: Int32

    var ok: Bool { code == 0 }
}

enum Shell {
    /// Runs a command via /usr/bin/env and returns its output. Never throws:
    /// failures are reported through the exit code / stderr.
    static func run(_ arguments: [String], cwd: URL? = nil, stdin: String? = nil) async -> ShellResult {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                process.arguments = arguments

                var env = ProcessInfo.processInfo.environment
                let path = env["PATH"] ?? "/usr/bin:/bin"
                if !path.contains("/opt/homebrew/bin") {
                    env["PATH"] = path + ":/opt/homebrew/bin:/usr/local/bin"
                }
                env["NO_COLOR"] = "1"
                env["GH_NO_UPDATE_NOTIFIER"] = "1"
                env["GIT_PAGER"] = "cat"
                process.environment = env
                if let cwd { process.currentDirectoryURL = cwd }

                let outPipe = Pipe()
                let errPipe = Pipe()
                process.standardOutput = outPipe
                process.standardError = errPipe

                let inPipe: Pipe? = stdin != nil ? Pipe() : nil
                if let inPipe {
                    process.standardInput = inPipe
                } else {
                    process.standardInput = FileHandle.nullDevice
                }

                do {
                    try process.run()
                } catch {
                    cont.resume(returning: ShellResult(stdout: "", stderr: "failed to launch \(arguments.first ?? "?"): \(error.localizedDescription)", code: -1))
                    return
                }

                if let stdin, let inPipe {
                    inPipe.fileHandleForWriting.write(Data(stdin.utf8))
                    inPipe.fileHandleForWriting.closeFile()
                }

                // Drain stderr concurrently so neither pipe can deadlock when full.
                var errData = Data()
                let group = DispatchGroup()
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                    group.leave()
                }
                let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                group.wait()
                process.waitUntilExit()

                cont.resume(returning: ShellResult(
                    stdout: String(data: outData, encoding: .utf8) ?? "",
                    stderr: String(data: errData, encoding: .utf8) ?? "",
                    code: process.terminationStatus
                ))
            }
        }
    }
}
