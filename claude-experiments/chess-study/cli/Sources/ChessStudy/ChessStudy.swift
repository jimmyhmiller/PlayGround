import Foundation

// MARK: - Data Models

struct Course: Codable {
    let id: String
    let name: String
    let color: String  // "W" or "B"
}

struct Chapter: Codable {
    let id: String
    let name: String
}

struct Study: Codable {
    let id: String
    let name: String
}

struct MoveData: Codable {
    let fen: String
    let san: String
    let nextFen: String
}

struct Variation: Codable {
    let id: String
    let studyId: String
    let chapterId: String
    let index: Int
    let moves: [String: MoveData]
}

struct Flashcard {
    let courseName: String
    let chapterName: String
    let studyName: String
    let positionFen: String
    let correctMove: String
    let resultingFen: String
    let moveNumber: Int
    let linePreview: String  // e.g., "1.e4 c6 2.d4 d5 3.?"
}

// MARK: - FEN Utilities

func whoToMove(fen: String) -> String {
    let parts = fen.split(separator: " ")
    return parts.count > 1 ? String(parts[1]) : "w"
}

func moveNumber(fen: String) -> Int {
    let parts = fen.split(separator: " ")
    return parts.count >= 6 ? Int(parts[5]) ?? 1 : 1
}

// MARK: - Move Tree Builder

/// Builds an ordered move sequence from start position to each leaf
func buildMoveSequences(from variation: Variation) -> [[MoveData]] {
    let startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    // Build adjacency: fen -> [moves from that position]
    var adjacency: [String: [MoveData]] = [:]
    for (_, moveData) in variation.moves {
        adjacency[moveData.fen, default: []].append(moveData)
    }

    var sequences: [[MoveData]] = []

    func dfs(currentFen: String, path: [MoveData]) {
        guard let movesFromHere = adjacency[currentFen], !movesFromHere.isEmpty else {
            // Leaf node - save this sequence
            if !path.isEmpty {
                sequences.append(path)
            }
            return
        }

        for move in movesFromHere {
            dfs(currentFen: move.nextFen, path: path + [move])
        }
    }

    dfs(currentFen: startFen, path: [])
    return sequences
}

// MARK: - Flashcard Generator

func generateFlashcards(
    variation: Variation,
    playerColor: String,  // "W" or "B"
    courseName: String,
    chapterName: String,
    studyName: String
) -> [Flashcard] {
    let sequences = buildMoveSequences(from: variation)
    var flashcards: [Flashcard] = []

    let playerToMove = playerColor == "W" ? "w" : "b"

    for sequence in sequences {
        var lineNotation: [String] = []

        for move in sequence {
            let moveNum = moveNumber(fen: move.fen)
            let isWhiteMove = whoToMove(fen: move.fen) == "w"

            // Build notation string
            if isWhiteMove {
                lineNotation.append("\(moveNum).\(move.san)")
            } else {
                if lineNotation.isEmpty || !lineNotation.last!.contains(".") {
                    lineNotation.append("\(moveNum)...\(move.san)")
                } else {
                    lineNotation.append(move.san)
                }
            }

            // If this is the player's move, create a flashcard
            if whoToMove(fen: move.fen) == playerToMove {
                // Line preview shows moves up to (but not including) this move
                let previewMoves = Array(lineNotation.dropLast())
                let preview = previewMoves.joined(separator: " ") + " ?"

                let card = Flashcard(
                    courseName: courseName,
                    chapterName: chapterName,
                    studyName: studyName,
                    positionFen: move.fen,
                    correctMove: move.san,
                    resultingFen: move.nextFen,
                    moveNumber: moveNum,
                    linePreview: preview.isEmpty ? "?" : preview
                )
                flashcards.append(card)
            }
        }
    }

    return flashcards
}

// MARK: - Data Loading

func loadJSON<T: Codable>(from path: String) throws -> T {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(T.self, from: data)
}

func loadCourses(dataDir: String) throws -> [Course] {
    return try loadJSON(from: "\(dataDir)/courses.json")
}

func loadChapters(dataDir: String, courseId: String) throws -> [Chapter] {
    return try loadJSON(from: "\(dataDir)/courses/\(courseId).json")
}

func loadStudies(dataDir: String, chapterId: String) throws -> [Study] {
    return try loadJSON(from: "\(dataDir)/chapters/\(chapterId).json")
}

func loadVariationIds(dataDir: String, studyId: String) throws -> [String] {
    return try loadJSON(from: "\(dataDir)/studies/\(studyId).json")
}

func loadVariation(dataDir: String, studyId: String, variationId: String) throws -> Variation {
    return try loadJSON(from: "\(dataDir)/variations/\(studyId)/\(variationId).json")
}

// MARK: - CLI Quiz

func runQuiz(flashcards: [Flashcard]) {
    guard !flashcards.isEmpty else {
        print("No flashcards to study!")
        return
    }

    var shuffled = flashcards.shuffled()
    var correct = 0
    var total = 0

    print("\n Chess Opening Flashcards")
    print("Type your answer or 'q' to quit, 's' to skip\n")

    while !shuffled.isEmpty {
        let card = shuffled.removeFirst()
        total += 1

        print("----------------------------------------")
        print("\(card.courseName) > \(card.chapterName)")
        print("\(card.studyName)")
        print("")
        print("Line: \(card.linePreview)")
        print("")
        print("What is the next move?")
        print("> ", terminator: "")

        guard let input = readLine()?.trimmingCharacters(in: .whitespaces) else {
            break
        }

        if input.lowercased() == "q" {
            break
        }

        if input.lowercased() == "s" {
            print("Skipped. Answer was: \(card.correctMove)")
            shuffled.append(card)  // Put it back for later
            continue
        }

        // Normalize input (handle common variations)
        let normalized = input
            .replacingOccurrences(of: "0-0-0", with: "O-O-O")
            .replacingOccurrences(of: "0-0", with: "O-O")

        if normalized == card.correctMove {
            correct += 1
            print("Correct!")
        } else {
            print("Wrong. The answer was: \(card.correctMove)")
            shuffled.append(card)  // Put it back for later review
        }
        print("")
    }

    print("\n----------------------------------------")
    print("Session complete: \(correct)/\(total) correct")
}

// MARK: - Main

@main
struct ChessStudy {
    static func main() throws {
        // Find data directory relative to executable or use argument
        let args = CommandLine.arguments
        let dataDir: String

        if args.count > 1 {
            dataDir = args[1]
        } else {
            // Default: look for data directory relative to current directory
            let fm = FileManager.default
            let candidates = [
                "./data",
                "../data",
                "../../data"
            ]
            dataDir = candidates.first { fm.fileExists(atPath: "\($0)/courses.json") } ?? "./data"
        }

        print("Loading data from: \(dataDir)")

        // Load all courses
        let courses = try loadCourses(dataDir: dataDir)
        print("Found \(courses.count) courses")

        // Show course menu
        print("\nAvailable courses:")
        for (i, course) in courses.enumerated() {
            let colorIndicator = course.color == "W" ? "[W]" : "[B]"
            print("  \(i + 1). \(colorIndicator) \(course.name)")
        }

        print("\nEnter course number (or 'all' for all courses): ", terminator: "")
        guard let input = readLine()?.trimmingCharacters(in: .whitespaces) else {
            return
        }

        var selectedCourses: [Course]
        if input.lowercased() == "all" {
            selectedCourses = courses
        } else if let num = Int(input), num > 0, num <= courses.count {
            selectedCourses = [courses[num - 1]]
        } else {
            print("Invalid selection")
            return
        }

        // Generate flashcards for selected courses
        var allFlashcards: [Flashcard] = []

        for course in selectedCourses {
            print("\nLoading \(course.name)...")

            let chapters = try loadChapters(dataDir: dataDir, courseId: course.id)

            for chapter in chapters {
                let studies = try loadStudies(dataDir: dataDir, chapterId: chapter.id)

                for study in studies {
                    let variationIds = try loadVariationIds(dataDir: dataDir, studyId: study.id)

                    for varId in variationIds {
                        let variation = try loadVariation(dataDir: dataDir, studyId: study.id, variationId: varId)

                        let cards = generateFlashcards(
                            variation: variation,
                            playerColor: course.color,
                            courseName: course.name,
                            chapterName: chapter.name,
                            studyName: study.name
                        )
                        allFlashcards.append(contentsOf: cards)
                    }
                }
            }
        }

        print("\nGenerated \(allFlashcards.count) flashcards")

        // Deduplicate by position+move (same position might appear in multiple lines)
        var seen = Set<String>()
        let uniqueCards = allFlashcards.filter { card in
            let key = "\(card.positionFen)|\(card.correctMove)"
            if seen.contains(key) {
                return false
            }
            seen.insert(key)
            return true
        }

        // Find move depth range
        let minMove = uniqueCards.map { $0.moveNumber }.min() ?? 1
        let maxMove = uniqueCards.map { $0.moveNumber }.max() ?? 1

        print("\(uniqueCards.count) unique positions (moves \(minMove)-\(maxMove))")

        // Ask for depth filter
        print("\nMove depth range (e.g., '1-5' or '3-10', enter for all): ", terminator: "")
        let depthInput = readLine()?.trimmingCharacters(in: .whitespaces) ?? ""

        var filteredCards = uniqueCards
        if !depthInput.isEmpty {
            let parts = depthInput.split(separator: "-").compactMap { Int($0) }
            if parts.count == 2 {
                let depthMin = parts[0]
                let depthMax = parts[1]
                filteredCards = uniqueCards.filter { $0.moveNumber >= depthMin && $0.moveNumber <= depthMax }
                print("Filtered to \(filteredCards.count) positions (moves \(depthMin)-\(depthMax))")
            } else if parts.count == 1 {
                // Single number means exactly that move
                let depth = parts[0]
                filteredCards = uniqueCards.filter { $0.moveNumber == depth }
                print("Filtered to \(filteredCards.count) positions (move \(depth) only)")
            }
        }

        runQuiz(flashcards: filteredCards)
    }
}
