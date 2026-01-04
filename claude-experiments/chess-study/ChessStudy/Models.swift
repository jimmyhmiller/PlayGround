import Foundation

// MARK: - Data Models

struct Course: Codable, Identifiable {
    let id: String
    let name: String
    let color: String  // "W" or "B"
}

struct Chapter: Codable, Identifiable {
    let id: String
    let name: String
}

struct Study: Codable, Identifiable {
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

struct Flashcard: Identifiable {
    let id = UUID()
    let courseName: String
    let chapterName: String
    let studyName: String
    let positionFen: String
    let correctMove: String
    let resultingFen: String
    let moveNumber: Int
    let linePreview: String
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

func buildMoveSequences(from variation: Variation) -> [[MoveData]] {
    let startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    var adjacency: [String: [MoveData]] = [:]
    for (_, moveData) in variation.moves {
        adjacency[moveData.fen, default: []].append(moveData)
    }

    var sequences: [[MoveData]] = []

    func dfs(currentFen: String, path: [MoveData]) {
        guard let movesFromHere = adjacency[currentFen], !movesFromHere.isEmpty else {
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
    playerColor: String,
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

            if isWhiteMove {
                lineNotation.append("\(moveNum).\(move.san)")
            } else {
                if lineNotation.isEmpty || !lineNotation.last!.contains(".") {
                    lineNotation.append("\(moveNum)...\(move.san)")
                } else {
                    lineNotation.append(move.san)
                }
            }

            if whoToMove(fen: move.fen) == playerToMove {
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
