import SwiftUI

struct ChessBoardView: View {
    let fen: String
    let highlightFrom: String?  // e.g., "e2"
    let highlightTo: String?    // e.g., "e4"
    let playerColor: String     // "W" or "B" - determines board orientation

    private let files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    private let ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    var body: some View {
        let board = parseFEN(fen)
        let flipped = playerColor == "B"

        GeometryReader { geo in
            let totalSize = min(geo.size.width, geo.size.height)
            let labelSize: CGFloat = 16
            let boardSize = totalSize - labelSize
            let squareSize = boardSize / 8

            VStack(spacing: 0) {
                // Board with rank labels
                HStack(spacing: 0) {
                    // Rank labels (left side)
                    VStack(spacing: 0) {
                        ForEach(0..<8, id: \.self) { row in
                            let displayRow = flipped ? row : 7 - row
                            Text(ranks[displayRow])
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                                .frame(width: labelSize, height: squareSize)
                        }
                    }

                    // Board
                    VStack(spacing: 0) {
                        ForEach(0..<8, id: \.self) { row in
                            HStack(spacing: 0) {
                                ForEach(0..<8, id: \.self) { col in
                                    let displayRow = flipped ? row : 7 - row
                                    let displayCol = flipped ? 7 - col : col
                                    let square = "\(files[displayCol])\(ranks[displayRow])"
                                    let piece = board[displayRow][displayCol]
                                    let isLight = (displayRow + displayCol) % 2 == 1

                                    ZStack {
                                        // Square background
                                        Rectangle()
                                            .fill(squareColor(isLight: isLight, square: square))

                                        // Piece
                                        if let piece = piece {
                                            let info = pieceInfo(piece)
                                            ZStack {
                                                // Outline/shadow
                                                Text(info.emoji)
                                                    .font(.system(size: squareSize * 0.75))
                                                    .foregroundColor(info.isWhite ? .black : .white)
                                                    .offset(x: 1, y: 1)
                                                Text(info.emoji)
                                                    .font(.system(size: squareSize * 0.75))
                                                    .foregroundColor(info.isWhite ? .black : .white)
                                                    .offset(x: -1, y: -1)
                                                Text(info.emoji)
                                                    .font(.system(size: squareSize * 0.75))
                                                    .foregroundColor(info.isWhite ? .black : .white)
                                                    .offset(x: 1, y: -1)
                                                Text(info.emoji)
                                                    .font(.system(size: squareSize * 0.75))
                                                    .foregroundColor(info.isWhite ? .black : .white)
                                                    .offset(x: -1, y: 1)
                                                // Main piece
                                                Text(info.emoji)
                                                    .font(.system(size: squareSize * 0.75))
                                                    .foregroundColor(info.isWhite ? .white : .black)
                                            }
                                        }
                                    }
                                    .frame(width: squareSize, height: squareSize)
                                }
                            }
                        }
                    }
                }

                // File labels (bottom)
                HStack(spacing: 0) {
                    Spacer().frame(width: labelSize)
                    ForEach(0..<8, id: \.self) { col in
                        let displayCol = flipped ? 7 - col : col
                        Text(files[displayCol])
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                            .frame(width: squareSize, height: labelSize)
                    }
                }
            }
            .frame(width: totalSize, height: totalSize)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private func squareColor(isLight: Bool, square: String) -> Color {
        if square == highlightTo {
            return Color.green.opacity(0.6)
        } else if square == highlightFrom {
            return Color.yellow.opacity(0.5)
        } else if isLight {
            return Color(red: 0.94, green: 0.85, blue: 0.71)  // Light square
        } else {
            return Color(red: 0.71, green: 0.53, blue: 0.39)  // Dark square
        }
    }

    private func parseFEN(_ fen: String) -> [[Character?]] {
        var board: [[Character?]] = Array(repeating: Array(repeating: nil, count: 8), count: 8)

        let parts = fen.split(separator: " ")
        guard let position = parts.first else { return board }

        let rows = position.split(separator: "/")
        for (rankIndex, row) in rows.enumerated() {
            var fileIndex = 0
            for char in row {
                if let emptyCount = char.wholeNumberValue {
                    fileIndex += emptyCount
                } else {
                    if fileIndex < 8 && rankIndex < 8 {
                        board[7 - rankIndex][fileIndex] = char
                    }
                    fileIndex += 1
                }
            }
        }
        return board
    }

    private func pieceInfo(_ piece: Character) -> (emoji: String, isWhite: Bool) {
        switch piece {
        case "K": return ("♚", true)
        case "Q": return ("♛", true)
        case "R": return ("♜", true)
        case "B": return ("♝", true)
        case "N": return ("♞", true)
        case "P": return ("♟", true)
        case "k": return ("♚", false)
        case "q": return ("♛", false)
        case "r": return ("♜", false)
        case "b": return ("♝", false)
        case "n": return ("♞", false)
        case "p": return ("♟", false)
        default: return ("", true)
        }
    }
}

// MARK: - Move Detection

/// Finds the from and to squares by comparing two FEN positions
func findMoveSquares(beforeFen: String, afterFen: String) -> (from: String?, to: String?) {
    let files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    let ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]

    func parseFEN(_ fen: String) -> [[Character?]] {
        var board: [[Character?]] = Array(repeating: Array(repeating: nil, count: 8), count: 8)
        let parts = fen.split(separator: " ")
        guard let position = parts.first else { return board }
        let rows = position.split(separator: "/")
        for (rankIndex, row) in rows.enumerated() {
            var fileIndex = 0
            for char in row {
                if let emptyCount = char.wholeNumberValue {
                    fileIndex += emptyCount
                } else {
                    if fileIndex < 8 && rankIndex < 8 {
                        board[7 - rankIndex][fileIndex] = char
                    }
                    fileIndex += 1
                }
            }
        }
        return board
    }

    let before = parseFEN(beforeFen)
    let after = parseFEN(afterFen)

    var fromSquare: String?
    var toSquare: String?

    // Find squares that changed
    for rank in 0..<8 {
        for file in 0..<8 {
            let beforePiece = before[rank][file]
            let afterPiece = after[rank][file]

            if beforePiece != afterPiece {
                let square = "\(files[file])\(ranks[rank])"

                // If a square had a piece and now is empty (or different), it's likely the "from"
                if beforePiece != nil && afterPiece == nil {
                    fromSquare = square
                }
                // If a square was empty (or had opponent's piece) and now has a piece, it's the "to"
                else if afterPiece != nil && (beforePiece == nil || beforePiece?.isUppercase != afterPiece?.isUppercase) {
                    toSquare = square
                }
                // Piece replaced by same color piece (e.g., castling rook)
                else if beforePiece != nil && afterPiece != nil {
                    // Could be either from or to depending on context
                    if beforePiece != afterPiece {
                        // Piece changed, could be promotion or capture
                        if afterPiece?.lowercased() != beforePiece?.lowercased() {
                            toSquare = square
                        }
                    }
                }
            }
        }
    }

    // Handle castling - king moves 2 squares
    // If we found a "from" but no clear "to", look for king's new position
    if fromSquare != nil && toSquare == nil {
        for rank in 0..<8 {
            for file in 0..<8 {
                let afterPiece = after[rank][file]
                let beforePiece = before[rank][file]
                if afterPiece != nil && beforePiece != afterPiece {
                    let square = "\(files[file])\(ranks[rank])"
                    if square != fromSquare {
                        toSquare = square
                        break
                    }
                }
            }
            if toSquare != nil { break }
        }
    }

    return (fromSquare, toSquare)
}
