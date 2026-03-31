"""
Process Lichess PGN data into training examples for move-time prediction.

Each example contains:
- Board position as 12-channel 8x8 tensor (one channel per piece type/color)
- Scalar features: elo, time_control_base, time_control_increment, remaining_time, move_number
- Target: log(move_time_seconds + 1)
"""

import io
import re
import struct
import sys
import zstandard as zstd
import chess
import chess.pgn
import numpy as np
from pathlib import Path


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert board to 12-channel 8x8 array.

    Channels 0-5: white P, N, B, R, Q, K
    Channels 6-11: black P, N, B, R, Q, K
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    piece_map = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square, piece in board.piece_map().items():
        rank = square // 8
        file = square % 8
        channel = piece_map[(piece.piece_type, piece.color)]
        tensor[channel, rank, file] = 1.0

    return tensor


def parse_clock(comment: str) -> float | None:
    """Extract clock time in seconds from a PGN comment like '[%clk 0:05:00]'."""
    match = re.search(r'\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]', comment)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def parse_time_control(tc_str: str) -> tuple[int, int] | None:
    """Parse TimeControl header like '300+3' into (base_seconds, increment)."""
    match = re.match(r'(\d+)\+(\d+)', tc_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def process_game(game: chess.pgn.Game) -> list[dict]:
    """Extract training examples from a single game."""
    examples = []

    # Get headers
    white_elo = game.headers.get("WhiteElo", "?")
    black_elo = game.headers.get("BlackElo", "?")
    tc_str = game.headers.get("TimeControl", "?")

    if white_elo == "?" or black_elo == "?" or tc_str == "?":
        return []

    tc = parse_time_control(tc_str)
    if tc is None:
        return []

    base_time, increment = tc

    try:
        white_elo = int(white_elo)
        black_elo = int(black_elo)
    except ValueError:
        return []

    board = game.board()
    node = game
    prev_white_clock = None
    prev_black_clock = None
    move_number = 0

    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        comment = next_node.comment

        clock = parse_clock(comment)
        if clock is None:
            board.push(move)
            node = next_node
            move_number += 1
            continue

        is_white = board.turn == chess.WHITE
        elo = white_elo if is_white else black_elo
        prev_clock = prev_white_clock if is_white else prev_black_clock

        if prev_clock is not None:
            # Time spent = previous clock - current clock + increment
            time_spent = prev_clock - clock + increment

            # Filter out nonsensical values (premoves, disconnects, etc.)
            if time_spent < 0:
                time_spent = 0.0
            if time_spent > base_time * 2:  # way too long, probably disconnect
                board.push(move)
                node = next_node
                move_number += 1
                if is_white:
                    prev_white_clock = clock
                else:
                    prev_black_clock = clock
                continue

            board_tensor = board_to_tensor(board)

            examples.append({
                'board': board_tensor,
                'elo': elo,
                'base_time': base_time,
                'increment': increment,
                'remaining_time': prev_clock,
                'move_number': move_number,
                'turn': 1.0 if is_white else 0.0,
                'target': np.log1p(time_spent),  # log(time + 1)
            })

        if is_white:
            prev_white_clock = clock
        else:
            prev_black_clock = clock

        board.push(move)
        node = next_node
        move_number += 1

    return examples


# Binary format for compact storage:
# Each example: 12*8*8 floats (board) + 7 floats (features + target) = 775 float32s
EXAMPLE_SIZE = (12 * 8 * 8 + 7) * 4  # bytes


def write_example(f, example: dict):
    """Write a single example in binary format."""
    f.write(example['board'].tobytes())
    f.write(struct.pack('fffffff',
        example['elo'],
        example['base_time'],
        example['increment'],
        example['remaining_time'],
        example['move_number'],
        example['turn'],
        example['target'],
    ))


def process_pgn_file(input_path: str, output_path: str, max_games: int | None = None):
    """Process a .pgn.zst file into binary training data."""

    dctx = zstd.ZstdDecompressor()
    total_examples = 0
    total_games = 0
    skipped_games = 0

    with open(input_path, 'rb') as compressed, open(output_path, 'wb') as out:
        reader = dctx.stream_reader(compressed)
        text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')

        while True:
            if max_games is not None and total_games >= max_games:
                break

            try:
                game = chess.pgn.read_game(text_stream)
            except Exception as e:
                skipped_games += 1
                continue

            if game is None:
                break

            total_games += 1
            examples = process_game(game)

            for ex in examples:
                write_example(out, ex)
                total_examples += 1

            if total_games % 10000 == 0:
                print(f"Processed {total_games:,} games, {total_examples:,} examples, {skipped_games:,} skipped", flush=True)

    print(f"\nDone! {total_games:,} games -> {total_examples:,} examples ({skipped_games:,} skipped)")
    print(f"Output: {output_path} ({Path(output_path).stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/lichess_2017-04.pgn.zst"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/train.bin"
    max_games = int(sys.argv[3]) if len(sys.argv) > 3 else None

    process_pgn_file(input_path, output_path, max_games)
