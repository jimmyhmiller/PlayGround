use pgn_reader::{BufferedReader, RawComment, RawTag, SanPlus, Skip, Visitor};
use shakmaty::{Chess, Color, Position, Role, Square};
use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::Instant;

/// Compact format per example:
///   12 × u64 bitboards (96 bytes) + 7 × f32 scalars (28 bytes) = 124 bytes
const EXAMPLE_BYTES: usize = 12 * 8 + 7 * 4; // 124

fn board_to_bitboards(board: &Chess) -> [u64; 12] {
    let bb = board.board();
    [
        bb.by_piece(shakmaty::Piece { color: Color::White, role: Role::Pawn }).0,
        bb.by_piece(shakmaty::Piece { color: Color::White, role: Role::Knight }).0,
        bb.by_piece(shakmaty::Piece { color: Color::White, role: Role::Bishop }).0,
        bb.by_piece(shakmaty::Piece { color: Color::White, role: Role::Rook }).0,
        bb.by_piece(shakmaty::Piece { color: Color::White, role: Role::Queen }).0,
        bb.by_piece(shakmaty::Piece { color: Color::White, role: Role::King }).0,
        bb.by_piece(shakmaty::Piece { color: Color::Black, role: Role::Pawn }).0,
        bb.by_piece(shakmaty::Piece { color: Color::Black, role: Role::Knight }).0,
        bb.by_piece(shakmaty::Piece { color: Color::Black, role: Role::Bishop }).0,
        bb.by_piece(shakmaty::Piece { color: Color::Black, role: Role::Rook }).0,
        bb.by_piece(shakmaty::Piece { color: Color::Black, role: Role::Queen }).0,
        bb.by_piece(shakmaty::Piece { color: Color::Black, role: Role::King }).0,
    ]
}

fn parse_clock(comment: &[u8]) -> Option<f32> {
    let s = std::str::from_utf8(comment).ok()?;
    let start = s.find("[%clk ")? + 6;
    let end = s[start..].find(']')? + start;
    let clk = &s[start..end];
    let parts: Vec<&str> = clk.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f32 = parts[0].parse().ok()?;
    let m: f32 = parts[1].parse().ok()?;
    let s: f32 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

fn parse_time_control(tc: &str) -> Option<(f32, f32)> {
    let parts: Vec<&str> = tc.split('+').collect();
    if parts.len() != 2 {
        return None;
    }
    Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
}

struct GameProcessor {
    white_elo: Option<f32>,
    black_elo: Option<f32>,
    base_time: f32,
    increment: f32,
    tc_valid: bool,

    board: Chess,
    prev_white_clock: Option<f32>,
    prev_black_clock: Option<f32>,
    move_number: u32,
    skip_game: bool,

    writer: BufWriter<File>,
    buf: [u8; EXAMPLE_BYTES],

    total_examples: u64,
    total_games: u64,
    skipped_games: u64,
    start_time: Instant,
}

impl GameProcessor {
    fn new(writer: BufWriter<File>) -> Self {
        Self {
            white_elo: None,
            black_elo: None,
            base_time: 0.0,
            increment: 0.0,
            tc_valid: false,
            board: Chess::default(),
            prev_white_clock: None,
            prev_black_clock: None,
            move_number: 0,
            skip_game: false,
            writer,
            buf: [0u8; EXAMPLE_BYTES],
            total_examples: 0,
            total_games: 0,
            skipped_games: 0,
            start_time: Instant::now(),
        }
    }

    fn write_example(&mut self, elo: f32, remaining: f32, move_num: u32, is_white: bool, time_spent: f32) {
        let bitboards = board_to_bitboards(&self.board);

        // Write 12 bitboards (96 bytes)
        for (i, &bb) in bitboards.iter().enumerate() {
            self.buf[i * 8..(i + 1) * 8].copy_from_slice(&bb.to_le_bytes());
        }

        // Write 7 scalars (28 bytes)
        let scalars: [f32; 7] = [
            elo,
            self.base_time,
            self.increment,
            remaining,
            move_num as f32,
            if is_white { 1.0 } else { 0.0 },
            (time_spent + 1.0).ln(),
        ];
        for (i, &s) in scalars.iter().enumerate() {
            let offset = 96 + i * 4;
            self.buf[offset..offset + 4].copy_from_slice(&s.to_le_bytes());
        }

        self.writer.write_all(&self.buf).unwrap();
        self.total_examples += 1;
    }
}

impl Visitor for GameProcessor {
    type Result = ();

    fn begin_game(&mut self) {
        self.white_elo = None;
        self.black_elo = None;
        self.base_time = 0.0;
        self.increment = 0.0;
        self.tc_valid = false;
        self.board = Chess::default();
        self.prev_white_clock = None;
        self.prev_black_clock = None;
        self.move_number = 0;
        self.skip_game = false;
    }

    fn tag(&mut self, key: &[u8], value: RawTag<'_>) {
        match key {
            b"WhiteElo" => {
                if let Ok(s) = std::str::from_utf8(value.as_bytes()) {
                    self.white_elo = s.parse().ok();
                }
            }
            b"BlackElo" => {
                if let Ok(s) = std::str::from_utf8(value.as_bytes()) {
                    self.black_elo = s.parse().ok();
                }
            }
            b"TimeControl" => {
                if let Ok(s) = std::str::from_utf8(value.as_bytes()) {
                    if let Some((base, inc)) = parse_time_control(s) {
                        self.base_time = base;
                        self.increment = inc;
                        self.tc_valid = true;
                    }
                }
            }
            _ => {}
        }
    }

    fn end_tags(&mut self) -> Skip {
        if self.white_elo.is_none() || self.black_elo.is_none() || !self.tc_valid {
            self.skip_game = true;
            self.skipped_games += 1;
            return Skip(true);
        }
        Skip(false)
    }

    fn comment(&mut self, comment: RawComment<'_>) {
        if self.skip_game {
            return;
        }

        let clock = match parse_clock(comment.as_bytes()) {
            Some(c) => c,
            None => return,
        };

        let is_white = self.board.turn() == Color::White;
        let elo = if is_white {
            self.white_elo.unwrap()
        } else {
            self.black_elo.unwrap()
        };
        let prev_clock = if is_white {
            self.prev_white_clock
        } else {
            self.prev_black_clock
        };

        if let Some(prev) = prev_clock {
            let time_spent = prev - clock + self.increment;
            if time_spent >= 0.0 && time_spent <= self.base_time * 2.0 {
                self.write_example(elo, prev, self.move_number, is_white, time_spent);
            }
        }

        if is_white {
            self.prev_white_clock = Some(clock);
        } else {
            self.prev_black_clock = Some(clock);
        }
    }

    fn san(&mut self, san: SanPlus) {
        if self.skip_game {
            return;
        }
        if let Ok(m) = san.san.to_move(&self.board) {
            self.board.play_unchecked(m);
        }
        self.move_number += 1;
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn end_game(&mut self) {
        self.total_games += 1;
        if self.total_games % 100_000 == 0 {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let games_per_sec = self.total_games as f64 / elapsed;
            eprintln!(
                "Processed {:>10} games, {:>12} examples, {:>8} skipped | {:.0} games/s | {:.1}s",
                self.total_games, self.total_examples, self.skipped_games, games_per_sec, elapsed
            );
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: chess-preprocess <input.pgn.zst> <output.bin>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    eprintln!("Reading: {}", input_path);
    eprintln!("Writing: {} (bitboard format, {} bytes/example)", output_path, EXAMPLE_BYTES);

    let input_file = File::open(input_path)?;
    let decoder = zstd::Decoder::new(input_file)?;
    let output_file = File::create(output_path)?;
    let writer = BufWriter::with_capacity(64 * 1024 * 1024, output_file);

    let mut processor = GameProcessor::new(writer);
    let mut reader = BufferedReader::new(decoder);
    reader.read_all(&mut processor)?;
    processor.writer.flush()?;

    let elapsed = processor.start_time.elapsed().as_secs_f64();
    eprintln!("\nDone!");
    eprintln!("Games: {} ({} skipped)", processor.total_games, processor.skipped_games);
    eprintln!("Examples: {}", processor.total_examples);
    eprintln!("Time: {:.1}s ({:.0} games/s)", elapsed, processor.total_games as f64 / elapsed);

    let file_size = std::fs::metadata(output_path)?.len();
    eprintln!("Output: {} ({:.2} GB)", output_path, file_size as f64 / 1e9);

    Ok(())
}
