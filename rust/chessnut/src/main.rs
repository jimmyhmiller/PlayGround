// See the "macOS permissions note" in README.md before running this on macOS
// Big Sur or later.

use futures::StreamExt;
use std::collections::BTreeSet;
use std::error::Error;
use std::io::{Stdin, Stdout};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use std::vec;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::sleep;
use tokio::{io, task, time};

use btleplug::api::{
    Central, Characteristic, Manager as _, Peripheral, ScanFilter, ValueNotification,
};
use btleplug::platform::{self, Manager};
use vampirc_uci::{parse_one, UciMessage, UciMove, UciSearchControl, UciSquare};

const WRITE: &str = "1b7e8272-2877-41c3-b46e-cf057c562023";
const BOARD_DATA: &str = "1b7e8262-2877-41c3-b46e-cf057c562023";
const OTB_DATA: &str = "1b7e8283-2877-41c3-b46e-cf057c562023";
const MISC_DATA: &str = "1b7e8273-2877-41c3-b46e-cf057c562023";

fn create_led_control_message(rows: [u8; 8]) -> [u8; 10] {
    let control_bytes = [0x0A, 0x08];
    let mut message = [0u8; 10];
    message[..2].copy_from_slice(&control_bytes);
    message[2..].copy_from_slice(&rows);
    message
}

fn led_value_for_coord(coord: Coord) -> u8 {
    let row = match coord.file {
        File::A => 0,
        File::B => 1,
        File::C => 2,
        File::D => 3,
        File::E => 4,
        File::F => 5,
        File::G => 6,
        File::H => 7,
    };
    1 << (7 - row)
}

fn led_value_for_square(square: &str) -> u8 {
    let square = square.to_uppercase();
    match square.chars().nth(1) {
        Some(row_char) => {
            let row = row_char.to_digit(10).unwrap() as usize;
            let col = square.chars().nth(0).unwrap() as usize - 'A' as usize;
            1 << (7 - col)
        }
        None => 0,
    }
}

fn encode_move_to_leds(start: &str, end: &str) -> [u8; 10] {
    let start_led = led_value_for_square(start);
    let end_led = led_value_for_square(end);

    let start_row = start.chars().nth(1).unwrap().to_digit(10).unwrap() as usize;
    let end_row = end.chars().nth(1).unwrap().to_digit(10).unwrap() as usize;

    let mut rows = [0u8; 8];
    rows[8 - start_row] |= start_led;
    rows[8 - end_row] |= end_led;

    create_led_control_message(rows)
}

fn encode_leds(positions: Vec<Coord>) -> [u8; 10] {
    let mut rows = [0u8; 8];
    for position in positions {
        let position = coord_to_str(position);
        let led = led_value_for_square(&position);
        let row = position.chars().nth(1).unwrap().to_digit(10).unwrap() as usize;
        rows[8 - row] |= led;
    }
    create_led_control_message(rows)
}

fn turn_off_all_leds() -> [u8; 10] {
    create_led_control_message([0; 8])
}

fn board_state_as_square_and_piece(board_state: &[u8]) -> Vec<SquareAndPiece> {
    let mut result = Vec::new();

    if board_state.len() != 32 {
        return result;
    }

    for i in 0..32 {
        let pair = board_state[i as usize];
        let left = pair & 0xf;
        let right = pair >> 4;

        let piece = convert_num_to_piece(left);
        let square = 63 - i * 2;
        result.push(SquareAndPiece { square, piece });

        let piece = convert_num_to_piece(right);
        let square = 63 - (i * 2 + 1);
        result.push(SquareAndPiece { square, piece });
    }
    result.sort_by(|a, b| a.square.cmp(&b.square));

    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoardState {
    board: Vec<SquareAndPiece>,
}

fn fill_board_missing_pieces(board: Vec<SquareAndPiece>) -> Vec<SquareAndPiece> {
    let mut result: Vec<SquareAndPiece> = Vec::new();
    for i in 0..64 {
        let square = board.iter().find(|x| x.square == i);
        match square {
            Some(s) => result.push(*s),
            None => result.push(SquareAndPiece {
                square: i,
                piece: None,
            }),
        }
    }
    result
}

impl BoardState {
    fn initial_board() -> BoardState {
        use PieceType::*;
        let mut board = fill_board_missing_pieces(vec![
            p(w(Rook), 0),
            p(w(Knight), 1),
            p(w(Bishop), 2),
            p(w(Queen), 3),
            p(w(King), 4),
            p(w(Bishop), 5),
            p(w(Knight), 6),
            p(w(Rook), 7),
            p(w(Pawn), 8),
            p(w(Pawn), 9),
            p(w(Pawn), 10),
            p(w(Pawn), 11),
            p(w(Pawn), 12),
            p(w(Pawn), 13),
            p(w(Pawn), 14),
            p(w(Pawn), 15),
            p(b(Pawn), 48),
            p(b(Pawn), 49),
            p(b(Pawn), 50),
            p(b(Pawn), 51),
            p(b(Pawn), 52),
            p(b(Pawn), 53),
            p(b(Pawn), 54),
            p(b(Pawn), 55),
            p(b(Rook), 56),
            p(b(Knight), 57),
            p(b(Bishop), 58),
            p(b(Queen), 59),
            p(b(King), 60),
            p(b(Bishop), 61),
            p(b(Knight), 62),
            p(b(Rook), 63),
        ]);
        board.sort_by(|a, b| a.square.cmp(&b.square));

        BoardState { board }
    }

    fn move_piece(&mut self, from: &str, to: &str) {
        let from = coord_to_square_num(str_to_coord(&from));
        let to = coord_to_square_num(str_to_coord(&to));
        let from_piece = self.board[from].piece;
        self.board[to].piece = from_piece;
        self.board[from].piece = None;
    }

    fn with_move_piece(&self, from: &str, to: &str) -> BoardState {
        let mut new_board = self.clone();
        new_board.move_piece(from, to);
        new_board
    }

    fn diff_board(&self, other: &BoardState) -> Vec<(usize, Option<Piece>, Option<Piece>)> {
        let mut result = Vec::new();
        for i in 0..64 {
            let piece = self.board[i].piece;
            let other_piece = other.board[i].piece;
            if piece != other_piece {
                result.push((i, piece, other_piece));
            }
        }
        result
    }

    fn find_incorrect_squares(&self, desired: &BoardState) -> Vec<Coord> {
        if desired.board.len() < 64 || self.board.len() < 64 {
            // return every piece as incorrect
            return (0..64).map(|x| square_num_to_coord(x)).collect();
        }
        let mut result = Vec::new();
        for i in 0..64 {
            let piece = self.board[i].piece;
            let desired_piece = desired.board[i].piece;
            if piece != desired_piece {
                result.push(square_num_to_coord(i));
            }
        }
        result
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Color {
    White,
    Black,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum PieceType {
    Queen,
    King,
    Bishop,
    Pawn,
    Knight,
    Rook,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct Piece {
    piece_type: PieceType,
    color: Color,
}

fn b(piece_type: PieceType) -> Piece {
    Piece {
        piece_type,
        color: Color::Black,
    }
}

fn w(piece_type: PieceType) -> Piece {
    Piece {
        piece_type,
        color: Color::White,
    }
}

fn convert_num_to_piece(num: u8) -> Option<Piece> {
    match num {
        0 => None,
        1 => Some(Piece {
            piece_type: PieceType::Queen,
            color: Color::Black,
        }),
        2 => Some(Piece {
            piece_type: PieceType::King,
            color: Color::Black,
        }),
        3 => Some(Piece {
            piece_type: PieceType::Bishop,
            color: Color::Black,
        }),
        4 => Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::Black,
        }),
        5 => Some(Piece {
            piece_type: PieceType::Knight,
            color: Color::Black,
        }),
        6 => Some(Piece {
            piece_type: PieceType::Rook,
            color: Color::White,
        }),
        7 => Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::White,
        }),
        8 => Some(Piece {
            piece_type: PieceType::Rook,
            color: Color::Black,
        }),
        9 => Some(Piece {
            piece_type: PieceType::Bishop,
            color: Color::White,
        }),
        10 => Some(Piece {
            piece_type: PieceType::Knight,
            color: Color::White,
        }),
        11 => Some(Piece {
            piece_type: PieceType::Queen,
            color: Color::White,
        }),
        12 => Some(Piece {
            piece_type: PieceType::King,
            color: Color::White,
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SquareAndPiece {
    square: usize, // Changed to usize for simplicity, assuming no Square type available
    piece: Option<Piece>,
}

fn p(piece: Piece, n: usize) -> SquareAndPiece {
    SquareAndPiece {
        square: n,
        piece: Some(piece),
    }
}

#[test]
fn test_board_state_as_square_and_piece() {
    let board_state = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 112,
    ];
    let result = board_state_as_square_and_piece(&board_state);
    println!("{:?}", result);
    let num_pieces = result.iter().filter(|x| x.piece.is_some()).count();
    // square 1 is white pawn
    assert_eq!(num_pieces, 1);
    assert_eq!(result[0].square, 0);
    assert_eq!(
        result[0].piece,
        Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::White
        })
    );
}

fn print_ascii_board(board: Vec<SquareAndPiece>) {
    println!("  +-----------------+");
    for row in 0..8 {
        print!("{} |", 8 - row); // Print row numbers
        for col in 0..8 {
            let index = row * 8 + col;
            let square = &board[index];
            let symbol = match &square.piece {
                Some(piece) => match piece {
                    Piece { piece_type, color } => match (piece_type, color) {
                        (PieceType::Pawn, Color::White) => 'P',
                        (PieceType::Pawn, Color::Black) => 'p',
                        (PieceType::Rook, Color::White) => 'R',
                        (PieceType::Rook, Color::Black) => 'r',
                        (PieceType::Knight, Color::White) => 'N',
                        (PieceType::Knight, Color::Black) => 'n',
                        (PieceType::Bishop, Color::White) => 'B',
                        (PieceType::Bishop, Color::Black) => 'b',
                        (PieceType::Queen, Color::White) => 'Q',
                        (PieceType::Queen, Color::Black) => 'q',
                        (PieceType::King, Color::White) => 'K',
                        (PieceType::King, Color::Black) => 'k',
                        _ => ' ',
                    },
                },
                None => ' ',
            };
            print!("{} ", symbol);
        }
        println!("|");
    }
    println!("  +-----------------+");
    println!("    a b c d e f g h  "); // Print column letters
}

async fn get_chessnut_board() -> Result<platform::Peripheral, Box<dyn Error>> {
    let manager = Manager::new().await?;
    let adapter_list = manager.adapters().await?;
    if adapter_list.is_empty() {
        eprintln!("No Bluetooth adapters found");
    }
    loop {
        for adapter in adapter_list.iter() {
            adapter
                .start_scan(ScanFilter::default())
                .await
                .expect("Can't scan BLE adapter for connected devices...");
            let peripherals = adapter.peripherals().await?;

            for peripheral in peripherals.iter() {
                let properties = peripheral.properties().await?;
                let local_name = properties
                    .unwrap()
                    .local_name
                    .unwrap_or(String::from("(peripheral name unknown)"));
                if !local_name.contains("Chessnut") {
                    continue;
                }
                peripheral.connect().await?;
                peripheral.discover_services().await?;
                for service in peripheral.services() {
                    for characteristic in service.characteristics {
                        let characterist_uuid = characteristic.uuid.to_string();
                        if characterist_uuid == WRITE {
                            peripheral
                                .write(
                                    &characteristic,
                                    &[0x21, 0x01, 0x00],
                                    btleplug::api::WriteType::WithResponse,
                                )
                                .await?;
                        }
                        if characterist_uuid == BOARD_DATA {
                            peripheral.subscribe(&characteristic).await?;
                        }
                        if characterist_uuid == OTB_DATA {
                            peripheral.subscribe(&characteristic).await?;
                        }
                        if characterist_uuid == MISC_DATA {
                            peripheral.subscribe(&characteristic).await?;
                        }
                    }
                }
                return Ok(peripheral.clone());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum File {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Coord {
    file: File,
    rank: usize,
}

fn square_num_to_coord(num: usize) -> Coord {
    Coord {
        file: match num % 8 {
            0 => File::A,
            1 => File::B,
            2 => File::C,
            3 => File::D,
            4 => File::E,
            5 => File::F,
            6 => File::G,
            7 => File::H,
            _ => File::A,
        },
        rank: num / 8,
    }
}
#[test]
fn test_square_num_to_coord() {
    assert_eq!(
        square_num_to_coord(0),
        Coord {
            file: File::A,
            rank: 0
        }
    );
    assert_eq!(
        square_num_to_coord(1),
        Coord {
            file: File::B,
            rank: 0
        }
    );
    assert_eq!(
        square_num_to_coord(8),
        Coord {
            file: File::A,
            rank: 1
        }
    );
    assert_eq!(
        square_num_to_coord(63),
        Coord {
            file: File::H,
            rank: 7
        }
    );
}

fn coord_to_square_num(coord: Coord) -> usize {
    let row = match coord.file {
        File::A => 0,
        File::B => 1,
        File::C => 2,
        File::D => 3,
        File::E => 4,
        File::F => 5,
        File::G => 6,
        File::H => 7,
    };
    row * 8 + coord.rank
}

#[test]
fn test_coord_to_square_num() {
    assert_eq!(
        coord_to_square_num(Coord {
            file: File::A,
            rank: 0
        }),
        0
    );
    assert_eq!(
        coord_to_square_num(Coord {
            file: File::A,
            rank: 1
        }),
        1
    );
    assert_eq!(
        coord_to_square_num(Coord {
            file: File::B,
            rank: 0
        }),
        8
    );
    assert_eq!(
        coord_to_square_num(Coord {
            file: File::H,
            rank: 7
        }),
        63
    );
}

fn str_to_coord(s: &str) -> Coord {
    let file = match s.chars().nth(0) {
        Some('a') => File::A,
        Some('b') => File::B,
        Some('c') => File::C,
        Some('d') => File::D,
        Some('e') => File::E,
        Some('f') => File::F,
        Some('g') => File::G,
        Some('h') => File::H,
        _ => panic!("Invalid row"),
    };
    let rank = match s.chars().nth(1) {
        Some('1') => 0,
        Some('2') => 1,
        Some('3') => 2,
        Some('4') => 3,
        Some('5') => 4,
        Some('6') => 5,
        Some('7') => 6,
        Some('8') => 7,
        _ => panic!("Invalid col"),
    };
    Coord {
        file: file,
        rank: rank,
    }
}

#[test]
fn test_str_to_coord() {
    assert_eq!(
        str_to_coord("a1"),
        Coord {
            file: File::A,
            rank: 0
        }
    );
    assert_eq!(
        str_to_coord("a2"),
        Coord {
            file: File::A,
            rank: 1
        }
    );
    assert_eq!(
        str_to_coord("b1"),
        Coord {
            file: File::B,
            rank: 0
        }
    );
    assert_eq!(
        str_to_coord("h8"),
        Coord {
            file: File::H,
            rank: 7
        }
    );
}

fn coord_to_str(coord: Coord) -> String {
    let row = match coord.file {
        File::A => "a",
        File::B => "b",
        File::C => "c",
        File::D => "d",
        File::E => "e",
        File::F => "f",
        File::G => "g",
        File::H => "h",
    };
    format!("{}{}", row, coord.rank + 1)
}

#[test]
fn test_coord_to_str() {
    assert_eq!(
        coord_to_str(Coord {
            file: File::A,
            rank: 0
        }),
        "a1"
    );
    assert_eq!(
        coord_to_str(Coord {
            file: File::A,
            rank: 1
        }),
        "a2"
    );
    assert_eq!(
        coord_to_str(Coord {
            file: File::B,
            rank: 0
        }),
        "b1"
    );
    assert_eq!(
        coord_to_str(Coord {
            file: File::H,
            rank: 7
        }),
        "h8"
    );
}

async fn send_message(
    stdin: &mut tokio::process::ChildStdin,
    message: UciMessage,
) -> Result<(), Box<dyn Error>> {
    let message = message.to_string();
    stdin.write_all(format!("{}\n", message).as_bytes()).await?;
    stdin.flush().await?;
    Ok(())
}

async fn init_game(
    stdin: &mut ChildStdin,
    lines: &mut Lines<BufReader<ChildStdout>>,
) -> Result<(), Box<dyn Error>> {
    send_message(stdin, UciMessage::Uci).await?;
    while let Some(line) = lines.next_line().await? {
        let msg: UciMessage = parse_one(&line);
        match msg {
            UciMessage::UciOk => break,
            _ => {
                println!("Unexpected message: {:?}", msg);
            }
        }
    }

    println!("Starting game loop");
    send_message(stdin, UciMessage::UciNewGame).await?;
    send_message(
        stdin,
        UciMessage::Position {
            startpos: true,
            fen: None,
            moves: vec![],
        },
    )
    .await?;

    Ok(())
}

async fn process_chessnut(
    chessnut: Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardState>>>,
) -> Result<(), String> {
    let mut notifaction_stream = chessnut.notifications().await.expect("error");
    while let Some(notification) = notifaction_stream.next().await {
        if notification.uuid.to_string() != BOARD_DATA {
            continue;
        }
        let data = notification.value;
        let board_state = BoardState {
            board: board_state_as_square_and_piece(&data[2..34]),
        };
        let mut chessnut_board_position = chessnut_board_position.lock().await;
        *chessnut_board_position = Some(board_state);
    }
    Ok(())
}

#[test]
fn test_encoding_led() {
    // Every coordate
    let mut coords = Vec::new();
    for row in 0..8 {
        for col in 0..8 {
            coords.push(Coord {
                file: match row {
                    0 => File::A,
                    1 => File::B,
                    2 => File::C,
                    3 => File::D,
                    4 => File::E,
                    5 => File::F,
                    6 => File::G,
                    7 => File::H,
                    _ => File::A,
                },
                rank: col,
            });
        }
    }
    let message = encode_leds(coords);
    assert_eq!(
        message,
        [0x0A, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    );
}

async fn wait_for_board_to_be_correct(
    writer: &Characteristic,
    chessnut: &Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardState>>>,
    desired_position: BoardState,
) -> Result<(), Box<dyn Error>> {
    loop {
        let current_chessnut_board = chessnut_board_position.lock().await;
        if let Some(chessnut_board_state) = &*current_chessnut_board {
            let incorrect = chessnut_board_state.find_incorrect_squares(&desired_position);
            if incorrect.is_empty() {
                return Ok(());
            }
            let message = encode_leds(incorrect);
            chessnut
                .write(&writer, &message, btleplug::api::WriteType::WithResponse)
                .await?;
        }
    }
}

async fn wait_for_board_to_change(
    chessnut_board_position: &Arc<Mutex<Option<BoardState>>>,
) -> Result<BoardState, Box<dyn Error>> {
    let cloned_position = chessnut_board_position.clone();
    let current_board = cloned_position.lock().await;
    let cloned_board = current_board.as_ref().unwrap().clone();
    drop(current_board);

    loop {
        let current_chessnut_board = chessnut_board_position.lock().await;
        if let Some(chessnut_board_state) = &*current_chessnut_board {
            if chessnut_board_state != &cloned_board {
                return Ok(chessnut_board_state.clone());
            }
            sleep(Duration::from_millis(100)).await;
        }
    }
}

async fn get_next_board(
    stdin: &mut ChildStdin,
    lines: &mut Lines<BufReader<ChildStdout>>,
    board_state: BoardState,
    writer: &Characteristic,
    chessnut: &Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardState>>>,
) -> Result<BoardState, Box<dyn Error>> {
    send_message(
        stdin,
        UciMessage::Go {
            time_control: None,
            search_control: Some(UciSearchControl {
                search_moves: vec![],
                mate: None,
                depth: None,
                nodes: Some(1),
            }),
        },
    )
    .await?;

    while let Some(line) = lines.next_line().await? {
        let msg: UciMessage = parse_one(&line);
        match msg {
            UciMessage::BestMove { best_move, ponder } => {
                let from = best_move.from.to_string();
                let to = best_move.to.to_string();
                let message = encode_move_to_leds(&from, &to);
                chessnut
                    .write(&writer, &message, btleplug::api::WriteType::WithResponse)
                    .await?;
                let new_board = board_state.with_move_piece(&from, &to);
                wait_for_board_to_be_correct(
                    &writer,
                    &chessnut,
                    chessnut_board_position.clone(),
                    new_board.clone(),
                )
                .await?;
                return Ok(new_board);
            }
            _ => {
                println!("Unexpected message: {:?}", msg);
            }
        }
    }
    Ok(board_state)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut board_state = BoardState::initial_board();

    let chessnut_board_position: Arc<Mutex<Option<BoardState>>> = Arc::new(Mutex::new(None));

    let chessnut = Arc::new(Box::new(get_chessnut_board().await?));

    let cloned_chessnut = chessnut.clone();
    let cloned_chessnut_board_position = chessnut_board_position.clone();
    task::spawn(
        async move { process_chessnut(cloned_chessnut, cloned_chessnut_board_position).await },
    );

    let writer = chessnut.characteristics();
    let writer = writer.iter().find(|x| x.uuid.to_string() == WRITE).unwrap();

    // run the maia1100 command so we can get our chessbot going
    let mut start_maia = Command::new("lc0")
        .arg("-w")
        .arg("/opt/homebrew/Cellar/lc0/0.30.0/libexec/maia-1100.pb.gz")
        .stdout(std::process::Stdio::piped())
        .stdin(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to start maia1100");

    let mut stdin = start_maia.stdin.take().unwrap();
    let stdout = start_maia.stdout.take().unwrap();
    let reader = io::BufReader::new(stdout);
    let mut lines = reader.lines();

    init_game(&mut stdin, &mut lines).await?;

    wait_for_board_to_be_correct(
        &writer,
        &chessnut.clone(),
        chessnut_board_position.clone(),
        board_state.clone(),
    )
    .await?;

    loop {
        let new_board = wait_for_board_to_change(&chessnut_board_position).await?;
        let new_move = new_board.diff_board(&board_state);
        if new_move.len() == 1 {
            let (from, _, _) = new_move[0];
            let (to, _, _) = new_move[1];
            let from = coord_to_str(square_num_to_coord(from));
            let to = coord_to_str(square_num_to_coord(to));
            send_message(
                &mut stdin,
                UciMessage::Position {
                    startpos: false,
                    fen: None,
                    moves: vec![UciMove {
                        from: UciSquare {
                            file: from.chars().nth(0).unwrap(),
                            rank: from.chars().nth(1).unwrap().to_digit(10).unwrap() as u8,
                        },
                        to: UciSquare {
                            file: to.chars().nth(0).unwrap(),
                            rank: to.chars().nth(1).unwrap().to_digit(10).unwrap() as u8,
                        },
                        promotion: None,
                    }],
                },
            ).await?;
        }

        let new_board = get_next_board(
            &mut stdin,
            &mut lines,
            board_state.clone(),
            &writer,
            &chessnut,
            chessnut_board_position.clone(),
        )
        .await?;
        board_state = new_board;
    }

    Ok(())
}
