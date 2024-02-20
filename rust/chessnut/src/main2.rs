use std::{error::Error, str::FromStr, sync::Arc};

use btleplug::{api::{Characteristic, Peripheral}, platform};
use chess::{Board, BoardBuilder, BoardStatus, ChessMove, Color, File, Piece, Rank, Square, ALL_FILES, ALL_SQUARES};
use futures::StreamExt;
use tokio::{io::{self, AsyncBufReadExt, BufReader, Lines}, process::{ChildStdin, ChildStdout, Command}, sync::Mutex, task, time::sleep};
use vampirc_uci::{parse_one, UciFen, UciMessage, UciMove, UciPiece, UciSearchControl, UciSquare};

use crate::{create_led_control_message, get_chessnut_board, init_game, print_ascii_board, send_message, turn_off_all_leds, BOARD_DATA, WRITE};


trait BoardDiff {
    fn find_incorrect_squares(&self, other: &BoardBuilder) -> Vec<Square>;
}

impl BoardDiff for BoardBuilder {
    fn find_incorrect_squares(&self, other: &BoardBuilder) -> Vec<Square> {
        let mut incorrect = vec![];
        for square in ALL_SQUARES {
            if self[square] != other[square] {
                incorrect.push(square);
            }
        }
        incorrect
    }
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



fn encode_leds(positions: Vec<Square>) -> [u8; 10] {
    let mut rows = [0u8; 8];
    for position in positions {
        let position = position.to_string();
        let led = led_value_for_square(&position);
        let row = position.chars().nth(1).unwrap().to_digit(10).unwrap() as usize;
        rows[8 - row] |= led;
    }
    create_led_control_message(rows)
}

async fn wait_for_board_to_be_correct(
    writer: &Characteristic,
    chessnut: &Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>>,
    desired_position: BoardBuilder,
) -> Result<(), Box<dyn Error>> {
    loop {
        let current_chessnut_board = chessnut_board_position.lock().await;
        if let Some(chessnut_board_state) = &*current_chessnut_board {
            let incorrect = chessnut_board_state.find_incorrect_squares(&desired_position);
            if incorrect.is_empty() {
                let message = turn_off_all_leds();
                chessnut
                    .write(&writer, &message, btleplug::api::WriteType::WithResponse)
                    .await?;
                return Ok(());
            }
            let message = encode_leds(incorrect);
            chessnut
                .write(&writer, &message, btleplug::api::WriteType::WithResponse)
                .await?;
        }
    }
}

fn convert_num_to_piece(num: u8) -> Option<Piece> {

    // Value 0123456789ABC 
    // Piece .qkbpnRPrBNQK
    match num {
        0 => None,
        1 => Some(Piece::Queen),
        2 => Some(Piece::King),
        3 => Some(Piece::Bishop),
        4 => Some(Piece::Pawn),
        5 => Some(Piece::Knight),
        6 => Some(Piece::Rook),
        7 => Some(Piece::Pawn),
        8 => Some(Piece::Rook),
        9 => Some(Piece::Bishop),
        10 => Some(Piece::Knight),
        11 => Some(Piece::Queen),
        12 => Some(Piece::King),
        _ => None,
    }
}

fn convert_num_to_color(num: u8) -> Option<Color> {
    // Value 0123456789ABC 
    // Piece .qkbpnRPrBNQK
    // lowercase is black
    // uppercase is white
    match num {
        0 => None,
        1 => Some(Color::Black),
        2 => Some(Color::Black),
        3 => Some(Color::Black),
        4 => Some(Color::Black),
        5 => Some(Color::Black),
        6 => Some(Color::White),
        7 => Some(Color::White),
        8 => Some(Color::Black),
        9 => Some(Color::White),
        10 => Some(Color::White),
        11 => Some(Color::White),
        12 => Some(Color::White),
        _ => None,
        
    }
}


fn board_state_as_square_and_piece(board_state: &[u8]) -> BoardBuilder {
    let mut board_builder = BoardBuilder::new();
    for i in 0..32 {
        let pair = board_state[i as usize];
        let left = pair & 0xf;
        let right = pair >> 4;

        let square = 63 - i * 2;

        let square = unsafe { Square::new(square) };
        if let Some(piece) = convert_num_to_piece(left) {
            if let Some(color) = convert_num_to_color(left) {
                board_builder.piece(square, piece, color);
            }
        }

        let square = 63 - (i * 2 + 1);
        let square = unsafe { Square::new(square) };
        if let Some(piece) = convert_num_to_piece(right) {
            if let Some(color) = convert_num_to_color(right) {
                board_builder.piece(square, piece, color);
            }
        }
    }
    board_builder
}


#[test]
fn test_board_state_as_square_and_piece() {
    let board_state = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 112,
    ];
    let result = board_state_as_square_and_piece(&board_state);
    println!("{}", result);
    // square 1 is white pawn

    let a1 = result[Square::A1];
    assert_eq!(
        result[Square::A1],
        Some((Piece::Pawn, Color::White)),
    );
}

async fn process_chessnut(
    chessnut: Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>>,
) -> Result<(), String> {
    let mut notifaction_stream = chessnut.notifications().await.expect("error");
    while let Some(notification) = notifaction_stream.next().await {
        if notification.uuid.to_string() != BOARD_DATA {
            continue;
        }
        let data = notification.value;
        let board_state = board_state_as_square_and_piece(&data[2..34]);
        let mut chessnut_board_position = chessnut_board_position.lock().await;
        *chessnut_board_position = Some(board_state);
    }
    Ok(())
}


fn draw_board_ascii(board_builder: &BoardBuilder) {
    println!("  +-----------------+");
    for rank in (Rank::First as u8..=Rank::Eighth as u8).rev() {
        print!("{} | ", rank + 1); // Print the rank numbers on the left
        for file in ALL_FILES {
            let square = Square::make_square(Rank::from_index(rank as usize), File::from_index(file as usize));
            let piece_option = board_builder[square];
            let piece_char = match piece_option {
                Some(piece) => match piece {
                    (Piece::Pawn, Color::White) => "P",
                    (Piece::Knight, Color::White) => "N",
                    (Piece::Bishop, Color::White) => "B",
                    (Piece::Rook, Color::White) => "R",
                    (Piece::Queen, Color::White) => "Q",
                    (Piece::King, Color::White) => "K",
                    (Piece::Pawn, Color::Black) => "p",
                    (Piece::Knight, Color::Black) => "n",
                    (Piece::Bishop, Color::Black) => "b",
                    (Piece::Rook, Color::Black) => "r",
                    (Piece::Queen, Color::Black) => "q",
                    (Piece::King, Color::Black) => "k",
                },
                None => ".",
            };
            print!("{} ", piece_char);
        }
        println!("|");
    }
    println!("  +-----------------+");
    println!("    a b c d e f g h  "); // Print the file letters below
}


pub async fn main() -> Result<(), Box<dyn Error>> {
    let mut board_state = BoardBuilder::default();

    let chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>> = Arc::new(Mutex::new(None));

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
        let (new_board, new_move) = wait_for_next_move(chessnut_board_position.clone()).await;
        
        send_move_to_bot(board_state, new_move, &mut stdin, &mut lines).await?;

        board_state = new_board.clone();
        board_state.next_side();

        if new_board.status() != BoardStatus::Ongoing {
            println!("Game over");
            break;
        }
        
        let new_board = wait_for_bot_move(
            &mut stdin,
            &mut lines,
            new_board,
            &writer,
            &chessnut,
            chessnut_board_position.clone(),
        ).await?;

        board_state = new_board.clone();
        board_state.next_side();
        if board_state.status() != BoardStatus::Ongoing {
            println!("Game over");
            break;
        }

        let clear_leds = turn_off_all_leds();
        chessnut
            .write(&writer, &clear_leds, btleplug::api::WriteType::WithResponse)
            .await?;

    }

    Ok(())
}

trait FileUciExtensions {
    fn to_uci(&self) -> char;
}

impl FileUciExtensions for File {
    fn to_uci(&self) -> char {
        match self {
            File::A => 'a',
            File::B => 'b',
            File::C => 'c',
            File::D => 'd',
            File::E => 'e',
            File::F => 'f',
            File::G => 'g',
            File::H => 'h',
        }
    }
}

trait RankUciExtensions {
    fn to_uci(&self) -> u8;
}

impl RankUciExtensions for Rank {
    fn to_uci(&self) -> u8 {
        match self {
            Rank::First => 1,
            Rank::Second => 2,
            Rank::Third => 3,
            Rank::Fourth => 4,
            Rank::Fifth => 5,
            Rank::Sixth => 6,
            Rank::Seventh => 7,
            Rank::Eighth => 8,
        }
    }
}

trait PieceUciExtensions {
    fn to_uci(&self) -> UciPiece;
}

impl PieceUciExtensions for Piece {
    fn to_uci(&self) -> UciPiece {
        match self {
            Piece::Pawn => UciPiece::Pawn,
            Piece::Knight => UciPiece::Knight,
            Piece::Bishop => UciPiece::Bishop,
            Piece::Rook => UciPiece::Rook,
            Piece::Queen => UciPiece::Queen,
            Piece::King => UciPiece::King,
        }
    }
}


async fn send_move_to_bot(new_board: BoardBuilder, new_move: ChessMove, stdin: &mut tokio::process::ChildStdin, lines: &mut io::Lines<io::BufReader<tokio::process::ChildStdout>>) -> Result<(), Box<dyn Error>>{
    send_message(
        stdin,
        UciMessage::Position {
            startpos: false,
            fen: Some(UciFen(new_board.to_string())),
            moves: vec![UciMove {
                from: UciSquare {
                    file: new_move.get_source().get_file().to_uci(),
                    rank: new_move.get_source().get_rank().to_uci()
                },
                to: UciSquare {
                    file: new_move.get_dest().get_file().to_uci(),
                    rank: new_move.get_dest().get_rank().to_uci()
                },
                promotion: new_move.get_promotion().map(|x| x.to_uci()),
            }],
        },
    ).await?;
    Ok(())
}

fn are_same_board(board1: &Option<BoardBuilder>, board2: &Option<BoardBuilder>) -> bool {
    if board1.is_none() || board2.is_none() {
        // This is a little backwards logically
        // but really I just want to see if I have new baord position
        // I can compare
        // I don't think we will hit this case.
        // but if we did, I couldn't compare anything so they might as well be the same.
        return true;
    }
    for square in ALL_SQUARES {
        if board1.unwrap()[square] != board2.unwrap()[square] {
            return false;
        }
    }
    true
}

async fn wait_for_next_move(chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>>) -> (BoardBuilder, ChessMove) {
    let current_position = chessnut_board_position.lock().await;
    let current_board_position = current_position.clone();
    drop(current_position);
    loop {
        let new_position = chessnut_board_position.lock().await;
        let mut new_position = new_position.clone();
        new_position.as_mut().map(|x| x.next_side());

        if !are_same_board(&new_position, &current_board_position) {
            if let Ok(current_board) = Board::try_from(current_board_position.unwrap()) {
                match Board::try_from(new_position.unwrap()) {
                    Ok(new_board) => {
                        if let Some((new_board, new_move)) = infer_move(&current_board, &new_board) {
                            let board_builder = BoardBuilder::try_from(new_board).unwrap();
                            return (board_builder, new_move);
                        } else {
                            // println!("No legal move inferred.");
                        }
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
        }
    }
}


fn infer_move(before: &Board, after: &Board) -> Option<(Board, ChessMove)> {

    // Check each possible castle scenario first
    // by manually looking at king and rook positions before and after

    if before.piece_on(Square::from_str("e1").unwrap()) == Some(Piece::King) && after.piece_on(Square::from_str("g1").unwrap()) == Some(Piece::King) {
        if before.piece_on(Square::from_str("h1").unwrap()) == Some(Piece::Rook) && after.piece_on(Square::from_str("f1").unwrap()) == Some(Piece::Rook) {
            return Some(
                (before.make_move_new(ChessMove::new(Square::from_str("e1").unwrap(), Square::from_str("g1").unwrap(), None)),
                ChessMove::new(Square::from_str("e1").unwrap(), Square::from_str("g1").unwrap(), None)));
        }
    }
    if before.piece_on(Square::from_str("e1").unwrap()) == Some(Piece::King) && after.piece_on(Square::from_str("c1").unwrap()) == Some(Piece::King) {
        if before.piece_on(Square::from_str("a1").unwrap()) == Some(Piece::Rook) && after.piece_on(Square::from_str("d1").unwrap()) == Some(Piece::Rook) {
            return Some(
                (before.make_move_new(ChessMove::new(Square::from_str("e1").unwrap(), Square::from_str("c1").unwrap(), None)),
                ChessMove::new(Square::from_str("e1").unwrap(), Square::from_str("c1").unwrap(), None)));
        }
    }
    if before.piece_on(Square::from_str("e8").unwrap()) == Some(Piece::King) && after.piece_on(Square::from_str("g8").unwrap()) == Some(Piece::King) {
        if before.piece_on(Square::from_str("h8").unwrap()) == Some(Piece::Rook) && after.piece_on(Square::from_str("f8").unwrap()) == Some(Piece::Rook) {
            return Some(
                (before.make_move_new(ChessMove::new(Square::from_str("e8").unwrap(), Square::from_str("g8").unwrap(), None)),
                ChessMove::new(Square::from_str("e8").unwrap(), Square::from_str("g8").unwrap(), None)));
        }
    }
    if before.piece_on(Square::from_str("e8").unwrap()) == Some(Piece::King) && after.piece_on(Square::from_str("c8").unwrap()) == Some(Piece::King) {
        if before.piece_on(Square::from_str("a8").unwrap()) == Some(Piece::Rook) && after.piece_on(Square::from_str("d8").unwrap()) == Some(Piece::Rook) {
            return Some(
                (before.make_move_new(ChessMove::new(Square::from_str("e8").unwrap(), Square::from_str("c8").unwrap(), None)),
                ChessMove::new(Square::from_str("e8").unwrap(), Square::from_str("c8").unwrap(), None)));
        }
    }
    
    for from_square in ALL_SQUARES {
        for to_square in ALL_SQUARES {

            if from_square == to_square {
                continue;
            }

            if before.piece_on(from_square).is_none() {
                continue;
            }
            if after.piece_on(to_square).is_none() {
                continue;
            }


            let before_color = before.color_on(from_square).unwrap();
            let after_color = after.color_on(to_square).unwrap();
            if before_color != after_color {
                continue;
            }

            if from_square.get_file() == File::E && from_square.get_rank() == Rank::Second {
                if to_square.get_file() == File::E && to_square.get_rank() == Rank::Fourth {
                    println!("Found e2 to e4");
                }
            }
            let piece_before = before.piece_on(from_square).unwrap();
            let piece_after = after.piece_on(to_square).unwrap();
            let mut promotion = None;
            if piece_before == Piece::Pawn && (to_square.get_rank() == Rank::First || to_square.get_rank() == Rank::Eighth) {
                promotion = Some(piece_after);
            }
            let move_ = ChessMove::new(from_square, to_square, promotion);
            let after_move = before.make_move_new(move_);

            if before.legal(move_) && are_same_board(&Some(BoardBuilder::from(after_move)), &Some(BoardBuilder::from(after))) {
                return Some((before.make_move_new(move_), move_));
            }
        }
    }
    None
}


async fn wait_for_bot_move(
    stdin: &mut ChildStdin,
    lines: &mut Lines<BufReader<ChildStdout>>,
    board_state: BoardBuilder,
    writer: &Characteristic,
    chessnut: &Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>>,
) -> Result<BoardBuilder, Box<dyn Error>> {
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
                println!("Best move: {:?}", best_move);
                let from = best_move.from.to_string();
                let to = best_move.to.to_string();
                let message = encode_leds(vec![Square::from_str(&from).unwrap(), Square::from_str(&to).unwrap()]);
                chessnut
                    .write(&writer, &message, btleplug::api::WriteType::WithResponse)
                    .await?;
                let board = Board::try_from(board_state.clone()).unwrap();
                let move_ = ChessMove::new(
                    Square::from_str(&from).unwrap(),
                    Square::from_str(&to).unwrap(),
                    best_move.promotion.map(|x| match x {
                        UciPiece::Queen => Piece::Queen,
                        UciPiece::Rook => Piece::Rook,
                        UciPiece::Bishop => Piece::Bishop,
                        UciPiece::Knight => Piece::Knight,
                        UciPiece::Pawn => Piece::Pawn,
                        UciPiece::King => Piece::King,
                        
                    })
                );
                let new_board = board.make_move_new(move_);
                let mut new_board = BoardBuilder::try_from(new_board).unwrap();
                new_board.next_side();

                send_move_to_bot(board_state, move_, stdin, lines).await?;
                wait_for_board_to_be_correct(
                    &writer,
                    &chessnut,
                    chessnut_board_position,
                    new_board,
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


trait BoardExtensions {
    fn next_side(&mut self);
    fn status(&self) -> BoardStatus;
}

impl BoardExtensions for BoardBuilder {
    fn next_side(&mut self) {
        let current_side = self.get_side_to_move();
        self.side_to_move(if current_side == Color::White {
            Color::Black
        } else {
            Color::White
        });
    }

    fn status(&self) -> BoardStatus {
        let board = Board::try_from(self.clone()).unwrap();
        board.status()
    }
}