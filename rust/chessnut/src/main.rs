mod pgn_processor;
use std::{error::Error, str::FromStr, sync::Arc};

use crate::pgn_processor::OpeningBook;
use btleplug::api::{Central, Characteristic, Manager as _, Peripheral, ScanFilter};
use btleplug::platform::{self, Manager};
use chess::{
    Board, BoardBuilder, BoardStatus, ChessMove, Color, File, Piece, Rank, Square, ALL_FILES,
    ALL_SQUARES,
};
use clipboard::{ClipboardContext, ClipboardProvider};
use futures::StreamExt;
use rand::Rng;
use tokio::{
    io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{ChildStdin, ChildStdout, Command},
    sync::Mutex,
    task,
};
use vampirc_uci::{parse_one, UciFen, UciMessage, UciMove, UciPiece, UciSearchControl, UciSquare};

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

fn turn_off_all_leds() -> [u8; 10] {
    create_led_control_message([0; 8])
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

async fn send_message(
    stdin: &mut tokio::process::ChildStdin,
    message: UciMessage,
) -> Result<(), Box<dyn Error>> {
    let message = message.to_string();
    println!("Sending message: {}", message);
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
        Some(_) => {
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
) -> Result<BoardBuilder, Box<dyn Error>> {
    loop {
        let current_chessnut_board = chessnut_board_position.lock().await;
        if let Some(chessnut_board_state) = &*current_chessnut_board {
            let incorrect = chessnut_board_state.find_incorrect_squares(&desired_position);

            let mut number_of_queens = 0;
            for square in ALL_SQUARES {
                if chessnut_board_state[square] == Some((Piece::Queen, Color::White)) {
                    number_of_queens += 1;
                }
            }

            if number_of_queens == 2 {
                let desired_position = chessnut_board_state.clone();
                let message = turn_off_all_leds();
                chessnut
                    .write(&writer, &message, btleplug::api::WriteType::WithResponse)
                    .await?;
                return Ok(desired_position);
            }

            if incorrect.is_empty() {
                let message = turn_off_all_leds();
                chessnut
                    .write(&writer, &message, btleplug::api::WriteType::WithResponse)
                    .await?;
                return Ok(desired_position);
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

    assert_eq!(result[Square::A1], Some((Piece::Pawn, Color::White)),);
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

#[allow(dead_code)]
fn draw_board_ascii(board_builder: &BoardBuilder) {
    println!("  +-----------------+");
    for rank in (Rank::First as u8..=Rank::Eighth as u8).rev() {
        print!("{} | ", rank + 1); // Print the rank numbers on the left
        for file in ALL_FILES {
            let square = Square::make_square(
                Rank::from_index(rank as usize),
                File::from_index(file as usize),
            );
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

pub async fn start_process() -> Result<(), Box<dyn Error>> {
    let chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>> = Arc::new(Mutex::new(None));

    let chessnut = Arc::new(Box::new(get_chessnut_board().await?));

    let cloned_chessnut = chessnut.clone();
    let cloned_chessnut_board_position = chessnut_board_position.clone();
    task::spawn(
        async move { process_chessnut(cloned_chessnut, cloned_chessnut_board_position).await },
    );

    let openings_book = OpeningBook::open("trie3.txt")?;

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

    loop {
        let mut board_state = BoardBuilder::default();
        init_game(&mut stdin, &mut lines).await?;

        let desired_position = wait_for_board_to_be_correct(
            &writer,
            &chessnut.clone(),
            chessnut_board_position.clone(),
            board_state.clone(),
        )
        .await?;

        let mut is_initial_position = true;

        if !are_same_board(&Some(desired_position), &Some(board_state)) {
            let desired_position =
                wait_for_two_queens(&chessnut.clone(), chessnut_board_position.clone()).await;
            send_message(&mut stdin, UciMessage::UciNewGame).await?;
            send_message(
                &mut stdin,
                UciMessage::Position {
                    startpos: false,
                    fen: Some(UciFen(desired_position.to_string())),
                    moves: vec![],
                },
            )
            .await?;
            is_initial_position = false;
            board_state = desired_position.clone();
        }

        let color =
            wait_for_color_chosen(&chessnut.clone(), chessnut_board_position.clone()).await?;

        if !is_initial_position {
            let color =
                wait_for_color_chosen(&chessnut.clone(), chessnut_board_position.clone()).await?;
            board_state.side_to_move(color);
        }

        wait_for_board_to_be_correct(
            &writer,
            &chessnut.clone(),
            chessnut_board_position.clone(),
            board_state.clone(),
        )
        .await?;

        let how_many_openings_from_book = if is_initial_position {
            rand::thread_rng().gen_range(2..10)
        } else {
            0
        };
        println!(
            "How many openings from book: {}",
            how_many_openings_from_book
        );
        let mut moves: Vec<ChessMove> = vec![];

        if color == Color::Black && board_state.get_side_to_move() == Color::White {
            let (new_board, chess_move) = wait_for_bot_move(
                &mut stdin,
                &mut lines,
                &openings_book,
                how_many_openings_from_book,
                &moves,
                board_state.clone(),
                &writer,
                &chessnut,
                chessnut_board_position.clone(),
            )
            .await?;
            moves.push(chess_move.clone());
            board_state = new_board.clone();
            board_state.next_side();
        }

        loop {
            let next_state = wait_for_next_move(board_state, chessnut_board_position.clone()).await;

            if next_state.is_err() {
                // TODO: Print this properly with different starting positions and all that
                moves.add_to_clipboard();
                println!("Game over");
                break;
            }

            let (new_board, new_move) = next_state.unwrap();

            moves.push(new_move.clone());
            send_move_to_bot(board_state, new_move, &mut stdin, &mut lines).await?;

            board_state = new_board.clone();
            board_state.next_side();

            if new_board.status() != BoardStatus::Ongoing {
                moves.add_to_clipboard();
                println!("Game over");
                break;
            }

            let (new_board, chess_move) = wait_for_bot_move(
                &mut stdin,
                &mut lines,
                &openings_book,
                how_many_openings_from_book,
                &moves,
                new_board,
                &writer,
                &chessnut,
                chessnut_board_position.clone(),
            )
            .await?;

            moves.push(chess_move.clone());

            board_state = new_board.clone();
            board_state.next_side();
            if board_state.status() != BoardStatus::Ongoing {
                moves.add_to_clipboard();
                println!("Game over");
                break;
            }

            let clear_leds = turn_off_all_leds();
            chessnut
                .write(&writer, &clear_leds, btleplug::api::WriteType::WithResponse)
                .await?;
        }
    }
}

async fn wait_for_two_queens(
    _chessnut: &Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>>,
) -> BoardBuilder {
    loop {
        let new_position = chessnut_board_position.lock().await;
        let mut number_of_queens = 0;
        if let Some(new_position) = &*new_position {
            for square in ALL_SQUARES {
                if new_position[square] == Some((Piece::Queen, Color::White)) {
                    number_of_queens += 1;
                }
            }
        }
        if number_of_queens == 1 {
            return new_position.clone().unwrap();
        }
    }
}

async fn wait_for_color_chosen(
    _chessnut: &Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>>,
) -> Result<Color, Box<dyn Error>> {
    // TODO: Make lights be all fancy
    loop {
        let new_position = chessnut_board_position.lock().await;
        let mut found_white_king = false;
        let mut found_black_king = false;
        if let Some(new_position) = &*new_position {
            for square in ALL_SQUARES {
                if new_position[square] == Some((Piece::King, Color::White)) {
                    found_white_king = true;
                }
                if new_position[square] == Some((Piece::King, Color::Black)) {
                    found_black_king = true;
                }
            }
        }
        if !found_white_king {
            return Ok(Color::White);
        }
        if !found_black_king {
            return Ok(Color::Black);
        }
    }
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

trait ClipBoardExtensions {
    fn add_to_clipboard(&self);
}

impl ClipBoardExtensions for Vec<ChessMove> {
    fn add_to_clipboard(&self) {
        let mut string = String::new();
        for move_ in self {
            string.push_str(&format!("{} ", move_));
        }
        let mut clipboard: ClipboardContext = clipboard::ClipboardProvider::new().unwrap();
        clipboard.set_contents(string.clone()).unwrap();

        println!("{}", string);
    }
}

async fn send_move_to_bot(
    old_board: BoardBuilder,
    new_move: ChessMove,
    stdin: &mut tokio::process::ChildStdin,
    _lines: &mut io::Lines<io::BufReader<tokio::process::ChildStdout>>,
) -> Result<(), Box<dyn Error>> {
    send_message(
        stdin,
        UciMessage::Position {
            startpos: false,
            fen: Some(UciFen(old_board.to_string())),
            moves: vec![UciMove {
                from: UciSquare {
                    file: new_move.get_source().get_file().to_uci(),
                    rank: new_move.get_source().get_rank().to_uci(),
                },
                to: UciSquare {
                    file: new_move.get_dest().get_file().to_uci(),
                    rank: new_move.get_dest().get_rank().to_uci(),
                },
                promotion: new_move.get_promotion().map(|x| x.to_uci()),
            }],
        },
    )
    .await?;
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

async fn wait_for_next_move(
    board_state: BoardBuilder,
    chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>>,
) -> Result<(BoardBuilder, ChessMove), GameState> {
    let original_board_position = Some(board_state.clone());
    let mut has_sent_error = false;
    loop {
        let new_position = chessnut_board_position.lock().await;
        let mut new_position = new_position.clone();
        // The new position is one where the current side has moved
        new_position
            .as_mut()
            .map(|x| x.side_to_move(board_state.next_side_new().get_side_to_move()));

        if !are_same_board(&new_position, &original_board_position) {
            let mut original_board_position = original_board_position.clone().unwrap();
            original_board_position.side_to_move(board_state.get_side_to_move());
            let resigned = check_for_resign(&new_position.unwrap());
            if let Err(e) = resigned {
                return Err(e);
            }

            if let Ok(current_board) = Board::try_from(original_board_position) {
                match Board::try_from(new_position.unwrap()) {
                    Ok(new_board) => {
                        if let Some((new_board, new_move)) = infer_move(&current_board, &new_board)
                        {
                            let board_builder = BoardBuilder::try_from(new_board).unwrap();
                            return Ok((board_builder, new_move));
                        } else {
                            // println!("No legal move inferred.");
                        }
                    }
                    Err(e) => {
                        if !has_sent_error {
                            has_sent_error = true;
                            println!("Error: {}", e);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
enum GameState {
    Resign,
}

fn check_for_resign(current_board_position: &BoardBuilder) -> Result<(), GameState> {
    let mut white_king_found = false;
    let mut black_king_found = false;
    for square in ALL_SQUARES {
        if current_board_position[square] == Some((Piece::King, Color::White)) {
            white_king_found = true;
        }
        if current_board_position[square] == Some((Piece::King, Color::Black)) {
            black_king_found = true;
        }
    }
    if !white_king_found && !black_king_found {
        return Err(GameState::Resign);
    }
    Ok(())
}

fn infer_move(before: &Board, after: &Board) -> Option<(Board, ChessMove)> {
    // Check each possible castle scenario first
    // by manually looking at king and rook positions before and after
    if before.piece_on(Square::from_str("e1").unwrap()) == Some(Piece::King)
        && after.piece_on(Square::from_str("g1").unwrap()) == Some(Piece::King)
    {
        if before.piece_on(Square::from_str("h1").unwrap()) == Some(Piece::Rook)
            && after.piece_on(Square::from_str("f1").unwrap()) == Some(Piece::Rook)
        {
            return Some((
                before.make_move_new(ChessMove::new(
                    Square::from_str("e1").unwrap(),
                    Square::from_str("g1").unwrap(),
                    None,
                )),
                ChessMove::new(
                    Square::from_str("e1").unwrap(),
                    Square::from_str("g1").unwrap(),
                    None,
                ),
            ));
        }
    }
    if before.piece_on(Square::from_str("e1").unwrap()) == Some(Piece::King)
        && after.piece_on(Square::from_str("c1").unwrap()) == Some(Piece::King)
    {
        if before.piece_on(Square::from_str("a1").unwrap()) == Some(Piece::Rook)
            && after.piece_on(Square::from_str("d1").unwrap()) == Some(Piece::Rook)
        {
            return Some((
                before.make_move_new(ChessMove::new(
                    Square::from_str("e1").unwrap(),
                    Square::from_str("c1").unwrap(),
                    None,
                )),
                ChessMove::new(
                    Square::from_str("e1").unwrap(),
                    Square::from_str("c1").unwrap(),
                    None,
                ),
            ));
        }
    }
    if before.piece_on(Square::from_str("e8").unwrap()) == Some(Piece::King)
        && after.piece_on(Square::from_str("g8").unwrap()) == Some(Piece::King)
    {
        if before.piece_on(Square::from_str("h8").unwrap()) == Some(Piece::Rook)
            && after.piece_on(Square::from_str("f8").unwrap()) == Some(Piece::Rook)
        {
            return Some((
                before.make_move_new(ChessMove::new(
                    Square::from_str("e8").unwrap(),
                    Square::from_str("g8").unwrap(),
                    None,
                )),
                ChessMove::new(
                    Square::from_str("e8").unwrap(),
                    Square::from_str("g8").unwrap(),
                    None,
                ),
            ));
        }
    }
    if before.piece_on(Square::from_str("e8").unwrap()) == Some(Piece::King)
        && after.piece_on(Square::from_str("c8").unwrap()) == Some(Piece::King)
    {
        if before.piece_on(Square::from_str("a8").unwrap()) == Some(Piece::Rook)
            && after.piece_on(Square::from_str("d8").unwrap()) == Some(Piece::Rook)
        {
            return Some((
                before.make_move_new(ChessMove::new(
                    Square::from_str("e8").unwrap(),
                    Square::from_str("c8").unwrap(),
                    None,
                )),
                ChessMove::new(
                    Square::from_str("e8").unwrap(),
                    Square::from_str("c8").unwrap(),
                    None,
                ),
            ));
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
            let piece_before = before.piece_on(from_square).unwrap();
            let piece_after = after.piece_on(to_square).unwrap();
            let mut promotion = None;
            if piece_before == Piece::Pawn
                && (to_square.get_rank() == Rank::First || to_square.get_rank() == Rank::Eighth)
            {
                promotion = Some(piece_after);
            }
            let move_ = ChessMove::new(from_square, to_square, promotion);
            let after_move = before.make_move_new(move_);

            if before.legal(move_)
                && are_same_board(
                    &Some(BoardBuilder::from(after_move)),
                    &Some(BoardBuilder::from(after)),
                )
            {
                return Some((before.make_move_new(move_), move_));
            }
        }
    }
    None
}

async fn wait_for_bot_move(
    stdin: &mut ChildStdin,
    lines: &mut Lines<BufReader<ChildStdout>>,
    openings_book: &OpeningBook,
    how_many_openings_from_book: usize,
    played_moves: &Vec<ChessMove>,
    board_state: BoardBuilder,
    writer: &Characteristic,
    chessnut: &Arc<Box<platform::Peripheral>>,
    chessnut_board_position: Arc<Mutex<Option<BoardBuilder>>>,
) -> Result<(BoardBuilder, ChessMove), Box<dyn Error>> {
    if how_many_openings_from_book > 0 && (played_moves.len() / 2) <= how_many_openings_from_book {
        let board = Board::try_from(board_state.clone()).unwrap();
        if let Some(choices) = openings_book.get_choices(&played_moves) {
            let chess_move = choices.choose_weighted(&board);
            let new_board = board.make_move_new(chess_move.clone());
            let mut new_board = BoardBuilder::try_from(new_board).unwrap();
            new_board.next_side();
            send_move_to_bot(new_board, chess_move.clone(), stdin, lines).await?;
            wait_for_board_to_be_correct(&writer, &chessnut, chessnut_board_position, new_board)
                .await?;
            return Ok((new_board, chess_move.clone()));
        }
    }

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
            UciMessage::BestMove {
                best_move,
                ponder: _,
            } => {
                println!("Best move: {:?}", best_move);
                let from = best_move.from.to_string();
                let to = best_move.to.to_string();
                let message = encode_leds(vec![
                    Square::from_str(&from).unwrap(),
                    Square::from_str(&to).unwrap(),
                ]);
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
                    }),
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
                return Ok((new_board, move_));
            }
            _ => {
                // println!("Unexpected message: {:?}", msg);
            }
        }
    }
    panic!("No best move found");
}

trait BoardExtensions {
    fn next_side(&mut self);
    fn next_side_new(&self) -> BoardBuilder;
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

    fn next_side_new(&self) -> BoardBuilder {
        let mut new_board = self.clone();
        new_board.next_side();
        new_board
    }

    fn status(&self) -> BoardStatus {
        let board = Board::try_from(self.clone()).unwrap();
        board.status()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    start_process().await?;
    // pgn_processor::main_old()?;
    // println!("{}", pgn_processor::main()?);
    Ok(())
}

// TODO: Absolutely mess but kind of working
// Biggest issue is detecting moves I make
// I need them to be valid moves
// I much just make version 2,
// pulling in rust chess to handle the board state