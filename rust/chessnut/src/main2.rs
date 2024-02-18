use std::{error::Error, sync::Arc};

use btleplug::{api::{Characteristic, Peripheral}, platform};
use chess::{Board, BoardBuilder, Color, File, Piece, Rank, Square, ALL_FILES, ALL_SQUARES};
use futures::StreamExt;
use tokio::{io::{self, AsyncBufReadExt}, process::Command, sync::Mutex, task, time::sleep};

use crate::{create_led_control_message, get_chessnut_board, init_game, turn_off_all_leds, BOARD_DATA, WRITE};


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

    // loop {
    //     let new_board = wait_for_board_to_change(&chessnut_board_position).await?;
    //     let new_move = new_board.diff_board(&board_state);
    //     if new_move.len() == 2 {
    //         board_state.board = new_board;
    //         let (from, _, _) = new_move[0];
    //         let (to, _, _) = new_move[1];
    //         let from = coord_to_str(square_num_to_coord(from));
    //         let to = coord_to_str(square_num_to_coord(to));
    //         send_message(
    //             &mut stdin,
    //             UciMessage::Position {
    //                 startpos: false,
    //                 fen: Some(UciFen(board_state.to_fen())),
    //                 moves: vec![UciMove {
    //                     from: UciSquare {
    //                         file: from.chars().nth(0).unwrap(),
    //                         rank: from.chars().nth(1).unwrap().to_digit(10).unwrap() as u8,
    //                     },
    //                     to: UciSquare {
    //                         file: to.chars().nth(0).unwrap(),
    //                         rank: to.chars().nth(1).unwrap().to_digit(10).unwrap() as u8,
    //                     },
    //                     promotion: None,
    //                 }],
    //             },
    //         ).await?;
    //     } else {
    //         continue;
    //     }

    //     board_state.active_color = match board_state.active_color {
    //         Color::White => Color::Black,
    //         Color::Black => Color::White,
    //     };

    //     let new_board = get_next_board(
    //         &mut stdin,
    //         &mut lines,
    //         board_state.clone(),
    //         &writer,
    //         &chessnut,
    //         chessnut_board_position.clone(),
    //     )
    //     .await?;
    //     board_state = new_board;
    //     board_state.active_color = match board_state.active_color {
    //         Color::White => Color::Black,
    //         Color::Black => Color::White,
    //     };

    //     let clear_leds = turn_off_all_leds();
    //     chessnut
    //         .write(&writer, &clear_leds, btleplug::api::WriteType::WithResponse)
    //         .await?;

    // }

    Ok(())
}