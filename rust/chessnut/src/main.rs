
mod main2;
mod pgn_processor;

use std::error::Error;
use std::vec;
use tokio::io::{AsyncWriteExt, BufReader, Lines};
use tokio::process::{ChildStdin, ChildStdout};
use btleplug::api::{
    Central, Manager as _, Peripheral, ScanFilter,
};
use btleplug::platform::{self, Manager};
use vampirc_uci::{parse_one, UciMessage};

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






#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    main2::main().await?;
    // pgn_processor::main_old()?;
    // println!("{}", pgn_processor::main()?);
    Ok(())
}

// TODO: Absolutely mess but kind of working
// Biggest issue is detecting moves I make
// I need them to be valid moves
// I much just make version 2,
// pulling in rust chess to handle the board state