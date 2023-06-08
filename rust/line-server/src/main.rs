use log::LevelFilter;
use simplelog::{CombinedLogger, TermLogger, Config, TerminalMode, ColorChoice, WriteLogger};
use sync_file::{ReadAt, SyncFile};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Bytes, Read};
use std::str::from_utf8;
use std::sync::Arc;

const MESSAGE_PREFIX : &str = "OK\r\n";
const MESSAGE_SUFFIX: &str = "\r\n";


/// Finds new lines and builds a vector of (start, end) byte indices for each line.
/// The end index is exclusive.
/// If the file is empty, returns a single line with start and end both 0.
/// There are definitely more efficient ways to do this, but this is simple and works.
/// If we really wanted to make this faster, we could use SIMD to search for newlines.
/// We could also structure it for autovectorization by rust, but probably overkill here
// TODO: Add tests for parse_lines
pub fn parse_lines<T: Read>(chars: Bytes<std::io::BufReader<T>>) -> Vec<(usize, usize)> {
    let mut line_start = 0;
    let mut line_range = vec![];
    let mut last_line_end = 0;
    for (line_end, byte) in chars.enumerate() {
        if let Ok(b'\n') = byte {
            line_range.push((line_start, line_end));
            line_start = line_end + 1;
        }
        last_line_end = line_end;
    }

    line_range.push((line_start, last_line_end + 1));
    if line_range.is_empty() {
        // 
        line_range.push((0, 0));
    }
    line_range
}

#[derive(Debug)]
enum Message {
    Line(usize),
    Quit,
    Shutdown,
}

impl Message {
    /// Parses the bytes of the message
    /// into a the expect format. Case sensitive.
    fn parse_message(message: &[u8]) -> Option<Message> {
        let message = from_utf8(message).ok()?;
        let message = message.trim();
        if message == "QUIT" {
            Some(Message::Quit)
        } else if message == "SHUTDOWN" {
            Some(Message::Shutdown)
        } else if message.starts_with("GET") {
            // I wish rust had a substring method that returned an option
            if message.len() < "GET".len() + 1 {
                return None;
            }
            let line_number = message["GET".len() + 1..].parse::<usize>().ok()?;
            Some(Message::Line(line_number))
        } else {
            None
        }
    }
}

struct Server {
    synced_file: SyncFile,
    new_lines_to_byte: Arc<Vec<(usize, usize)>>,
    longest_line: usize,
    last_client_id: usize,
}

impl Server {
    fn new(file_path: &str) -> Self {
        // Using sync file here so we don't have to worry about concurrent access
        // TODO: Gracefully handle file not existing
        let synced_file = SyncFile::open(file_path).unwrap();
        let buf_reader = BufReader::new(synced_file.clone());
        let new_lines_to_byte = Arc::new(parse_lines(buf_reader.bytes()));
        let longest_line = new_lines_to_byte
            .iter()
            .map(|(start, end)| end - start)
            .max()
            .unwrap_or(0);
        Server {
            synced_file,
            new_lines_to_byte,
            longest_line,
            last_client_id: 0,
        }
    }

    fn next_client_id(&mut self) -> usize {
        self.last_client_id += 1;
        self.last_client_id
    }

    fn make_connection(&mut self, socket: tokio::net::TcpStream) -> Connection {
        let client_id = self.next_client_id();
        Connection {
            client_id,
            socket,
            synced_file: self.synced_file.clone(),
            new_lines_to_byte: self.new_lines_to_byte.clone(),
            message_buffer: vec![0; 64],
            // Set our buffer to be the longest line + the prefix and suffix
            // so that we never have to allocate inside the loop
            output_buffer: vec![0; self.longest_line + MESSAGE_PREFIX.len() + MESSAGE_SUFFIX.len()],
        }
    }
}

struct Connection {
    client_id: usize,
    synced_file: SyncFile,
    new_lines_to_byte: Arc<Vec<(usize, usize)>>,
    socket: tokio::net::TcpStream,
    message_buffer: Vec<u8>,
    output_buffer: Vec<u8>,
}

#[derive(Debug)]
enum MessageError {
    UnparsedMessage,
    Disconnect,
    Unknown,
}

impl Connection {
    async fn receive_message(&mut self) -> Result<Message, MessageError> {
        // TODO: If we want to be robust here, we get no guarantees about
        // the end of this message really being the end of the message in our system.
        // In quick testing it works out. But we should probably be more robust here.
        let bytes_read = self.socket.read(&mut self.message_buffer).await;
        if let Err(e) = bytes_read {
            if e.kind() == tokio::io::ErrorKind::ConnectionAborted {
                return Err(MessageError::Disconnect);
            }
            // TODO: Robustness, check the error kind to be sure it's a disconnect
            log::info!("Client disconnected {}", self.client_id);
            return Err(MessageError::Unknown);
        }
        let bytes_read = bytes_read.unwrap();
        if bytes_read == 0 {
            log::info!("Client disconnected {}", self.client_id);
            return Err(MessageError::Disconnect);
        }
        Message::parse_message(&self.message_buffer[0..bytes_read]).ok_or(MessageError::UnparsedMessage)
    }
    async fn return_err(&mut self) {
        log::warn!("Error for client {}", self.client_id);
        self.socket
            .write_all("ERR\r\n".as_bytes())
            .await
            .unwrap_or_else(|_| {
                log::error!("Error writing to socket {}", self.client_id);
            });
    }
    async fn handle_message(&mut self, message: &Message) -> Result<(), Box<dyn Error>> {
        match message {
            Message::Line(line_number) => {
                if let Some((start, end)) =
                    self.new_lines_to_byte.get(line_number.saturating_sub(1))
                {
                    let buffer_start = MESSAGE_PREFIX.len();
                    let line_end = buffer_start + (end - start);
                    // Fine with unwrap here because we know these are valid indices
                    self.synced_file
                        .read_exact_at(&mut self.output_buffer[buffer_start..line_end], *start as u64)
                        .unwrap();

                    // Kind of ugly, but saves us from allocating stuff
                    let output = &mut self.output_buffer[0..line_end + MESSAGE_SUFFIX.len()];
                    output[0..MESSAGE_PREFIX.len()].copy_from_slice(MESSAGE_PREFIX.as_bytes());
                    output[line_end..line_end + MESSAGE_SUFFIX.len()]
                        .copy_from_slice(MESSAGE_SUFFIX.as_bytes());

                    self.socket.write_all(output).await?;
                    Ok(())
                } else {
                    self.socket
                        .write_all("ERR\r\n".as_bytes())
                        .await?;
                    Ok(())
                }
            }
            Message::Quit => {
                log::info!("Quiting client {}", self.client_id);
                self.socket.shutdown().await?;
                Ok(())
            }
            Message::Shutdown => {
                log::info!("Shutting down from client {}", self.client_id);
                std::process::exit(0);
            }
        }
    }
}

fn setup_logging() {
    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Info, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Info, Config::default(), File::create("line_server.log").unwrap()),
        ]
    ).unwrap();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    setup_logging();
    
    let addr = "127.0.0.1:10497";
    let listener = TcpListener::bind(&addr).await?;
    log::info!("Listening on: {}", addr);

    let file_path = std::env::args().nth(1).unwrap_or("sample.txt".to_string());
    println!("Using file: {}", file_path);

    let mut server = Server::new(&file_path);
    loop {
        let (socket, _) = listener.accept().await?;

        let mut connection = server.make_connection(socket);
        log::info!("Client connected {}", connection.client_id);

        tokio::spawn(async move {
            loop {

                let message = connection.receive_message().await;
                if let Err(e) = message {
                    match e {
                        MessageError::UnparsedMessage |  MessageError::Unknown  => {
                            // TODO: Log what was received
                            connection.return_err().await;
                            continue;
                        }
                        MessageError::Disconnect => {
                            break;
                        }
                    }
                }
                let message = message.unwrap();
                if let Err(e) = connection.handle_message(&message).await {
                    log::error!("Error handling message from client {}: {}", connection.client_id, e.to_string());
                    continue;
                }

                log::info!("Client {} received message {:?}", connection.client_id, message);
            }
        });
    }
}
