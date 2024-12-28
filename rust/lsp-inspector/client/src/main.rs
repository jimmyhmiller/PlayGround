use std::io::{self, BufRead, BufReader, Read};
use nonblock::NonBlockingReader;
use reqwest::blocking::Client;
use serde_json::Value;

// TODO: FIx this because its wrong
fn parse_message<T: Read>(reader: &mut BufReader<T>) -> Result<Value, Box<dyn std::error::Error>> {
    loop {
        let mut headers = String::new();

        // Read headers until an empty line (\r\n)
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap() == 0 {
                // EOF reached
                return Err("EOF reached".into());
            }

            if line == "\r\n" {
                break;
            }
            headers.push_str(&line);
        }

        // Parse Content-Length from headers
        let content_length = headers
            .lines()
            .find_map(|line| {
                if line.to_lowercase().starts_with("content-length:") {
                    line.split_once(':').map(|(_, v)| v.trim().parse::<usize>().ok()).flatten()
                } else {
                    None
                }
            })
            .expect("Content-Length header is missing");

        // Read the content part of the message
        let mut content = vec![0; content_length];
        reader.read_exact(&mut content)?;
        let content = String::from_utf8(content)?;

        // Parse the JSON content
        match serde_json::from_str::<Value>(&content) {
            Ok(json) => {
                return Ok(json);
            }
            Err(e) => eprintln!("Invalid JSON content: {}", e),
        }
    }
}


fn main() {
    let client = Client::new();
    let request_url = "http://localhost:8000/request"; // Replace with your actual HTTP server URL
    let stdin = io::stdin();
    let mut stdin = NonBlockingReader::from_fd(stdin).unwrap();
    let mut available_bytes = Vec::new();
    loop {
        eprintln!("Looping");
        
        stdin.read_available(&mut available_bytes).unwrap();
        let cloned_bytes = available_bytes.clone();
        let mut reader = BufReader::new(cloned_bytes.as_slice());
        if let Ok(message) = parse_message(&mut reader) {
            available_bytes.clear();
            reader.read_to_end(&mut available_bytes).unwrap();
            // println!("Received message: {}", message);

            let response = client.post(request_url).body(message.to_string()).send();

            match response {
                Ok(res) => {
                    eprintln!("Sent message: {:?}", message.to_string());
                    eprintln!("Server responded with: {}", res.status());
                }
                Err(e) => eprintln!("Failed to send message: {}", e),
            }
        }
        let responses = client.get("http://localhost:8000/drain_responses").send();
        match responses {
            Ok(res) => {
                if res.status().as_u16() != 204 {
                    println!("{}", res.text().unwrap())
                }
            }
            Err(e) => eprintln!("Failed to drain responses: {}", e),
        }
    }
}