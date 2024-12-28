use std::{io::{self, BufRead, BufReader, Read}, thread, time::Duration};
use nonblock::NonBlockingReader;
use reqwest::blocking::Client;
use serde_json::Value;

fn main() {
    let client = Client::new();
    let request_url = "http://localhost:8000/request"; // Replace with your actual HTTP server URL
    let stdin = io::stdin();
    let mut stdin = NonBlockingReader::from_fd(stdin).unwrap();
    let mut available_bytes = Vec::new();
    loop {
        thread::sleep(Duration::from_millis(1));
        // eprintln!("Looping");
        
        stdin.read_available(&mut available_bytes).unwrap();
        if !available_bytes.is_empty() {
            let message = String::from_utf8(available_bytes.clone()).unwrap();
            available_bytes.clear();
            let response = client.post(request_url).body(message.to_string()).send();
        }
        
        // eprintln!("sent message: {}", message);
        let responses = client.get("http://localhost:8000/drain_responses").send();
        match responses {
            Ok(res) => {
                if res.status().as_u16() != 204 {
                    print!("{}", res.text().unwrap())
                }
            }
            Err(e) => eprintln!("Failed to drain responses: {}", e),
        }
    }
}